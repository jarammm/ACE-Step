from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning import Trainer
from datetime import datetime
import argparse
import torch
import json
import matplotlib
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.text2music_dataset import Text2MusicDataset
from loguru import logger
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from acestep.apg_guidance import apg_forward, MomentumBuffer
from tqdm import tqdm
import random
import os
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.models.lyrics_utils.vocab_utils import DEFAULT_VOCAB_NAME


matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")


class Pipeline(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_workers: int = 4,
        train: bool = True,
        T: int = 1000,
        weight_decay: float = 1e-2,
        every_plot_step: int = 2000,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        timestep_densities_type: str = "logit_normal",
        ssl_coeff: float = 1.0,
        checkpoint_dir=None,
        max_steps: int = 200000,
        warmup_steps: int = 10,
        dataset_path: str = "./data/your_dataset_path",
        lora_config_path: str = None,
        adapter_name: str = "lora_adapter",
        vocab_name: str = DEFAULT_VOCAB_NAME
    ):
        super().__init__()

        self.save_hyperparameters()
        self.is_train = train
        self.T = T

        # Initialize scheduler
        self.scheduler = self.get_scheduler()

        # step 1: load model
        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        acestep_pipeline.load_checkpoint(checkpoint_dir=acestep_pipeline.checkpoint_dir,
                                         vocab_name=vocab_name)

        transformers = acestep_pipeline.ace_step_transformer.float().cpu()
        transformers.enable_gradient_checkpointing()

        assert lora_config_path is not None, "Please provide a LoRA config path"
        if lora_config_path is not None:
            try:
                from peft import LoraConfig
            except ImportError:
                raise ImportError("Please install peft library to use LoRA training")
            with open(lora_config_path, encoding="utf-8") as f:
                import json
                lora_config = json.load(f)
            lora_config = LoraConfig(**lora_config)
            transformers.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
            self.adapter_name = adapter_name

        self.transformers = transformers

        self.dcae = acestep_pipeline.music_dcae.float().cpu()
        self.dcae.requires_grad_(False)

        self.text_encoder_model = acestep_pipeline.text_encoder_model.float().cpu()
        self.text_encoder_model.requires_grad_(False)
        self.text_tokenizer = acestep_pipeline.text_tokenizer

        if self.is_train:
            self.transformers.train()

            # download first
            try:
                self.mert_model = AutoModel.from_pretrained(
                    "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
                ).eval()
            except:
                import json
                import os

                mert_config_path = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "huggingface",
                    "hub",
                    "models--m-a-p--MERT-v1-330M",
                    "blobs",
                    "14f770758c7fe5c5e8ead4fe0f8e5fa727eb6942"
                )

                with open(mert_config_path) as f:
                    mert_config = json.load(f)
                mert_config["conv_pos_batch_norm"] = False
                with open(mert_config_path, mode="w") as f:
                    json.dump(mert_config, f)
                self.mert_model = AutoModel.from_pretrained(
                    "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
                ).eval()
            self.mert_model.requires_grad_(False)
            self.resampler_mert = torchaudio.transforms.Resample(
                orig_freq=48000, new_freq=24000
            )
            self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained(
                "m-a-p/MERT-v1-330M", trust_remote_code=True
            )

            self.hubert_model = AutoModel.from_pretrained("utter-project/mHuBERT-147").eval()
            self.hubert_model.requires_grad_(False)
            self.resampler_mhubert = torchaudio.transforms.Resample(
                orig_freq=48000, new_freq=16000
            )
            self.processor_mhubert = Wav2Vec2FeatureExtractor.from_pretrained(
                "utter-project/mHuBERT-147",
                cache_dir=checkpoint_dir,
            )

            self.ssl_coeff = ssl_coeff

    def infer_mert_ssl(self, target_wavs, wav_lengths):
        # target_wavs [bs, 2, wav_len], wav_lengths [bs]
        # Input is N x 2 x T (48kHz), convert to N x T (24kHz), mono
        # 1. convert audio sr 48k stereo to 24k mono : target_wavs -> mono(mean) -> resample(48k to 24k)
        mert_input_wavs_mono_24k = self.resampler_mert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_24k = wav_lengths // 2  # 48kHz -> 24kHz

        # 2. Normalize the actual audio part
        means = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].mean()
                for i in range(bsz)
            ]
        ) # 2-1. converted mono 24k 오디오데이터에 대한 mean값
        vars = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].var()
                for i in range(bsz)
            ]
        ) # 2-2. converted mono 24k 오디오데이터에 대한 variance값
        mert_input_wavs_mono_24k = (
            mert_input_wavs_mono_24k - means.view(-1, 1)
        ) / torch.sqrt(vars.view(-1, 1) + 1e-7) # 2-3. make target_wavs N(0,1)

        # 3. get mert hidden states
        # MERT SSL constraint
        # Define the length of each chunk (5 seconds of samples)
        chunk_size = 24000 * 5  # 5 seconds, 24000 samples per second
        total_length = mert_input_wavs_mono_24k.shape[1]
        # 3-1. calculate the number of chunks (5 seconds of samples)
        num_chunks_per_audio = (actual_lengths_24k + chunk_size - 1) // chunk_size # 올림 계산

        # 4. Process chunks
        # 4-1. pad the last chunk of each audio if it's less than chunk_size
        all_chunks = []
        chunk_actual_lengths = []
        for i in range(bsz):
            audio = mert_input_wavs_mono_24k[i]
            actual_length = actual_lengths_24k[i].item()
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(
                        chunk, (0, chunk_size - len(chunk))
                    )  # Pad insufficient parts with zeros
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        # 4-2. Stack all chunks to (total_chunks=len(all_chunks), chunk_size=120000)
        all_chunks = torch.stack(all_chunks, dim=0) # [total chunks num, chunk size]

        # 5. Batch inference : get last hidden states from mert forward
        with torch.no_grad():
            # Output shape: (total_chunks, seq_len, hidden_size)
            mert_ssl_hidden_states = self.mert_model(all_chunks).last_hidden_state # [all_chunks, 374(5sec seq token), 1024]

        # 6. Calculate the number of features for each chunk (assuming model stride of 320)
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths] # list, size of total_chunks

        # 7. Trim the hidden states of each chunk : get chunk_hidden_states
        chunk_hidden_states = [
            mert_ssl_hidden_states[i, : chunk_num_features[i], :]
            for i in range(len(all_chunks))
        ] # total_chunk size of list(with [374(5sec seq token), 1024])

        # 8. Organize hidden states by each audio
        mert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[
                chunk_idx : chunk_idx + num_chunks_per_audio[i]
            ] # 8-1. audio chunks for a single audio / len(audio_chunks) = length of the audio chunks, audio_chunks[0] : torch.Size([374(5sec seq token), 1024])
            audio_hidden = torch.cat(
                audio_chunks, dim=0
            )  # 8-2. Concatenate chunks of the same audio / [concat chunk features len, 1024]
            mert_ssl_hidden_states_list.append(audio_hidden)
            chunk_idx += num_chunks_per_audio[i]

        return mert_ssl_hidden_states_list

    def infer_mhubert_ssl(self, target_wavs, wav_lengths):
        # Step 1: Preprocess audio
        # Input: N x 2 x T (48kHz, stereo) -> N x T (16kHz, mono)
        mhubert_input_wavs_mono_16k = self.resampler_mhubert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_16k = wav_lengths // 3  # Convert lengths from 48kHz to 16kHz

        # Step 2: Zero-mean unit-variance normalization (only on actual audio)
        means = torch.stack(
            [
                mhubert_input_wavs_mono_16k[i, : actual_lengths_16k[i]].mean()
                for i in range(bsz)
            ]
        )
        vars = torch.stack(
            [
                mhubert_input_wavs_mono_16k[i, : actual_lengths_16k[i]].var()
                for i in range(bsz)
            ]
        )
        mhubert_input_wavs_mono_16k = (
            mhubert_input_wavs_mono_16k - means.view(-1, 1)
        ) / torch.sqrt(vars.view(-1, 1) + 1e-7)

        # Step 3: Define chunk size for MHubert (30 seconds at 16kHz)
        chunk_size = 16000 * 30  # 30 seconds = 480,000 samples

        # Step 4: Split audio into chunks
        num_chunks_per_audio = (
            actual_lengths_16k + chunk_size - 1
        ) // chunk_size  # Ceiling division
        all_chunks = []
        chunk_actual_lengths = []

        for i in range(bsz):
            audio = mhubert_input_wavs_mono_16k[i]
            actual_length = actual_lengths_16k[i].item()
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))  # Pad with zeros
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        # Step 5: Stack all chunks for batch inference
        all_chunks = torch.stack(all_chunks, dim=0)  # Shape: (total_chunks, chunk_size)

        # Step 6: Batch inference with MHubert model
        with torch.no_grad():
            mhubert_ssl_hidden_states = self.hubert_model(all_chunks).last_hidden_state
            # Shape: (total_chunks, seq_len, hidden_size)

        # Step 7: Compute number of features per chunk (assuming model stride of 320)
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]

        # Step 8: Trim hidden states to remove padding effects
        chunk_hidden_states = [
            mhubert_ssl_hidden_states[i, : chunk_num_features[i], :]
            for i in range(len(all_chunks))
        ]

        # Step 9: Reorganize hidden states by original audio
        mhubert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[
                chunk_idx : chunk_idx + num_chunks_per_audio[i]
            ]
            audio_hidden = torch.cat(
                audio_chunks, dim=0
            )  # Concatenate chunks for this audio
            mhubert_ssl_hidden_states_list.append(audio_hidden)
            chunk_idx += num_chunks_per_audio[i]
        return mhubert_ssl_hidden_states_list

    def get_text_embeddings(self, texts, device, text_max_length=256):
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.text_encoder_model.device != device:
            self.text_encoder_model.to(device)
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs) # 프롬프트 인코더 / 
            last_hidden_states = outputs.last_hidden_state # [bs, prompt_seq_len, 768]
        attention_mask = inputs["attention_mask"] # [bs, prompt_seq_len]
        return last_hidden_states, attention_mask

    def preprocess(self, batch, train=True):
        # 0: get ssl hidden states(mert & mhubert)
        target_wavs = batch["target_wavs"] # [bs, 2, wav_len] / float
        wav_lengths = batch["wav_lengths"] # [bs] /int

        dtype = target_wavs.dtype
        bs = target_wavs.shape[0]
        device = target_wavs.device

        # SSL constraints
        mert_ssl_hidden_states = None
        mhubert_ssl_hidden_states = None
        if train:
            with torch.amp.autocast(device_type="cuda", dtype=dtype): # dtype=float32
                mert_ssl_hidden_states = self.infer_mert_ssl(target_wavs, wav_lengths) # len(mert_ssl_hidden_states)=batch size, mert_ssl_hidden_states[idx] : [idx-th audio's concat chunk features len, 1024]
                mhubert_ssl_hidden_states = self.infer_mhubert_ssl(
                    target_wavs, wav_lengths
                ) # len(mert_ssl_hidden_states)=batch size, mert_ssl_hidden_states[idx] : [idx-th audio's concat chunk features len, 768]

        # 1: text embedding
        texts = batch["prompts"]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts, device
        ) # [bs, prompt_seq_len, 768], [bs, prompt_seq_len]
        encoder_text_hidden_states = encoder_text_hidden_states.to(dtype)
        # 2. dcae embedding of target wavs and attention mask
        target_latents, _ = self.dcae.encode(target_wavs, wav_lengths) # [bs, 8, 16, mel_seq_len/8]
        attention_mask = torch.ones(
            bs, target_latents.shape[-1], device=device, dtype=dtype
        )

        speaker_embds = batch["speaker_embs"].to(dtype) # zero-vector [bs, 512]
        keys = batch["keys"] # [bs]
        lyric_token_ids = batch["lyric_token_ids"] # [bs, lyric_seq_len]
        lyric_mask = batch["lyric_masks"] # [bs, lyric_seq_len]

        # cfg : cfg 스타일 학습을 위해 각 condition 입력에 대해 확률적으로 0으로 masking
        if train:
            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.15),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device),
            ).long() # [bs]
            # N x T x 768
            encoder_text_hidden_states = torch.where(
                full_cfg_condition_mask.unsqueeze(1).unsqueeze(1).bool(),
                encoder_text_hidden_states,
                torch.zeros_like(encoder_text_hidden_states),
            ) # [bs, prompt_seq_len, 768]

            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.50),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device),
            ).long() # [bs]
            # N x 512
            speaker_embds = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                speaker_embds,
                torch.zeros_like(speaker_embds),
            ) # [bs, 512]

            # Lyrics
            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.15),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device),
            ).long() # [bs]
            lyric_token_ids = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                lyric_token_ids,
                torch.zeros_like(lyric_token_ids),
            ) # [bs, lyric_seq_len]
            lyric_mask = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                lyric_mask,
                torch.zeros_like(lyric_mask),
            ) # [bs, lyric_seq_len]

        return (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        )

    def get_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.T,
            shift=self.hparams.shift,
        )

    def configure_optimizers(self):
        trainable_params = [
            p for name, p in self.transformers.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            params=[
                {"params": trainable_params},
            ],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.8, 0.9),
        )
        max_steps = self.hparams.max_steps
        warmup_steps = self.hparams.warmup_steps  # New hyperparameter for warmup steps

        # Create a scheduler that first warms up linearly, then decays linearly
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from 0 to learning_rate
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay from learning_rate to 0
                progress = float(current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                return max(0.0, 1.0 - progress)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        self.train_dataset = Text2MusicDataset(split="train", dataset_path=self.hparams.dataset_path)
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=2,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn,
        )
    
    def val_dataloader(self):
        self.val_dataset = Text2MusicDataset(split="val", dataset_path=self.hparams.dataset_path)
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=2,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        self.test_dataset = Text2MusicDataset(split="test", dataset_path=self.hparams.dataset_path)
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=2,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate_fn,
        )


    def get_sd3_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32): # stable-diffusion3 style-model에서 given timestep에 해당하는 노이즈 강도 sigma 추출
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1) # batch-wise sigma
        return sigma

    def get_timestep(self, bsz, device):
        if self.hparams.timestep_densities_type == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            # In practice, we sample the random variable u from a normal distribution u ∼ N (u; m, s)
            # and map it through the standard logistic function
            u = torch.normal(
                mean=self.hparams.logit_mean,
                std=self.hparams.logit_std,
                size=(bsz,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(
                indices, 0, self.scheduler.config.num_train_timesteps - 1
            )
            timesteps = self.scheduler.timesteps[indices].to(device)

        return timesteps

    def run_step(self, batch, batch_idx):
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        ) = self.preprocess(batch)

        target_image = target_latents # [bs, 8, 16, seq_len of dcae]
        device = target_image.device
        dtype = target_image.dtype
        # Step 1: Generate random noise, initialize settings
        noise = torch.randn_like(target_image, device=device)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, device)

        # Add noise according to flow matching.(flow-matching euler discrete scheduler)
        sigmas = self.get_sd3_sigmas(
            timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype
        )
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image # 원본 target image에 sigmas만큼의 노이즈를 추가한 이미지

        # This is the flow-matching target for vanilla SD3.
        target = target_image

        # SSL constraints for CLAP and vocal_latent_channel2
        all_ssl_hiden_states = []
        if mert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mert_ssl_hidden_states)
        if mhubert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mhubert_ssl_hidden_states)

        # N x H -> N x c x W x H
        x = noisy_image
        # Step 5: Predict noise
        transformer_output = self.transformers(
            hidden_states=x,
            attention_mask=attention_mask, # [bs, seq_len of dcae]
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device).to(dtype),
            ssl_hidden_states=all_ssl_hiden_states, # [mert embed, mhubert embed]
        )
        model_pred = transformer_output.sample # [bs, 8, 16, seq_len of dcae]
        proj_losses = transformer_output.proj_losses # [('mert', mert_loss_float), ('m-hubert', m-hubert_loss_float)]

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_image

        # Compute loss. Only calculate loss where chunk_mask is 1 and there is no padding
        # N x T x 64
        # N x T -> N x c x W x T
        mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, target_image.shape[1], target_image.shape[2], -1)
        ) # [bs, 8, 16, seq_len of dcae]

        selected_model_pred = (model_pred * mask).reshape(bsz, -1).contiguous() # [bs, 8 * 16 * seq_len of dcae]
        selected_target = (target * mask).reshape(bsz, -1).contiguous() # [bs, 8 * 16 * seq_len of dcae]

        denoising_loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
        denoising_loss = denoising_loss.mean(1)
        denoising_loss = denoising_loss * mask.reshape(bsz, -1).mean(1)
        denoising_loss = denoising_loss.mean()

        total_proj_loss = 0.0
        for k, v in proj_losses:
            total_proj_loss += v

        if len(proj_losses) > 0:
            total_proj_loss = total_proj_loss / len(proj_losses)

        total_loss = denoising_loss + total_proj_loss * self.ssl_coeff

        # Log learning rate if scheduler exists
        if self.lr_schedulers() is not None:
            learning_rate = self.lr_schedulers().get_last_lr()[0]
        else:
            learning_rate = None
        # with torch.autograd.detect_anomaly():
        #     self.manual_backward(loss)
        return total_loss, denoising_loss, proj_losses, learning_rate
    
    def training_step(self, batch, batch_idx):
        self.plot_step(batch, batch_idx)
        total_loss, denoising_loss, proj_losses, lr = self.run_step(batch, batch_idx)
        
        self.log("train/denoising_loss", denoising_loss, on_step=True, on_epoch=False, prog_bar=True)
        for name, value in proj_losses:
            self.log(f"train/{name}_loss", value, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        if lr is not None:
            self.log("train/learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True)

        return total_loss

    
    def validation_step(self, batch, batch_idx):
        total_loss, denoising_loss, proj_losses, _ = self.run_step(batch, batch_idx)

        self.log("val/denoising_loss", denoising_loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in proj_losses:
            self.log(f"val/{name}_loss", value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    @torch.no_grad()
    def diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
    ):

        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.T,
            shift=3.0,
        )

        frame_length = int(duration * 44100 / 512 / 8) # dcae로 8배 압축하기 때문에 8로 나눠줘야 length 맞음
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps=infer_steps, device=device, timesteps=None
        )

        target_latents = randn_tensor(
            shape=(bsz, 8, 16, frame_length),
            generator=random_generators,
            device=device,
            dtype=dtype,
        )
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2, dim=0)
            encoder_text_hidden_states = torch.cat(
                [
                    encoder_text_hidden_states,
                    torch.zeros_like(encoder_text_hidden_states),
                ],
                0,
            )
            text_attention_mask = torch.cat([text_attention_mask] * 2, dim=0)

            speaker_embds = torch.cat(
                [speaker_embds, torch.zeros_like(speaker_embds)], 0
            )

            lyric_token_ids = torch.cat(
                [lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0
            )
            lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)], 0)

        momentum_buffer = MomentumBuffer()

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = target_latents
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformers(
                hidden_states=latent_model_input,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timestep,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_with_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = apg_forward(
                    pred_cond=noise_pred_with_cond,
                    pred_uncond=noise_pred_uncond,
                    guidance_scale=guidance_scale,
                    momentum_buffer=momentum_buffer,
                )

            target_latents = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=target_latents,
                return_dict=False,
                omega=omega_scale,
            )[0]

        return target_latents

    def predict_step(self, batch):
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        ) = self.preprocess(batch, train=False)

        infer_steps = 60
        guidance_scale = 15.0
        omega_scale = 10.0
        seed_num = 1234
        random.seed(seed_num)
        bsz = target_latents.shape[0]
        random_generators = [torch.Generator(device=self.device) for _ in range(bsz)]
        seeds = []
        for i in range(bsz):
            seed = random.randint(0, 2**32 - 1)
            random_generators[i].manual_seed(seed)
            seeds.append(seed)
        duration = 240  # Fixed duration (24 * 10)
        pred_latents = self.diffusion_process(
            duration=duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embds,
            lyric_token_ids=lyric_token_ids,
            lyric_mask=lyric_mask,
            random_generators=random_generators,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
        )

        audio_lengths = batch["wav_lengths"]
        sr, pred_wavs = self.dcae.decode(
            pred_latents, audio_lengths=audio_lengths, sr=48000
        )
        return {
            "target_wavs": batch["target_wavs"],
            "pred_wavs": pred_wavs,
            "keys": keys,
            "prompts": batch["prompts"],
            "candidate_lyric_chunks": batch["candidate_lyric_chunks"],
            "sr": sr,
            "seeds": seeds,
        }

    def construct_lyrics(self, candidate_lyric_chunk):
        lyrics = []
        for chunk in candidate_lyric_chunk:
            lyrics.append(chunk["lyric"])

        lyrics = "\n".join(lyrics)
        return lyrics

    def plot_step(self, batch, batch_idx):
        global_step = self.global_step
        if (
            global_step % self.hparams.every_plot_step != 0
            or self.local_rank != 0
            or torch.distributed.get_rank() != 0
            or torch.cuda.current_device() != 0
        ):
            return
        
        test_loader = self.test_dataloader()
        if isinstance(test_loader, list):
            test_loader = test_loader[0]
        try:
            batch = next(iter(test_loader))
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        except StopIteration:
            return
        
        results = self.predict_step(batch)

        target_wavs = results["target_wavs"]
        pred_wavs = results["pred_wavs"]
        keys = results["keys"]
        prompts = results["prompts"]
        candidate_lyric_chunks = results["candidate_lyric_chunks"]
        sr = results["sr"]
        seeds = results["seeds"]
        i = 0
        for key, target_wav, pred_wav, prompt, candidate_lyric_chunk, seed in zip(
            keys, target_wavs, pred_wavs, prompts, candidate_lyric_chunks, seeds
        ):
            key = key
            prompt = prompt
            lyric = self.construct_lyrics(candidate_lyric_chunk)
            key_prompt_lyric = f"# KEY\n\n{key}\n\n\n# PROMPT\n\n{prompt}\n\n\n# LYRIC\n\n{lyric}\n\n# SEED\n\n{seed}\n\n"
            log_dir = self.logger.save_dir
            save_dir = f"{log_dir}/test_results/step_{self.global_step}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torchaudio.save(
                f"{save_dir}/target_wav_{key}_{i}.wav", target_wav.float().cpu(), sr
            )
            torchaudio.save(
                f"{save_dir}/pred_wav_{key}_{i}.wav", pred_wav.float().cpu(), sr
            )
            with open(
                f"{save_dir}/key_prompt_lyric_{key}_{i}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(key_prompt_lyric)
            i += 1

class SaveLoraCallback(Callback):
    def __init__(self, adapter_name="default", every_n_steps=1000):
        self.adapter_name = adapter_name
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0 and trainer.is_global_zero:
            log_dir = trainer.logger.save_dir
            checkpoint_dir = os.path.join(log_dir, "checkpoints", f"step={step}_lora")
            os.makedirs(checkpoint_dir, exist_ok=True)
            pl_module.transformers.save_lora_adapter(checkpoint_dir, adapter_name=self.adapter_name)

def main(args):
    model = Pipeline(
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        shift=args.shift,
        max_steps=args.max_steps,
        every_plot_step=args.every_plot_step,
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        adapter_name=args.exp_name,
        lora_config_path=args.lora_config_path,
        vocab_name=args.vocab_name
    )

    lora_callback = SaveLoraCallback(
        adapter_name=args.exp_name,
        every_n_steps=args.every_n_train_steps
    )
    # add datetime str to version
    logger_callback = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + args.exp_name,
        save_dir=args.logger_dir,
    )
    
    wandb_logger = WandbLogger(
    project=args.wandb_project,
    name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + args.exp_name,
    save_dir=args.logger_dir,
    log_model=False,
    settings=wandb.Settings(start_method="fork")
)

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="ddp_find_unused_parameters_true",
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[lora_callback],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
        val_check_interval=args.val_check_interval,
    )

    trainer.fit(
        model,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--num_nodes", type=int, default=1)
    args.add_argument("--shift", type=float, default=3.0)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--num_workers", type=int, default=8)
    args.add_argument("--epochs", type=int, default=-1)
    args.add_argument("--max_steps", type=int, default=4000)
    args.add_argument("--every_n_train_steps", type=int, default=500)
    args.add_argument("--dataset_path", type=str, default="./lora_dataset")
    args.add_argument("--exp_name", type=str, default="speaker_emb")
    args.add_argument("--precision", type=str, default="32")
    args.add_argument("--accumulate_grad_batches", type=int, default=1)
    args.add_argument("--devices", type=int, default=1)
    args.add_argument("--logger_dir", type=str, default="./exps/speaker_emb/")
    args.add_argument("--ckpt_path", type=str, default=None)
    args.add_argument("--checkpoint_dir", type=str, default=None)
    args.add_argument("--gradient_clip_val", type=float, default=0.5)
    args.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    args.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    args.add_argument("--every_plot_step", type=int, default=1000)
    args.add_argument("--val_check_interval", type=int, default=500)
    args.add_argument("--lora_config_path", type=str, default="config/lora_config_v1.json")
    args.add_argument('--wandb_project', type=str, default="pansori-gen")
    args.add_argument('--wandb_name', type=str, default="speaker_emb")
    args.add_argument('--vocab_name', type=str, default="vocab")
    args = args.parse_args()
    main(args)
