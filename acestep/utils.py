import torch.nn as nn
import torch


def resize_and_initialize_embedding(model, tokenizer, embedding_attr_name: str, init_token_name: str):
    old_embedding = getattr(model, embedding_attr_name)
    old_num_tokens, emb_dim = old_embedding.weight.shape
    new_num_tokens = len(tokenizer)
    verse_id = tokenizer.encode(init_token_name, 'ko')[-1]

    if new_num_tokens <= old_num_tokens:
        return

    with torch.no_grad():
        verse_emb = model.lyric_embs(torch.tensor([[verse_id]], device=old_embedding.weight.device))[0]
    
    new_embedding = nn.Embedding(new_num_tokens, emb_dim, device=old_embedding.weight.device)
    with torch.no_grad():
        new_embedding.weight[:old_num_tokens] = old_embedding.weight
        new_embedding.weight[old_num_tokens:] = verse_emb.expand((new_num_tokens - old_num_tokens), -1).clone()

    setattr(model, embedding_attr_name, new_embedding)
