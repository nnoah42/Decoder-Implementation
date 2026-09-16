import torch
import torch.nn as nn

def embed(numdim: int, encodings: list[list[list[int]]], vocab_size: int):
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=numdim,
    )
    input_ids = torch.tensor(encodings, dtype=torch.long)
    return embedding(input_ids)