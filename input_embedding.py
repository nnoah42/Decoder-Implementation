import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, numdim: int, block_size: int, vocab_size: int):
        super().__init__()
        self.numdim = numdim
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=numdim
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=numdim,
        )
    def forward(self, input_ids: torch.tensor):
        token_ids = torch.tensor(input_ids, dtype=torch.long)
        token_embeddings = self.token_embedding(token_ids)

        sequence_length = token_ids.shape[-1]
        position_ids = torch.arange(sequence_length)
        positional_embeddings = self.positional_embedding(position_ids)

        return token_embeddings + positional_embeddings
        