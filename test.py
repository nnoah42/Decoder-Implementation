import tiktoken

from encode import encode_text as encode
from blocks import create_block_batches
from token_embedding import token_embedding

encoding = tiktoken.get_encoding("gpt2")
tokenized_data = encode("the-verdict.txt", "gpt2")
blocks = create_block_batches(tokenized_data, block_size=8)
token_embeddings = token_embedding(numdim=8, encodings=blocks, vocab_size=encoding.n_vocab)