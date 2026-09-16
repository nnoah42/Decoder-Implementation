import tiktoken

from encode import encode_text as encode
from blocks import create_block_batches
from input_embedding import InputEmbedding

encoding = tiktoken.get_encoding("gpt2")
tokenized_data = encode("the-verdict.txt", "gpt2")
blocks = create_block_batches(tokenized_data, block_size=8)
input_embedding = InputEmbedding(numdim=8, block_size=8, vocab_size=encoding.n_vocab)
embeddings = input_embedding(blocks)
