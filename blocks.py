def create_block_batches(data: list[int], block_size: int) -> list[list[int]]:
    """
    Splits the given list of token IDs into blocks of the specified size.

    Args:
        data (list[int]): The list of token IDs to split into blocks.
        block_size (int): The size of each block.

    Returns:
        list[list[int]]: A list of blocks, where each block is a list of token IDs.
    """
    # Create blocks by slicing the data into chunks of block_size
    blocks = [data[i:i + block_size] for i in range(0, len(data), block_size)]
    batches = []
    # Group blocks into batches of 8
    for i in range(0, len(blocks), 8):
        batch = blocks[i:i + 8]
        batches.append(batch)
    # Filter out incomplete batches
    batches = [batch for batch in batches if len(batch) == 8]
    return batches