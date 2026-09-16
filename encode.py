import tiktoken
from pathlib import Path

def encode_text(file_path: Path, model_name: str) -> list[int]:
    """
    Encodes the given text into a list of token IDs using the specified model.

    Args:
        text (str): The text to encode.
        model_name (str): The name of the model to use for encoding.

    Returns:
        list[int]: A list of token IDs representing the encoded text.
    """
    # Read the text from the specified file path
    text = Path(file_path).read_text(encoding="utf-8")  # Read the text from the file
    # Get the encoding for the specified model
    encoding = tiktoken.get_encoding(model_name)
    
    # Encode the text and return the list of token IDs
    return encoding.encode(text)