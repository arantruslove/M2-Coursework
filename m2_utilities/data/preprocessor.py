import torch
from transformers import AutoTokenizer

# Valid token ids corresponding to integers 0 through 9, dots, commas, and semicolons
valid_tokens = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 11, 13, 26}


def get_tokenizer():
    """Load the Qwen2.5 tokenizer.

    Returns:
        AutoTokenizer: The tokenizer for Qwen2.5-0.5B-Instruct.
    """
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


tokenizer = get_tokenizer()


def scale(trajectories):
    """Scale trajectory values to ensure the maximum value is less than 10.

    Args:
        trajectories (torch.Tensor): Input tensor of shape (batch_size, seq_len, 2).

    Returns:
        tuple: Scaled trajectories and the scaling factor (alpha).
    """
    max_val = torch.max(trajectories).item()
    if max_val >= 10.0:
        EPSILON = 0.1  # To ensure that the scaled max is strictly less than 10
        alpha = max_val / (10.0 - EPSILON)
    else:
        alpha = 1.0
    return trajectories / alpha, alpha


def string_dp(val, decimals):
    """Convert a float to a string with fixed decimal precision.

    Args:
        val (float): Value to format.
        decimals (int): Number of decimal places.

    Returns:
        str: Formatted string.
    """
    return f"{val:.{decimals}f}"


def stringify(trajectory, decimals):
    """Convert a 2D trajectory tensor to a formatted string.

    Args:
        trajectory (torch.Tensor): Tensor of shape (seq_len, 2).
        decimals (int): Decimal precision for formatting.

    Returns:
        str: String representation of the trajectory.
    """
    text = ""
    for pred, prey in trajectory:
        text += string_dp(pred, decimals) + "," + string_dp(prey, decimals) + ";"

    # Remove the final semicolon and return
    return text[:-1]


def batch_stringify(trajectories, decimals):
    """Apply `stringify` to a batch of trajectories.

    Args:
        trajectories (torch.Tensor): Tensor of shape (batch_size, seq_len, 2).
        decimals (int): Decimal precision.

    Returns:
        list[str]: List of stringified trajectories.
    """
    texts = []
    for trajectory in trajectories:
        text = stringify(trajectory, decimals)
        texts.append(text)
    return texts


def destringify(text):
    """Convert a trajectory string back to numerical tensor format.

    Args:
        text (str): Encoded trajectory string.

    Returns:
        torch.Tensor: Tensor of shape (seq_len, 2).
    """
    num_trajectory = []
    for point in text.split(";"):
        pred, prey = point.split(",")
        pred, prey = float(pred), float(prey)
        num_trajectory.append([pred, prey])

    return torch.tensor(num_trajectory)


def batch_destringify(texts):
    """Convert a batch of encoded strings into numerical trajectories.

    Args:
        texts (list[str]): List of trajectory strings.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, seq_len, 2).
    """
    trajectories = []
    for text in texts:
        try:
            trajectory = destringify(text)
            trajectories.append(trajectory)
        except:
            trajectories.append(None)

    # Replacing nones with tensors of nan values of the same shape
    for trajectory in trajectories:
        if trajectory is not None:
            shape = trajectory.shape
            break

    for i in range(len(trajectories)):
        if trajectories[i] is None:
            trajectories[i] = torch.full(shape, float("nan"))
    return torch.stack(trajectories)


def batch_decode(batch_token_ids):
    """Decode token IDs to numerical trajectories.

    Args:
        batch_token_ids (torch.Tensor): Token IDs to decode.

    Returns:
        torch.Tensor: Numerical trajectories of shape (batch_size, seq_len, 2).
    """
    texts = tokenizer.batch_decode(
        batch_token_ids, return_tensors="pt", add_special_tokens=False
    )
    trajectories = batch_destringify(texts)
    return trajectories


def process_sequences(texts, max_length=512, stride=256):
    """Tokenize and chunk a batch of strings using a sliding window.

    Args:
        texts (list[str]): List of input strings.
        max_length (int): Maximum token length for each chunk.
        stride (int): Step size for sliding window.

    Returns:
        torch.Tensor: Tensor of shape (num_chunks, max_length).
    """
    all_input_ids = []
    for text in texts:
        # Apply Qwen's tokenization scheme to the text:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]

        # Create sliding windows to further divide the data into chunks:
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat(
                    [
                        chunk,
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id),
                    ]
                )
            all_input_ids.append(chunk)
    return torch.stack(all_input_ids)


class Preprocessor:
    """A utility class for encoding and decoding predator-prey trajectories."""

    def __init__(self, decimals):
        """
        Args:
            decimals (int): Number of decimal places to use for string conversion.
        """
        self.decimals = decimals
        self.alpha = None

    def encode(self, trajectories, chunk=False, max_length=512, stride=256):
        """Convert numerical trajectories into token IDs.

        Args:
            trajectories (torch.Tensor): Tensor of shape (batch_size, seq_len, 2).
            chunk (bool): Whether to apply token chunking. Defaults to False.
            max_length (int): Max token length if chunking is enabled.
            stride (int): Sliding window stride if chunking is enabled.

        Returns:
            torch.Tensor: Encoded token IDs.
        """
        # Scale
        scaled_trajectories, self.alpha = scale(trajectories)

        # Stringify
        texts = batch_stringify(scaled_trajectories, self.decimals)

        # Include chunking
        if chunk:
            chunked_token_ids = process_sequences(texts, max_length, stride)
            return chunked_token_ids

        # No chunking
        batch_token_ids = tokenizer(
            texts, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        return batch_token_ids

    def decode(self, batch_token_ids):
        """Decode token IDs back into scaled numerical trajectories.

        Args:
            batch_token_ids (torch.Tensor): Tensor of token IDs.

        Returns:
            torch.Tensor: Decoded and rescaled trajectories.
        """

        # Detokenize to text
        texts = tokenizer.batch_decode(
            batch_token_ids, return_tensors="pt", add_special_tokens=False
        )

        # Destringify
        trajectories = batch_destringify(texts)

        # Unscale
        trajectories = trajectories * self.alpha
        return trajectories
