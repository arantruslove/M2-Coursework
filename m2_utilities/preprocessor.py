import torch
from transformers import AutoTokenizer


def get_tokenizer():
    """Get the Qwen2.5 tokenizer."""
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


tokenizer = get_tokenizer()


def scale(trajectories):
    """
    Scale the trajectories such that the maximum value is less than 10 by dividing the
    trajectories by the maximum value.
    """
    max_val = torch.max(trajectories).item()
    if max_val >= 10.0:
        EPSILON = 0.1  # To ensure that the scaled max is striclty less than 10
        alpha = max_val / (10.0 - EPSILON)
    else:
        alpha = 1.0
    return trajectories / alpha


def string_dp(val, decimals):
    """Convert a float to a string with a fixed number of decimal places."""
    return f"{val:.{decimals}f}"


def stringify(trajectory, decimals):
    """
    Stringify the trajectory. Uses commas to separate between predator and prey
    values. Uses semicolons to separate between time points.
    """
    text = ""
    for pred, prey in trajectory:
        text += string_dp(pred, decimals) + "," + string_dp(prey, decimals) + ";"

    # Remove the final semicolon and return
    return text[:-1]


def batch_stringify(trajectories, decimals):
    """Stringify a batch of trajectories to return a list of strings."""
    texts = []
    for trajectory in trajectories:
        text = stringify(trajectory, decimals)
        texts.append(text)
    return texts


def destringify(text):
    """Convert string format predator and prey positions to numerical format."""
    num_trajectory = []
    for point in text.split(";"):
        pred, prey = point.split(",")
        pred, prey = float(pred), float(prey)
        num_trajectory.append([pred, prey])

    return torch.tensor(num_trajectory)


def batch_destringify(texts):
    """Destringify a batch of texts to return an array of numberical trajectories"""
    trajectories = []
    for text in texts:
        trajectory = destringify(text)
        trajectories.append(trajectory)

    return torch.stack(trajectories)


def batch_decode(batch_token_ids):
    """
    Decode a batch of token ids to restore the corresponding numerical trajectories.
    """
    texts = tokenizer.batch_decode(
        batch_token_ids, return_tensors="pt", add_special_tokens=False
    )
    trajectories = batch_destringify(texts)
    return trajectories


def process_sequences(texts, max_length=512, stride=256):
    """
    Tokenize a batch of texts and splits into overlapping chunks using a sliding window.
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
    def __init__(self, decimals):
        self.decimals = decimals

    def encode(self, trajectories, chunk=False, max_length=512, stride=256):
        """
        Encode from numerical trajectories to token ids. Set scaling coefficient alpha.
        Apply Chunking.
        """
        # Stringify
        texts = batch_stringify(trajectories, self.decimals)

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
        """Restore the numerical trajectories from a batch of token ids."""
        # Detokenize to text
        texts = tokenizer.batch_decode(
            batch_token_ids, return_tensors="pt", add_special_tokens=False
        )

        # Destringify
        trajectories = batch_destringify(texts)
        return trajectories


# Functions to sanitzise and validate outputs of the Qwen2.5 model
def truncate_sequence(token_ids, n_points):
    """Truncate the sequence so that there are no incomplete timesteps."""
    # Locate the first semicolon to determine the width of a single timestep
    SEMICOLON_TOKEN = 26
    count = 0
    for i in range(len(token_ids)):
        id = token_ids[i]
        if id == SEMICOLON_TOKEN:
            count += 1
            if count == n_points:
                return token_ids[:i]


def batch_truncate_sequence(batch_token_ids, n_points):
    """Apply 'truncate_sequence' across a batch of token ids."""
    trimmed_sequence = []
    for token_ids in batch_token_ids:
        trimmed = truncate_sequence(token_ids, n_points)
        trimmed_sequence.append(trimmed)
    return torch.stack(trimmed_sequence)


def validate_sequence(tokens):
    """
    Ensure that the sequence only contains valid tokens. Tokens 15-24 correspond to
    integers 0-9. 11 corresponds to ',', 13 corresponds to '.' and 26 corresponds to ';'.
    """
    VALID_TOKENS = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 11, 13, 26}

    tokens_list = tokens.tolist()
    for token in tokens_list:
        if token not in VALID_TOKENS:
            raise ValueError(f"Sequence contains '{token}' which is not a valid token.")
