import torch
from transformers import AutoTokenizer


def get_tokenizer():
    """Get the Qwen2.5 tokenizer."""
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


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


def process_sequences(texts, tokenizer, max_length=512, stride=256):
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


def trim_sequence(tokens):
    """Truncate the sequence so that there are no incomplete timesteps."""
    # Locate the first semicolon to determine the width of a single timestep
    SEMICOLON_TOKEN = 26
    indices = torch.where(tokens == SEMICOLON_TOKEN)[0]

    first = torch.min(indices).item()
    last = torch.max(indices).item()

    if (len(tokens) + 1) % (first + 1) == 0:
        # If the final token is not a semicolon but is a complete timestep
        return tokens
    return tokens[:last]


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
