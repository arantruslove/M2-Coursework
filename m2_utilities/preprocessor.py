import torch
from transformers import AutoTokenizer
import h5py

# Initialising the tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_trajectories(data_path):
    with h5py.File(data_path, "r") as f:
        # Access the trajectories
        trajectories = f["trajectories"][:]

        # Converting to a torch tensor
        trajectories = torch.from_numpy(trajectories)
    return trajectories


def string_dp(val, decimals):
    """Convert a float to a string with a fixed number of decimal places."""
    return f"{val:.{decimals}f}"


def stringify(trajectory, decimals):
    """
    Stringify the trajectory. Uses commas to separate between predator and prey
    values. Uses semicolons to separate between time points.
    """

    stringified = ""
    for pred, prey in trajectory:
        stringified += string_dp(pred, decimals) + "," + string_dp(prey, decimals) + ";"

    # Remove the final semicolon and return
    return stringified[:-1]


def destringify(stringified):
    """Convert string format predator and prey positions to numerical format."""
    num_trajectory = []
    for point in stringified.split(";"):
        pred, prey = point.split(",")
        pred, prey = float(pred), float(prey)
        num_trajectory.append([pred, prey])

    return torch.tensor(num_trajectory)


def tokenize(trajectory, decimals):
    """Convert numerical trajectory to Qwen2.5 token representations."""
    stringified = stringify(trajectory, decimals)
    tokens = tokenizer(stringified, return_tensors="pt")["input_ids"][0]
    return tokens


def detokenize(tokens):
    """Convert Qwen2.5 tokens to numerical trajectories."""
    stringified = tokenizer.decode(tokens, skip_special_tokens=True)
    trajectory = destringify(stringified)
    return trajectory


def load_and_preprocess(data_path, alpha, decimals):
    """Load and stringify the trajectories dataset."""
    # Load and scale
    trajectories = load_trajectories(data_path)
    trajectories /= alpha

    # Stringify
    texts = []
    for trajectory in trajectories:
        text = stringify(trajectory, decimals)
        texts.append(text)
    return texts


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
