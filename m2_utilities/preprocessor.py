import torch
from transformers import AutoTokenizer

# Initialising the tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def string_dp(val, decimals):
    """Convert a float to a string with a fixed number of decimal places."""
    return f"{val:.{decimals}f}"


def stringify(trajectories, decimals):
    """
    Stringify the trajectories. Uses commas to separate between predator and prey
    values. Uses semicolons to separate between time points.
    """
    # Convert to strings with a fixed number of dcimal places
    stringified = []
    for trajectory in trajectories:
        trajectory_str = ""
        for pred, prey in trajectory:
            trajectory_str += (
                string_dp(pred, decimals) + "," + string_dp(prey, decimals) + ";"
            )

        # Remove final semicolon and add to the list
        stringified.append(trajectory_str[:-1])

    # Remove the final semicolon and return
    return stringified


def destringify(stringified):
    """Convert string format predator and prey positions to numerical format."""
    destringified = []
    for str_trajectory in stringified:
        num_trajectory = []
        for elem in str_trajectory.split(";"):
            pred, prey = elem.split(",")
            pred, prey = float(pred), float(prey)
            num_trajectory.append([pred, prey])
        destringified.append(num_trajectory)

    return torch.tensor(destringified)


def tokenize(trajectories, decimals):
    """Convert numerical trajectories to Qwen2.5 token representations."""
    stringified = stringify(trajectories, decimals)
    tokens = tokenizer(stringified, return_tensors="pt")["input_ids"]
    return tokens


def detokenize(tokens):
    """Convert Qwen2.5 tokens to numerical trajectories."""
    stringified = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    num_trajectories = destringify(stringified)
    return num_trajectories
