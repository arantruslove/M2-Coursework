"""
Functions to truncate the Qwen2.5 model outputs to the desired number of time points.
"""


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
    return token_ids


def batch_truncate_sequence(batch_token_ids, n_points):
    """Apply 'truncate_sequence' across a batch of token ids."""
    trimmed_sequence = []
    for token_ids in batch_token_ids:
        trimmed = truncate_sequence(token_ids, n_points)
        trimmed_sequence.append(trimmed)
    return trimmed_sequence
