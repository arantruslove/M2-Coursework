def truncate_sequence(token_ids, n_points):
    """Truncate a token sequence to contain only complete timesteps.

    Args:
        token_ids (List[int] or torch.Tensor): Sequence of token IDs representing a time
            series.
        n_points (int): Number of complete timesteps to retain.

    Returns:
        Same type as input: A truncated sequence ending at the `n_points`-th semicolon.
    """
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
    """Apply `truncate_sequence` to a batch of token sequences.

    Args:
        batch_token_ids (List[torch.Tensor] or List[List[int]]): Batch of token
            sequences.
        n_points (int): Number of timesteps to retain in each sequence.

    Returns:
        List: A list of truncated sequences.
    """
    trimmed_sequence = []
    for token_ids in batch_token_ids:
        trimmed = truncate_sequence(token_ids, n_points)
        trimmed_sequence.append(trimmed)
    return trimmed_sequence
