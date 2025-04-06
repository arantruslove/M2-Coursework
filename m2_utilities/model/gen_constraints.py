import torch
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessor,
    LogitsProcessorList,
)

from m2_utilities.data.preprocessor import valid_tokens


class MaxSemicolonCriteria(StoppingCriteria):
    """Stopping criteria that halts generation after a specified number of semicolons.

    This is used to terminate generation after generating `n_points` time steps, where
    each step is expected to end with a semicolon token.

    Additionally, generation will also stop early if an invalid token is produced.

    Args:
        input_ids (torch.Tensor): Initial input IDs to initialize per-sequence counters.
        n_points (int): Number of target data points to generate.
    """

    def __init__(self, input_ids, n_points):
        self.max_semicolons = n_points + 1
        self.n_semicolons = torch.zeros(len(input_ids), dtype=torch.int)
        self.is_invalid = torch.zeros(len(input_ids), dtype=torch.bool)

    def __call__(self, input_ids, scores, **kwargs):
        SEMICOLON_ID = 26
        # Count semicolons
        for i in range(len(input_ids)):
            last_id = input_ids[i][-1].item()
            if last_id == SEMICOLON_ID:
                self.n_semicolons[i] += 1

            if last_id not in valid_tokens:
                self.is_invalid[i] = True

        return torch.all((self.n_semicolons >= self.max_semicolons) | self.is_invalid)


class OnlyAllowedTokens(LogitsProcessor):
    """Logits processor that masks all tokens not in the allowed list.

    This ensures that during generation, only tokens representing valid numeric
    characters or delimiters (e.g., commas, semicolons) can be produced.
    """

    def __init__(self):
        self.allowed_token_ids = list(valid_tokens)

    def __call__(self, input_ids, scores):
        # Set logits of all tokens not in the allowed list to -inf
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, self.allowed_token_ids] = False
        scores = scores.masked_fill(mask, float("-inf"))
        return scores


def stopping_criteria(input_ids, n_points):
    """Create a stopping criteria list for controlling generation termination.

    Args:
        input_ids (torch.Tensor): Initial token IDs for the batch.
        n_points (int): Desired number of forecasted time steps.

    Returns:
        StoppingCriteriaList: List with the MaxSemicolonCriteria applied.
    """
    return StoppingCriteriaList([MaxSemicolonCriteria(input_ids, n_points)])


def logits_processor():
    """Create a logits processor list for filtering invalid tokens.

    Returns:
        LogitsProcessorList: List with OnlyAllowedTokens applied.
    """
    return LogitsProcessorList([OnlyAllowedTokens()])
