import torch
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessor,
    LogitsProcessorList,
)

from m2_utilities.data.preprocessor import valid_tokens


class MaxSemicolonCriteria(StoppingCriteria):
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
    def __init__(self):
        self.allowed_token_ids = list(valid_tokens)

    def __call__(self, input_ids, scores):
        # Set logits of all tokens not in the allowed list to -inf
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, self.allowed_token_ids] = False
        scores = scores.masked_fill(mask, float("-inf"))
        return scores


def stopping_criteria(input_ids, n_points):
    return StoppingCriteriaList([MaxSemicolonCriteria(input_ids, n_points)])


def logits_processor():
    return LogitsProcessorList([OnlyAllowedTokens()])
