import torch
from transformers import StoppingCriteria, StoppingCriteriaList

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


def stopping_criteria(input_ids, n_points):
    return StoppingCriteriaList([MaxSemicolonCriteria(input_ids, n_points)])
