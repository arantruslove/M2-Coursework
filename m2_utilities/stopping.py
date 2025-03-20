import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class MaxSemicolonCriteria(StoppingCriteria):
    def __init__(self, input_ids, n_points):
        self.max_semicolons = n_points + 1
        self.n_semicolons = torch.zeros(len(input_ids), dtype=torch.int)

    def __call__(self, input_ids, scores, **kwargs):
        SEMICOLON_ID = 26
        # Count semicolons
        for i in range(len(input_ids)):
            last_id = input_ids[i][-1].item()
            if last_id == SEMICOLON_ID:
                self.n_semicolons[i] += 1

        return torch.all(self.n_semicolons >= self.max_semicolons)


def stopping_criteria(input_ids, n_points):
    return StoppingCriteriaList([MaxSemicolonCriteria(input_ids, n_points)])
