import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper for a linear layer.

    This module wraps a `torch.nn.Linear` layer to inject trainable low-rank matrices
    (A and B) for efficient fine-tuning. The original weights are frozen, and only
    the LoRA parameters are updated during training.

    Args:
        original_linear (nn.Linear): The original linear layer to be wrapped.
        r (int): Rank of the low-rank adaptation matrices.
        alpha (int, optional): Scaling factor for LoRA output. Defaults to `r` if not specified.

    Attributes:
        A (nn.Parameter): Low-rank weight matrix A.
        B (nn.Parameter): Low-rank weight matrix B.
        r (int): Rank of the LoRA matrices.
        alpha (int): Scaling factor applied to the LoRA output.
    """

    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r

        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))

        # Initialise A with He initialization
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)


def apply_lora(model, lora_rank):
    """Apply LoRA to query and value projections in each transformer layer of a model.

    Replaces the `q_proj` and `v_proj` layers of each attention block
    in the transformer with `LoRALinear` layers for low-rank fine-tuning.

    Args:
        model (nn.Module): The transformer model containing `.model.layers`.
        lora_rank (int): Rank to use for the LoRA adaptation.
    """
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)
