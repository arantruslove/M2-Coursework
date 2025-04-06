import torch
from transformers import AutoModelForCausalLM

from m2_utilities.data.preprocessor import get_tokenizer


def load_qwen():
    """Load the Qwen2.5-0.5B-Instruct model, freezing all layers except the bias head.

    Returns:
        tuple: A tuple containing:
            - model (AutoModelForCausalLM): The Qwen model with only the LM head bias trainable.
            - tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = get_tokenizer()

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to logits
    assert model.lm_head.bias is None
    model.lm_head.bias = torch.nn.Parameter(
        torch.zeros(model.config.vocab_size, device=model.device)
    )
    model.lm_head.bias.requires_grad = True

    return model, tokenizer
