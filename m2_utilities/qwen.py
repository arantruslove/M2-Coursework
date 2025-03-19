import torch
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from tqdm import tqdm
from m2_utilities.preprocessor import get_tokenizer


def load_qwen():
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


def train(model, train_loader, val_loader, max_steps=10000, learning_rate=1e-5):

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    # Prepare components with Accelerator
    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    steps = 0
    while steps < max_steps:
        # Train
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Steps {steps}")
        for (batch,) in progress_bar:

            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            steps += 1

            progress_bar.set_postfix(loss=loss.item())
            if steps > max_steps:
                break

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                outputs = model(batch, labels=batch)
                loss = outputs.loss
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")
