import torch
from accelerate import Accelerator
from tqdm import tqdm

from m2_utilities.flops import compute_flops


def eval(model, test_loader):
    """Evaluate the model on the test dataset and compute FLOPS.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        tuple: A tuple containing:
            - float: Average test loss.
            - int: Total FLOPS used during evaluation.
    """

    model.eval()
    flops = 0
    test_loss = 0.0
    with torch.no_grad():
        for (batch,) in test_loader:
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            test_loss += loss.item()

            flops += compute_flops(batch.shape[1], batch.shape[0], backpropagate=False)
    # Averaging over number of batches
    test_loss = test_loss / len(test_loader)

    return test_loss, flops


def train_model(
    model,
    train_loader,
    val_loader,
    eval_interval=100,
    max_steps=10000,
    learning_rate=1e-5,
    wandb=None,
):
    """Train a model with validation and FLOPS tracking.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        eval_interval (int, optional): How often (in steps) to evaluate the model. Defaults to 100.
        max_steps (int, optional): Maximum number of training steps. Defaults to 10000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-5.
        wandb (module, optional): Weights & Biases module for logging metrics. Defaults to None.

    Returns:
        None
    """

    # Configuring optimizer
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    # Prepare components with Accelerator
    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    total_flops = 0
    steps = 0

    while True:
        # Train
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Steps {steps}")

        for (batch,) in progress_bar:

            # Descent step
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            steps += 1
            progress_bar.set_postfix(loss=loss.item())

            total_flops += compute_flops(
                batch.shape[1], batch.shape[0], backpropagate=True
            )

            # Logging to wandb
            if wandb:
                wandb.log({"Loss": loss.item(), "Flops": total_flops})

            # Computing performance metrics
            if steps % eval_interval == 0:
                val_loss, eval_flops = eval(model, val_loader)
                total_flops += eval_flops
                model.train()

                # Logging to wandb
                if wandb:
                    wandb.log(
                        {
                            "Validation Loss": val_loss,
                            "Flops": total_flops,
                        }
                    )

            # End training when max_steps have been reached
            if steps > max_steps:
                return None
