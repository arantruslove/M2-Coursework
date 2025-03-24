import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
import wandb

from experiments.config import hyperparam_options
from m2_utilities.model.qwen import load_qwen
from m2_utilities.model.lora import apply_lora
from m2_utilities.model.train import train_model
from m2_utilities.data.load_data import load_trajectories
from m2_utilities.data.preprocessor import Preprocessor


def main():
    # Obtaining hyperparameters based off of 'config_no'
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_no", type=int, required=True)
    args = parser.parse_args()
    params = hyperparam_options[args.config_no]
    print(params)

    # Load and preprocess data
    trajectories = load_trajectories("data/lotka_volterra_data.h5")

    # Split
    N_VAL = 850
    train_trajectories = trajectories[: params["train_size"]]
    val_trajectories = trajectories[N_VAL:]

    # Preprocess
    preprocessor = Preprocessor(params["decimals"])
    train_ids = preprocessor.encode(
        train_trajectories, chunk=True, max_length=params["chunk_size"]
    )
    val_ids = preprocessor.encode(val_trajectories)

    # Initialise data loaders
    train_dataset = TensorDataset(train_ids)
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    val_dataset = TensorDataset(val_ids)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    # Obtain model and apply LoRA
    model, _ = load_qwen()
    apply_lora(model, params["lora_rank"])

    # Initialise wandb
    wandb.init(project="M2 Coursework", config=params)

    # Train the model and save
    try:
        train_model(
            model,
            train_loader,
            val_loader,
            learning_rate=params["lr"],
            eval_interval=params["eval_interval"],
            max_steps=params["max_steps"],
            wandb=wandb,
        )
    except KeyboardInterrupt:
        print("Saving model...")
    finally:
        torch.save(model.state_dict(), f"models/state_dict_{args.config_no}.pt")


if __name__ == "__main__":
    main()
