import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

from experiments.config import hyperparam_options
from m2_utilities.flops import compute_flops_gen
from m2_utilities.model.qwen import load_qwen
from m2_utilities.model.lora import apply_lora
from m2_utilities.model.infer import gen_points
from m2_utilities.model.train import eval
from m2_utilities.data.preprocessor import Preprocessor
from m2_utilities.data.load_data import load_trajectories
from m2_utilities.data.metrics import compute_metrics, count_nans


def main():
    # Obtaining hyperparameters based off of 'config_no'
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_no", type=int, required=True)
    parser.add_argument("--restrict_tokens", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()
    params = hyperparam_options[args.config_no]

    # Load validation data
    trajectories = load_trajectories("data/lotka_volterra_data.h5")
    val_trajectories = trajectories[850:]

    # Obtain token ids
    preprocessor = Preprocessor(params["decimals"])
    val_ids = preprocessor.encode(val_trajectories)
    val_dataset = TensorDataset(val_ids)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    # Load model
    model, _ = load_qwen()
    if args.config_no != 0:
        apply_lora(model, params["lora_rank"])
        state_dict = torch.load(f"models/state_dict_{args.config_no}.pt")
        model.load_state_dict(state_dict)
    model.eval()

    # Compute validation loss
    accelerator = Accelerator()
    model, val_loader = accelerator.prepare(model, val_loader)
    val_loss, flops_val = eval(model, val_loader)

    # Generate future prediction points
    # Context length of 40
    predict_40 = gen_points(
        model,
        val_trajectories[:, :40],
        n_points=20,
        decimals=params["decimals"],
        restrict_tokens=args.restrict_tokens,
    )
    nans_40 = count_nans(predict_40)
    pred_maes_40, prey_maes_40, pred_mraes_40, prey_mraes_40 = compute_metrics(
        val_trajectories[:, 40:60], predict_40
    )
    flops_40 = compute_flops_gen(40, 20, batch_size=150)

    # Context length of 80
    predict_80 = gen_points(
        model,
        val_trajectories[:, :80],
        n_points=20,
        decimals=params["decimals"],
        restrict_tokens=args.restrict_tokens,
    )
    nans_80 = count_nans(predict_80)
    pred_maes_80, prey_maes_80, pred_mraes_80, prey_mraes_80 = compute_metrics(
        val_trajectories[:, 80:], predict_80
    )
    flops_80 = compute_flops_gen(80, 20, batch_size=150)

    # Print metrics
    print(f"Validation Loss: {val_loss:.4e}")

    print("")
    print(f"Context Length: 40")
    print(f"Invalid Forcasts: {nans_40}")
    print(f"Pred MAE: {torch.mean(pred_maes_40)}")
    print(f"Prey MAE: {torch.mean(prey_maes_40)}")
    print(f"Pred MRAE: {torch.mean(pred_mraes_40)}")
    print(f"Prey MRAE: {torch.mean(prey_mraes_40)}")

    print("")
    print(f"Context Length: 80")
    print(f"Invalid Forecasts: {nans_80}")
    print(f"Pred MAE: {torch.mean(pred_maes_80)}")
    print(f"Prey MAE: {torch.mean(prey_maes_80)}")
    print(f"Pred MRAE: {torch.mean(pred_mraes_80)}")
    print(f"Prey MRAE: {torch.mean(prey_mraes_80)}")

    print("")
    print(f"Total Flops: {flops_val + flops_40 + flops_80:.4e}")

    # Save metrics
    if args.save:
        os.makedirs(f"output_data/config_{args.config_no}", exist_ok=True)
        torch.save(pred_maes_40, f"output_data/config_{args.config_no}/pred_maes_40.pt")
        torch.save(prey_maes_40, f"output_data/config_{args.config_no}/prey_maes_40.pt")
        torch.save(pred_maes_80, f"output_data/config_{args.config_no}/pred_maes_80.pt")
        torch.save(prey_maes_80, f"output_data/config_{args.config_no}/prey_maes_80.pt")
        torch.save(predict_80, f"output_data/config_{args.config_no}/predict_80.pt")

if __name__ == "__main__":
    main()
