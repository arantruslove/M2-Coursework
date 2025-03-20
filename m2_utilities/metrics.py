import torch

from m2_utilities.flops import compute_flops
from m2_utilities.preprocessor import Preprocessor


def calc_n_tokens(decimals):
    """
    Compute the number of tokens corresponding to a predator, prey datapoint given the
    number of decimal places.
    """
    N_SPECIAL = 6  # ;., and first digits of predator and preys
    return N_SPECIAL + 2 * decimals


def forecast_points(model, trajectories, n_forecast, decimals):
    """Forecast the final 'n_forecast' set of points in the trajectories."""
    # Process dataset into tokens
    preprocessor = Preprocessor(decimals)
    input_token_ids = preprocessor.encode(trajectories)

    # Forecast future points
    max_tokens = n_forecast * calc_n_tokens(decimals)
    output_token_ids = model.generate(
        input_token_ids, max_new_tokens=max_tokens, do_sample=False
    )

    forecast_trajectories = preprocessor.decode(output_token_ids)[:, -n_forecast:, :]
    return forecast_trajectories


def compute_mae(true_trajectories, forecast_trajectories):
    """Compute the mean absolute error between true and forecast trajectories."""
    return torch.mean(torch.abs(true_trajectories - forecast_trajectories)).item()


def eval(model, test_loader, calc_flops=False):
    """Evaluate model loss on test dataset."""

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

    if calc_flops:
        return test_loss, flops
    return test_loss
