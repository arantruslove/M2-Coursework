import torch
from accelerate import Accelerator

from m2_utilities.preprocessor import Preprocessor, batch_truncate_sequence
from m2_utilities.stopping import stopping_criteria


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
    input_ids = preprocessor.encode(trajectories)

    # Forecast future points
    accelerator = Accelerator()
    input_ids = input_ids.to(accelerator.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=1e6,
        stopping_criteria=stopping_criteria(input_ids, n_forecast),
        do_sample=False,
    )

    # Trim
    output_ids = batch_truncate_sequence(output_ids, len(trajectories) + n_forecast)

    forecast_trajectories = preprocessor.decode(output_ids)[:, -n_forecast:, :]
    return forecast_trajectories


def compute_mae(true_trajectories, forecast_trajectories):
    """Compute the mean absolute error between true and forecast trajectories."""
    return torch.mean(torch.abs(true_trajectories - forecast_trajectories)).item()


def compute_mrae(true_trajectories, forecast_trajectories, epsilon=1e-8):
    """
    Compute the mean relative absolute error between true and forecast trajectories.
    """
    return torch.mean(
        torch.abs(true_trajectories - forecast_trajectories)
        / torch.abs(true_trajectories + epsilon)
    ).item()


def forecast_metrics(model, test_trajectories, n_forecast, decimals):
    """
    Forecast the next 'n_forecast' points and compute MAE and MRAE for both predator
    and prey populations.
    """
    # Forecasting next steps
    model.eval()
    forecast = forecast_points(model, test_trajectories, n_forecast, decimals)

    # Splitting into predator and prey
    pred_true = test_trajectories[:, -n_forecast:, 0]
    prey_true = test_trajectories[:, -n_forecast:, 1]

    pred_forecast = forecast[:, :, 0]
    prey_forecast = forecast[:, :, 1]

    # Computing metrics
    pred_mae = compute_mae(pred_true, pred_forecast)
    prey_mae = compute_mae(prey_true, prey_forecast)
    pred_mrae = compute_mrae(pred_true, pred_forecast)
    prey_mrae = compute_mrae(prey_true, prey_forecast)

    model.train()
    return pred_mae, prey_mae, pred_mrae, prey_mrae
