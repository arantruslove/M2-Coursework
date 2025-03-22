import torch
from accelerate import Accelerator
from tqdm import tqdm

from m2_utilities.preprocessor import Preprocessor, batch_truncate_sequence
from m2_utilities.stopping import stopping_criteria

accelerator = Accelerator()

def calc_n_tokens(decimals):
    """
    Compute the number of tokens corresponding to a predator, prey datapoint given the
    number of decimal places.
    """
    N_SPECIAL = 6  # ;., and first digits of predator and preys
    return N_SPECIAL + 2 * decimals


def gen_points(model, trajectories, n_points, decimals):
    """Perform autoregressive inference to generate 'n_points' into the future."""
    # Process dataset into tokens
    preprocessor = Preprocessor(decimals)
    input_ids = preprocessor.encode(trajectories)

    # Forecast future points
    model = accelerator.prepare(model)
    input_ids = input_ids.to(accelerator.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=1e6,
        stopping_criteria=stopping_criteria(input_ids, n_points),
        do_sample=False,
    )

    # Isolate new tokens
    output_ids = output_ids[:, input_ids.shape[1] + 1 :]
    output_ids = batch_truncate_sequence(output_ids, n_points)

    forecast_trajectories = preprocessor.decode(output_ids)[:, -n_points:, :]
    return forecast_trajectories


def predict_next_points(model, trajectories, n_predictions, decimals):
    """Predict the next point for the last 'n_predictions' points in trajectories."""

    predictions = [[] for _ in range(len(trajectories))]
    for i in tqdm(range(n_predictions)):
        forecast = gen_points(model, trajectories[:, :-n_predictions+i], 1, decimals)
        for j in range(len(forecast)):
            predictions[j].append(forecast[j][0].tolist())

    return torch.tensor(predictions)
    
def remove_nans(true_trajectories, forecast_trajectories):
    """
    Remove nan entries present in 'forecast_trajectories' from both 'true_trajectories'
    and 'forecast_trajectories'. 
    """
    mask = ~torch.isnan(forecast_trajectories[:, 0, 0])
    return true_trajectories[mask], forecast_trajectories[mask]


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


def compute_metrics(true_trajectories, forecast_trajectories):
    """
    Compute MAE and MRAE at each point for both predator and prey populations.
    """
    # Removing nan entries
    true_trajectories, forecast_trajectories = remove_nans(true_trajectories, 
                                                           forecast_trajectories)

    # Splitting into predator and prey
    pred_true = true_trajectories[:, :, 0]
    prey_true = true_trajectories[:, :, 1]
    pred_forecast = forecast_trajectories[:, :, 0]
    prey_forecast = forecast_trajectories[:, :, 1]

    # Computing metrics
    pred_maes, prey_maes, pred_mraes, prey_mraes = [], [], [], []
    for i in range(pred_true.shape[1]):
        pred_mae = compute_mae(pred_true[:, i], pred_forecast[:, i])
        prey_mae = compute_mae(prey_true[:, i], prey_forecast[:, i])
        pred_mrae = compute_mrae(pred_true[:, i], pred_forecast[:, i])
        prey_mrae = compute_mrae(prey_true[:, i], prey_forecast[:, i])

        pred_maes.append(pred_mae)
        prey_maes.append(prey_mae)
        pred_mraes.append(pred_mrae)
        prey_mraes.append(prey_mrae)

    return torch.tensor(pred_maes), torch.tensor(prey_maes), torch.tensor(pred_mraes), torch.tensor(prey_mraes)
