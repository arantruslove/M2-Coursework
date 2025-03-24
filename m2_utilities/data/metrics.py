import torch


def count_nans(forecast_trajectories):
    """Count the number of nan trajectories in a batch of trajectories."""
    return torch.isnan(forecast_trajectories[:, 0, 0]).sum()


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
    true_trajectories, forecast_trajectories = remove_nans(
        true_trajectories, forecast_trajectories
    )

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

    return (
        torch.tensor(pred_maes),
        torch.tensor(prey_maes),
        torch.tensor(pred_mraes),
        torch.tensor(prey_mraes),
    )
