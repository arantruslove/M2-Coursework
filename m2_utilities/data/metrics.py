import torch


def count_nans(forecast_trajectories):
    """Count the number of NaN trajectories in a batch.

    Args:
        forecast_trajectories (torch.Tensor): Tensor of shape (batch_size, seq_len, 2)
            representing predicted [predator, prey] values.

    Returns:
        int: Number of NaN-affected trajectories.
    """
    return torch.isnan(forecast_trajectories[:, 0, 0]).sum()


def remove_nans(true_trajectories, forecast_trajectories):
    """Remove NaN-affected trajectories from both true and forecast tensors.

    Args:
        true_trajectories (torch.Tensor): Ground truth tensor of shape (batch_size, seq_len, 2).
        forecast_trajectories (torch.Tensor): Forecast tensor of the same shape.

    Returns:
        tuple: Two tensors with NaN-containing entries removed from both inputs.
    """
    mask = ~torch.isnan(forecast_trajectories[:, 0, 0])
    return true_trajectories[mask], forecast_trajectories[mask]


def compute_mae(true_trajectories, forecast_trajectories):
    """Compute Mean Absolute Error (MAE) between true and forecasted values.

    Args:
        true_trajectories (torch.Tensor): Ground truth tensor.
        forecast_trajectories (torch.Tensor): Forecast tensor.

    Returns:
        float: Mean absolute error value.
    """
    return torch.mean(torch.abs(true_trajectories - forecast_trajectories)).item()


def compute_mrae(true_trajectories, forecast_trajectories, epsilon=1e-8):
    """Compute Mean Relative Absolute Error (MRAE).

    The denominator is `true + epsilon` to avoid division by zero.

    Args:
        true_trajectories (torch.Tensor): Ground truth tensor.
        forecast_trajectories (torch.Tensor): Forecast tensor.
        epsilon (float, optional): Small value to prevent division by zero.
            Defaults to 1e-8.

    Returns:
        float: Mean relative absolute error.
    """
    return torch.mean(
        torch.abs(true_trajectories - forecast_trajectories)
        / torch.abs(true_trajectories + epsilon)
    ).item()


def compute_metrics(true_trajectories, forecast_trajectories):
    """Compute point-wise MAE and MRAE for both predator and prey trajectories.

    Args:
        true_trajectories (torch.Tensor): Ground truth of shape (batch_size, seq_len, 2).
        forecast_trajectories (torch.Tensor): Forecasted values of the same shape.

    Returns:
        tuple: Four 1D tensors, each of shape (seq_len,), containing:
            - predator MAE,
            - prey MAE,
            - predator MRAE,
            - prey MRAE.
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
