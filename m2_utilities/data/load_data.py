import torch
import h5py


def load_trajectories(data_path):
    """Load predator-prey trajectories from an HDF5 file and convert to a torch tensor.

    Args:
        data_path (str): Path to the HDF5 file containing the trajectory data.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, sequence_length, 2) with the
            loaded trajectories.
    """
    with h5py.File(data_path, "r") as f:
        # Access the trajectories
        trajectories = f["trajectories"][:]

        # Converting to a torch tensor
        trajectories = torch.from_numpy(trajectories)
    return trajectories
