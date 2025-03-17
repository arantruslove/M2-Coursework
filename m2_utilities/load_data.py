import numpy as np
import h5py


def load_trajectories(data_path):
    with h5py.File(data_path, "r") as f:
        # Access the trajectories
        trajectories = f["trajectories"][:]
    return trajectories
