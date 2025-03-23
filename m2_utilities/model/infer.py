import torch
from accelerate import Accelerator
from tqdm import tqdm

from m2_utilities.data.preprocessor import Preprocessor, batch_truncate_sequence
from m2_utilities.model.stopping import stopping_criteria

accelerator = Accelerator()


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