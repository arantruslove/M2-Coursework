from accelerate import Accelerator

from m2_utilities.data.preprocessor import Preprocessor
from m2_utilities.data.postprocessor import batch_truncate_sequence
from m2_utilities.model.gen_constraints import stopping_criteria, logits_processor

accelerator = Accelerator()


def gen_points(model, trajectories, n_points, decimals, restrict_tokens=False):
    """Generate future points in a time series using an autoregressive model.

    Generates 'n_points' into the future.

    Args:
        model (torch.nn.Module): The autoregressive language model (e.g., Qwen2.5).
        trajectories (np.ndarray): Input time series of shape (batch_size, seq_len, 2),
            where the last dimension corresponds to (prey, predator) values.
        n_points (int): Number of future points to generate.
        decimals (int): Number of decimal digits to preserve during preprocessing.
        restrict_tokens (bool, optional): If True, applies token-level constraints during generation.
            Defaults to False.

    Returns:
        np.ndarray: Array of shape (batch_size, n_points, 2) containing the generated
            (prey, predator) values for each input trajectory.
    """
    # Process dataset into tokens
    preprocessor = Preprocessor(decimals)
    input_ids = preprocessor.encode(trajectories)

    # Forecast future points
    model = accelerator.prepare(model)
    input_ids = input_ids.to(accelerator.device)

    # Conditionally restrict token outputs
    logits_processor_fn = None
    if restrict_tokens:
        logits_processor_fn = logits_processor()

    output_ids = model.generate(
        input_ids,
        max_new_tokens=1e6,
        stopping_criteria=stopping_criteria(input_ids, n_points),
        logits_processor=logits_processor_fn,
        do_sample=False,
    )

    # Isolate new tokens
    output_ids = output_ids[:, input_ids.shape[1] + 1 :]
    output_ids = batch_truncate_sequence(output_ids, n_points)

    forecast_trajectories = preprocessor.decode(output_ids)[:, -n_points:, :]
    return forecast_trajectories
