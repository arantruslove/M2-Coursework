def string_dp(val, decimals):
    """Convert a float to a string with a fixed number of decimal places."""
    return f"{val:.{decimals}f}"


def stringify(trajectories, decimals):
    """
    Stringify the trajectories. Uses commas to separate between predator and prey
    values. Uses semicolons to separate between time points.
    """
    # Convert to strings with a fixed number of dcimal places
    trajectories_str = []
    for trajectory in trajectories:
        trajectory_str = ""
        for predator, prey in trajectory:
            trajectory_str += (
                string_dp(predator, decimals) + "," + string_dp(prey, decimals) + ";"
            )

        # Remove final semicolon and add to the list
        trajectories_str.append(trajectory_str[:-1])

    # Remove the final semicolon and return
    return trajectories_str


def encode(trajectories, alpha, decimals):
    """
    Encode the trajectories by scaling and rounding the trajectories into a string
    representation.
    """

    # Scale trajectories by alpha
    scaled_trajectories = trajectories / alpha
    str_trajectories = stringify(scaled_trajectories, decimals)

    return str_trajectories
