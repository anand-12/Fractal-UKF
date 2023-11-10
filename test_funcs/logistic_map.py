def logistic_map(x, r):
    """
    Generate the next value in a Logistic Map time series.

    Args:
        x (list or array): History of previous values.
        r (float): Parameter controlling the behavior of the map.

    Returns:
        float: Next value in the Logistic Map series.
    """
    return r * x[-1] * (1 - x[-1])
