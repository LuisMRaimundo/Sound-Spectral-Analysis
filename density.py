# density.py

import numpy as np
import pandas as pd
from typing import Union, List, Any


def apply_density_metric(
    values: Union[List[float], np.ndarray],
    weight_function: str = 'linear'
) -> float:
    """
    Computes a weighted density metric for a list or array of values using a
    chosen weight function.

    Supported weight functions:
      - 'linear': Weights increase linearly with the index.
      - 'sqrt': Weights are the square root of the index.
      - 'exp': Weights increase exponentially with the (normalized) index.
      - 'log': Weights are the natural logarithm of index + 1.
      - 'inverse log': Weights are inversely proportional to log(index + 1).
      - 'sum': Uniform weights (all ones).

    Args:
        values (Union[List[float], np.ndarray]):
            A list or array of numeric values on which to compute the metric.
        weight_function (str, optional):
            The name of the weight function to apply. Defaults to 'linear'.

    Returns:
        float: The computed weighted metric.

    Raises:
        ValueError:
            - If 'values' is empty or None.
            - If 'values' contains non-finite elements (NaN, inf).
            - If 'weight_function' is not supported.

    Examples:
        >>> apply_density_metric([1, 2, 3], weight_function='sqrt')
        4.732050807568877
    """
    # Validate input
    if values is None or len(values) == 0:
        raise ValueError("The list or array of values is empty or None.")

    values_arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values_arr)):
        raise ValueError("The list of values contains non-finite (NaN or inf) elements.")

    n = len(values_arr)
    indices = np.arange(1, n + 1, dtype=float)

    # Define all weight functions
    def linear_weight(idx: np.ndarray) -> np.ndarray:
        return idx

    def sqrt_weight(idx: np.ndarray) -> np.ndarray:
        return np.sqrt(idx)

    def exp_weight(idx: np.ndarray) -> np.ndarray:
        # Normalizes idx by its maximum, then exponentiates
        return np.exp(idx / idx[-1])  # idx[-1] = max(idx) for 1..n

    def log_weight(idx: np.ndarray) -> np.ndarray:
        return np.log(idx + 1)

    def inverse_log_weight(idx: np.ndarray) -> np.ndarray:
        return idx / (np.log(idx + 1) + 1)

    def sum_weight(idx: np.ndarray) -> np.ndarray:
        return np.ones_like(idx)

    # Map user string to function
    weight_functions = {
        'linear': linear_weight,
        'sqrt': sqrt_weight,
        'exp': exp_weight,
        'log': log_weight,
        'inverse log': inverse_log_weight,
        'sum': sum_weight
    }

    if weight_function not in weight_functions:
        raise ValueError(
            f"Invalid weight function '{weight_function}'. "
            f"Choose from {list(weight_functions.keys())}."
        )

    # Compute weighted density
    weights = weight_functions[weight_function](indices)
    return float(np.sum(values_arr * weights))


def apply_density_metric_df(
    df: 'pd.DataFrame',
    column: str = 'Amplitude',
    weight_function: str = 'linear'
) -> float:
    """
    Applies a weighted density metric to a specified column in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str, optional):
            The name of the column to process. Defaults to 'Amplitude'.
        weight_function (str, optional):
            The weight function to apply to the column’s values. Defaults to 'linear'.

    Returns:
        float: The computed weighted density metric for the column.

    Raises:
        ValueError:
            - If the specified column does not exist.
            - If the column contains non-numeric values or is empty.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'Amplitude': [1.0, 2.0, 3.0]})
        >>> apply_density_metric_df(df, column='Amplitude', weight_function='linear')
        14.0
    """
    # Ensure the column exists
    if column not in df.columns:
        raise ValueError(f"The column '{column}' does not exist in the DataFrame.")

    # Extract numeric data
    try:
        values = df[column].dropna().astype(float).values
    except ValueError:
        raise ValueError(f"The column '{column}' contains non-numeric values.")

    # Apply metric computation
    return apply_density_metric(values, weight_function)





