import numpy as np
from typing import Union, List

def apply_density_metric(values: Union[List[float], np.ndarray], weight_function: str = 'linear') -> float:
  
    """
    Applies a weighted density metric to a list or array of numerical values.

    The function computes a weighted sum of the input values using the specified 
    weight function. Common weight functions include linear, square root, 
    exponential, logarithmic, and uniform weights.

    Parameters:
        values (list or np.ndarray): A list or array of numerical values.
        weight_function (str): The weight function to apply. Options are:
            - 'linear': Weights increase linearly with the index.
            - 'sqrt': Weights are the square root of the index.
            - 'exp': Weights increase exponentially with the index.
            - 'log': Weights are the natural logarithm of the index + 1.
            - 'inverse log': Weights are inversely proportional to the logarithm.
            - 'sum': Uniform weights (all ones).

    Returns:
        float: The result of the weighted density metric computation.

    Raises:
        ValueError: If `values` is empty, contains non-finite elements, 
                    or if `weight_function` is invalid.

    Example:
        >>> apply_density_metric([1, 2, 3], weight_function='sqrt')
        4.732050807568877
    """

    if values is None or len(values) == 0:
        raise ValueError("The list of values is empty or null.")
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("The list of values contains non-finite elements (NaN or infinity).")

    n = len(values)
    indices = np.arange(1, n + 1, dtype=float)

    # Define weight functions
    def linear_weight(indices):
        return indices

    def sqrt_weight(indices):
        return np.sqrt(indices)

    def exp_weight(indices):
        return np.exp(indices / np.max(indices))

    def log_weight(indices):
        return np.log(indices + 1)

    def inverse_log_weight(indices):
        return indices / (np.log(indices + 1) + 1)

    def sum_weight(indices):
        return np.ones_like(indices)

    # Dictionary of weight functions
    weight_functions = {
        'linear': linear_weight,
        'sqrt': sqrt_weight,
        'exp': exp_weight,
        'log': log_weight,
        'inverse log': inverse_log_weight,
        'sum': sum_weight
    }

    if weight_function not in weight_functions:
        raise ValueError(f"Invalid weight function. Choose from {list(weight_functions.keys())}.")

    weights = weight_functions[weight_function](indices)

    # Return the dot product of values and weights
    return np.sum(values * weights)


def apply_density_metric_df(df, column='Amplitude', weight_function='linear'):
    
    """
    Applies a density metric to a specified column in a pandas DataFrame.

    This function extracts numerical values from the specified column of the 
    DataFrame, applies the `apply_density_metric` function to compute a weighted 
    density metric, and returns the result.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to process. Default is 'Amplitude'.
        weight_function (str): The weight function to apply. See `apply_density_metric`.

    Returns:
        float: The result of the weighted density metric computation for the column.

    Raises:
        ValueError: If the specified column does not exist or contains non-numeric values.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'Amplitude': [1.0, 2.0, 3.0]})
        >>> apply_density_metric_df(df, column='Amplitude', weight_function='linear')
        14.0
    """

    if column not in df.columns:
        raise ValueError(f"The column '{column}' does not exist in the DataFrame.")
    try:
        values = df[column].dropna().astype(float).values
    except ValueError:
        raise ValueError(f"The column '{column}' contains non-numeric values.")
    return apply_density_metric(values, weight_function)





