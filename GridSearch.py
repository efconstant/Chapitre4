import itertools
import numpy as np
from sklearn.model_selection import cross_val_score


def grid_search(estimator, param_grid, X, y):
    """
    Grid search implementation.

    Parameters:
    - estimator: object type that implements the "fit" and "predict" methods
    - param_grid: dict of string to sequence, where each sequence is a list of valid values for a parameter
    - X: numpy array, input features
    - y: numpy array, target values

    Returns:
    - best_estimator: estimator object with the best parameters
    - best_score: float, the mean cross-validated score of the best_estimator
    """
    # Create a list of dictionaries where each dictionary is a set of parameters
    param_list = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    print(param_list)

    # Initialize variables to keep track of the best parameters and the best score
    best_score = float("-inf")
    best_estimator = None

    for i, params in enumerate(param_list):
        # Set the estimator's parameters
        estimator.set_params(**params)

        # Fit the estimator on the data and evaluate it using cross-validation
        scores = cross_val_score(estimator, X, y, cv=5)
        mean_score = np.mean(scores)
        print(mean_score)

        # Update the best parameters and the best score if the current mean score is better
        if mean_score > best_score:
            best_score = mean_score
            best_estimator = estimator

    return best_estimator, best_score
