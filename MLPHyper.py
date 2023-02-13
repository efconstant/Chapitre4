import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class NeuralNetMLP(BaseEstimator):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', alpha=0.0001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha

    def fit(self, X, y):
        # Fit the model using the training data
        print("fit...")
        # ...

    def predict(self, X):
        # Predict the target values using the model
        print("predict...")
        # ...

    def score(self, X, y):
        # Calculate the accuracy score of the model using the given data
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


# Define the parameter grid to search over
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,)],
    'activation': ['logistic', 'tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a GridSearchCV object to search over the parameter grid
grid_search = GridSearchCV(NeuralNetMLP(), param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(X, y)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters:", best_params)
print("Best score:", best_score)
