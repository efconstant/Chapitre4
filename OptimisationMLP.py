import itertools
import numpy as np
from sklearn.model_selection import cross_val_score
import MLP
import os
import struct

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# On récupère le dataset MNIST et on le preprocess
X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='t10k')

# Dictionnaire contenant l'ensemble des valeurs des hyperparamètres
param_grid = {'n_output': [10],
              'n_features': [X_train.shape[1]],
              'n_hidden': [50],
              'l2': [0.1],
              'l1': [0.0],
              'epochs': [5, 10, 15],
              'eta': [0.001],
              'alpha': [0.001],
              'decrease_const': [0.00001],
              'minibatches': [50],
              'shuffle': [True],
              'random_state': [1]}


# Grid search- implémentation manuelle ------

# Une liste avec l'ensembles des combinaisons possibles
param_list = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
print(param_list)

# Variable type tableau pour stocker les scores
best_accuracy = float("-inf")
best_combinaison= None

# Naviguer à travers toutes les combinaisons possibles
for i, params in enumerate(param_list):
    print("itération ", i, "--> ", params)
    # nn = MLP.NeuralNetMLP(params)
    nn = MLP.NeuralNetMLP(n_output=params['n_output'],
                          n_features=params['n_features'],
                          n_hidden=params['n_hidden'],
                          l2=params['l2'],
                          l1=params['l1'],
                          epochs=params['epochs'],
                          eta=params['eta'],
                          alpha=params['alpha'],
                          decrease_const=params['decrease_const'],
                          minibatches=params['minibatches'],
                          shuffle=params['shuffle'],
                          random_state=params['random_state'])
    # On lance l'entrainement
    nn.fit(X_train, y_train, print_progress=True)
    # Evaluation du score par combinaison
    y_train_pred = nn.predict(X_train)
    accuracy = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print("itération ", i, " - ", 'Training accuracy : %.2f%%' % (accuracy * 100))
    mean_accuracy= np.mean(accuracy)
    print(mean_accuracy)

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_combinaison= params

print("Le meilleur score est : %.2f%%" % (best_accuracy * 100))
print("La meilleure combinaison est :", best_combinaison)




#     # estimator.set_params(**params)
#
#     # Fit the estimator on the data and evaluate it using cross-validation
#     scores = cross_val_score(nn, X_train, y_train, cv=5)
#     mean_score = np.mean(scores)
#     print(mean_score)
#
#     # Update the best parameters and the best score if the current mean score is better
#     if mean_score > best_score:
#         best_score = mean_score
#
# print(best_score)
