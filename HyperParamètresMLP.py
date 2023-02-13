import MLP
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from GridSearch import grid_search


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

nn = MLP.NeuralNetMLP()

# Define a parameter grid to search
# param_grid = {'epochs': [10, 20, 50],
#               'alpha': [0.0001, 0.001]}

# Dictionnaire contenant l'ensemble des valeurs des hyperparams
param_grid = {'n_output': [10],
              'n_features': [X_train.shape[1]],
              'n_hidden': [50, 60],
              'l2': [0.1],
              'l1': [0.0],
              'epochs': [10, 20],
              'eta': [0.001],
              'alpha': [0.001, 0.0001],
              'decrease_const': [0.00001],
              'minibatches': [50],
              'shuffle': [True],
              'random_state': [1]}

# # Grid qui associe un model avec une liste d'hyperparam à tester
# grid_search = GridSearchCV(nn, param_grid, cv=5, scoring='accuracy')
grid_search = grid_search(estimator=nn, param_grid=param_grid, X= X_train, y= y_train) # cv=5, scoring='accuracy')

# # # On lance l'entrainement
# grid_search.fit(X_train, y_train)
#
# # # On affiche la combinaison d'hyper param qui nous a donné les meilleurs score
# # print('Meilleur combinaison de param')
# print('Best parameters: ', grid_search.best_params_)
# print('Best accuracy: ', grid_search.best_score_)

# use the best parameters to create the final model
# mlp_best = nn(**grid_search.best_params_)
# mlp_best.fit(X_train, y_train)

# # HYPER PARAM
# EPOCH = [10]
# BATCH_SIZE = [500]
# OPTIMIZER = ['rmsprop', 'adam']# Dictionnaire contenant l'ensemble des valeurs des hyperparams
#
# hyperMatrix = dict(epochs=EPOCH, batch_size=BATCH_SIZE)
#
# # Grid qui associe un model avec une liste d'hyperparam à tester
# grid = GridSearchCV(estimator=nn, param_grid=hyperMatrix, cv=5, scoring='accuracy')
#
#
# # On lance l'entrainement
# history = grid.fit(X_train, y_train)
#
# # On affiche la combinaison d'hyper param qui nous a donné les meilleurs score
# print('Meilleur combinaison de param')
# print(history.best_params_)
# print("Meilleur score")
# print(history.best_score_ )
# print("Perf sur l'ensemble des combinaisons")
# print(history.cv_results_)


# nn.fit(X_train, y_train, print_progress=True)

# plt.plot(range(len(nn.cost_)), nn.cost_)
# plt.ylim([0, 2000])
# plt.ylabel('Cost')
# plt.xlabel('Epochs * 50')
# plt.tight_layout()
# # plt.savefig('./figures/cost.png', dpi=300)
# plt.show()
#
# batches = np.array_split(range(len(nn.cost_)), 1000)
# cost_ary = np.array(nn.cost_)
# cost_avgs = [np.mean(cost_ary[i]) for i in batches]
#
# plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
# plt.ylim([0, 2000])
# plt.ylabel('Cost')
# plt.xlabel('Epochs')
# plt.tight_layout()
# #plt.savefig('./figures/cost2.png', dpi=300)
# plt.show()

#
# y_train_pred = nn.predict(X_train)
# acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
# print('Training accuracy: %.2f%%' % (acc * 100))
#
# y_test_pred = nn.predict(X_test)
# acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
# print('Test accuracy: %.2f%%' % (acc * 100))
#
#
# miscl_img = X_test[y_test != y_test_pred][:25]
# correct_lab = y_test[y_test != y_test_pred][:25]
# miscl_lab = y_test_pred[y_test != y_test_pred][:25]
#
# fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i in range(25):
#     img = miscl_img[i].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#     ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# # plt.savefig('./figures/mnist_miscl.png', dpi=300)
# plt.show()
