import numpy as np
from scipy.special import expit
import sys


# Implémentation du MLP
# Couche d'entrées, 1 couche cachée et couche de sorties
#
# Code similaire à Adaline

class MLP2HiddenLayer(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_output : int
        Number of output units, should be equal to the number of unique class labels.
    n_features : int
        Number of features (dimensions) in the target dataset.Should be equal to the number of columns in the X array.
    n_hidden : int (default: 30)
        Number of hidden units.
    l1 : float (default: 0.0)
        Lambda value for L1-regularization. No regularization if l1=0.0 (default)
    l2 : float (default: 0.0)
        Lambda value for L2-regularization. No regularization if l2=0.0 (default)
    epochs : int (default: 500)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    alpha : float (default: 0.0)
        Momentum constant. Factor multiplied with the gradient of the previous epoch t-1 to improve learning speed
        w(t) := w(t) - (grad(t) + alpha*grad(t-1))
    decrease_const : float (default: 0.0)
        Decrease constant. Shrinks the learning rate after each epoch via eta / (1 + epoch*decrease_const)
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatches : int (default: 1)
        Divides training data into k minibatches for efficiency. Normal gradient descent learning if k=1 (default).
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.
    n_hidden_layers : int (default: 2)
        Number of hidden layer

    Attributes
    -----------
    cost_ : list
      Sum of squared errors after each epoch.

    """

    def __init__(self, n_output, n_features, n_hidden_layers=2, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0,
                 decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.weights, self.biases = self._initialize_weights_2()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches




    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]   Target values.

        Returns
        -----------
        onehot : array, shape = (n_labels, n_samples)

        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
        w1 = np.random.uniform(-1.0, 1.0,
                               size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0,
                               size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _initialize_n_weights_2(self):
        # Pour la couche d'entrée -- initilisation du poids
        w1 = np.random.randn(self.n_features, self.n_hidden_layers[0])
        self.weights.append(w1)
        bias1 = np.zeros((1, self.n_hidden_layers[0]))
        self.biases.append(bias1)

        # Pour les couches cachées -- initilisation des poids
        for i in range(1, len(self.n_hidden_layers)):
            hidden_w = np.random.randn(self.n_hidden_layers[i - 1], self.n_hidden_layers[i])
            self.weights.append(hidden_w)
            hidden_bias = np.zeros((1, self.n_hidden_layers[i]))
            self.biases.append(hidden_bias)

        # Pour la couche de sortie -- initilisation du poids
        w2 = np.random.randn(self.n_hidden_layers[-1], self.n_output)
        self.weights.append(w2)
        bias2 = np.zeros((1, self.n_output))
        self.biases.append(bias2)

        return self.weights, self.biases

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)

        Uses scipy.special.expit to avoid overflow
        error for very small input values z.

        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Compute gradient of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1.0 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        """Compute feedforward step

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.

        Returns
        ----------
        a_in : array, shape = [n_samples, n_features+1]
            Input values with bias unit.
        z1 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer 1.
        a1 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer 1.
        z2 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer 2.
        a2 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer 2.

        z_out : array, shape = [n_output_units, n_samples]
            Net input of output layer.
        a_out : array, shape = [n_output_units, n_samples]
            Activation of output layer.

        """
        # Input Layer
        a_in = self._add_bias_unit(X, how='column')
        # Hidden Layer 1
        z1 = w1.dot(a_in.T)
        a1 = self._sigmoid(z1)
        a1 = self._add_bias_unit(a1, how='row')
        # Hidden Layer 2
        z2 = w1.dot(a1)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        # Output Layer
        z_out = w2.dot(a1)
        a_out = self._sigmoid(z_out)
        return a_in, z1, a1, z2, a2, z_out, a_out

    def _feedforward_2(self, X):
        # Calculate activations for input layer
        z = np.dot(X, self.weights[0]) + self.biases[0]
        a = self._sigmoid(z)

        a_n = [a]

        # Calculate activations for hidden layers
        for i in range(1, len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._sigmoid(z)
            a_n.append(a)

        # Calculate activations for output layer
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        a = self._sigmoid(z)
        a_n.append(a)

        return a_n


    def _L2_reg(self, lambda_, w1, w2):
        """Compute L2-regularization cost"""
        return (lambda_ / 2.0) * (np.sum(w1[:, 1:] ** 2) +
                                  np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        """Compute L1-regularization cost"""
        return (lambda_ / 2.0) * (np.abs(w1[:, 1:]).sum() +
                                  np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        output : array, shape = [n_output_units, n_samples]
            Activation of the output layer (feedforward)
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        cost : float
            Regularized cost.

        """
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    #
    # Nous verrons plus tard
    #
    def _get_gradient(self, a_in, a1, a2, a_out, z1, z2, y_enc, w1, w2):
        """ Compute gradient step using backpropagation.

        Parameters
        ------------
        a_in : array, shape = [n_samples, n_features+1]
            Input values with bias unit.
        a1 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer 1.
        a2 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer 2.
        a_out : array, shape = [n_output_units, n_samples]
            Activation of output layer.
        z1 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer 1.
        z2 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer 2.
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        grad1 : array, shape = [n_hidden_units, n_features]
            Gradient of the weight matrix w1.
        grad2 : array, shape = [n_output_units, n_hidden_units]
            Gradient of the weight matrix w2.

        """
        # backpropagation
        sigma_out = a_out - y_enc  # erreur de classification
        z1 = self._add_bias_unit(z1, how='row')
        z2 = self._add_bias_unit(z2, how='row')
        sigma1 = w2.T.dot(sigma_out) * self._sigmoid_gradient(z1)
        sigma1 = sigma1[1:, :]
        grad1 = sigma1.dot(a_in)
        grad2 = sigma_out.dot(a2.T)

        # regularize
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

        return grad1, grad2

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.

        """
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        a_in, z1, a1, z2, a2, z_out, a_out = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z_out, axis=0)
        return y_pred

    #
    # Fonction d'entraînement
    #
    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data.

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        y : array, shape = [n_samples]
            Target class labels.
        print_progress : bool (default: False)
            Prints progress as the number of epochs
            to stderr.

        Returns:
        ----------
        self

        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)  # Vecteur one-hot

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):  # Nombre de passage sur le dataset

            # adaptive learning rate
            self.eta /= (
                        1 + self.decrease_const * i)  # Permet de réduire le nombre d'epochs nécessaire à la convergence en limitant les risques de "pas" trop grand!

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:  # on mélange le dataset à chaque epoch
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            # Si le mode minibatch est activé, le dataset en entrée est divisé en batch pour le calcul des gradients
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # feedforward
                a_in, z1, a1, z2, a2, z_out, a_out = self._feedforward(X_data[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx], output=a_out, w1=self.w1, w2=self.w2)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a_in=a_in, a1=a1, a2=a2, a_out=a_out, z1=z1, z2=z2, y_enc=y_enc[:, idx], w1=self.w1,
                                                  w2=self.w2)

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self

# Retour sur le powerpoint
