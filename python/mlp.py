"""
High level design of neural network code with a total of K layers.
There are K-2 hidden layers.

May need to transpose 2D array containing features so that its shape is
(n_features, n_samples) for an efficient implementation.

"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


SUPPORTED_ACTIVATIONS = ["identity", "sigmoid", "relu", "softmax"]
SUPPORTED_INITIALIZATION_METHODS = ["glorot_uniform", "he_uniform"]
SUPPORTED_LOSS_METHODS = ["cce", "mse"]
SUPPORTED_OPTIMIZATION_METHODS = ["sgd", "adam"]


class MLP:

    def __init__(self, layer_sizes: list[int], layer_activations: list[str], init_method: str = "he_uniform", loss_method: str = "mse", optimizer="adam") -> None:
        """
        Constructor of neural network class.
        Initialize all required arrays.
        """

        self.rng = np.random.default_rng(seed=42)

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.init_method = init_method
        self.layer_activations = layer_activations
        self.loss_method = loss_method
        self.optimizer = optimizer

        # Check that specified activation functions are legal
        for ind, fcn in enumerate(layer_activations):
            if fcn not in SUPPORTED_ACTIVATIONS:
                logger.error(
                    "The specified activation function for layer %d is %s. It is currently unspported.", ind, fcn)
                raise ValueError(" Unsupported activation function")

        # Check that initalization method is legal
        if init_method not in SUPPORTED_INITIALIZATION_METHODS:
            logger.error(
                "The initialization method specified, %s, is currently unsupported", init_method)
        self._init_arrays(layer_sizes)

        # Check that loss calculation method is legal
        if self.loss_method not in SUPPORTED_LOSS_METHODS:
            logger.error(
                "The specified method of loss calculation, %s, is currently unsupported.", self.loss_method)
            raise ValueError("Unsupported Loss method")

        # Check that optimization method is legal
        if self.optimizer not in SUPPORTED_OPTIMIZATION_METHODS:
            logger.error(
                "The specified optimization method, %s, is currently unsupported.", self.optimizer)
            raise ValueError("Unsupported optimization method")

        self._init_adam()

        # Initialize data arrays to None
        self.features, self.labels = None, None

    def _init_adam(self) -> None:
        """
        Initializes arrays for Adam solver
        """
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.adam_beta_1_t, self.adam_beta_2_t = 1., 1.
        self.adam_epsilon = 1.0e-8
        self.adam_first_moment_weights = []
        self.adam_first_moment_biases = []
        self.adam_second_moment_weights = []
        self.adam_second_moment_biases = []

        number_neurons_prev = self.layer_sizes[0]

        for ind, num in enumerate(self.layer_sizes):
            number_neurons = self.layer_sizes[ind]
            self.adam_first_moment_weights.append(
                np.zeros((number_neurons, number_neurons_prev)))
            self.adam_second_moment_weights.append(
                np.zeros((number_neurons, number_neurons_prev)))
            self.adam_first_moment_biases.append(np.zeros((number_neurons, 1)))
            self.adam_second_moment_biases.append(
                np.zeros((number_neurons, 1)))
            number_neurons_prev = number_neurons

    def _get_std(self, layer_index: int) -> float:
        """
        Get standard deviation for weights initialization
        """

        if self.init_method == "glorot_uniform":
            limit_val = np.sqrt(
                6./(self.layer_sizes[layer_index - 1] + self.layer_sizes[layer_index]))
        elif self.init_method == "he_uniform":
            limit_val = np.sqrt(6./self.layer_sizes[layer_index])

        return limit_val

    def _init_arrays(self, layer_sizes: list[int]):
        """
        Initialize all required arrays.

        Initialize weights with zero mean and ? variance.
        Initialize biases to zero vectors.
        For first layer weights matrix is the identity matrix

        """

        self.weights = []
        self.biases = []
        self.y = []
        self.z = []
        self.dcdw = []
        self.dcdb = []
        self.dcdz = []

        # Separate initialization for first layer
        num_first = layer_sizes[0]
        weights_first = np.eye(num_first)
        self.weights.append(weights_first)
        self.biases.append(np.zeros((num_first, 1)))
        self.y.append(np.zeros((num_first, 1)))
        self.z.append(np.zeros((num_first, 1)))
        self.dcdw.append(np.zeros((num_first, num_first)))
        self.dcdb.append(np.zeros((num_first, 1)))
        self.dcdz.append(np.zeros((num_first, 1)))

        for ind, num in enumerate(layer_sizes[1:]):
            layer_index = ind + 1
            layer_limit = self._get_std(layer_index)
            weights_layer = self.rng.uniform(
                low=-1.0*layer_limit, high=layer_limit, size=(num, layer_sizes[ind]))
            self.weights.append(weights_layer)
            biases_layer = np.zeros((num, 1))
            self.biases.append(biases_layer)
            y_layer = np.zeros((num, 1))
            self.y.append(y_layer)
            z_layer = np.zeros((num, 1))
            self.z.append(z_layer)
            dcdw_layer = np.zeros((num, layer_sizes[ind]))
            self.dcdw.append(dcdw_layer)
            dcdb_layer = np.zeros((num, 1))
            self.dcdb.append(dcdb_layer)
            dcdz_layer = np.zeros((num, 1))
            self.dcdz.append(dcdz_layer)

    def set_data(self, x: np.ndarray, y: np.ndarray):
        """
        Initialize features and labels.
        Note that the feature array,x, must be of shape (n_features, n_samples).
        The array y must be one-hot encoded and have the shape (n_categories, n_samples).
        """

        # Check that array shapes are correct
        if x.shape[1] != y.shape[1]:
            logger.error("The second dimensions of x and y do not match.")
            raise ValueError("The shapes of x and y do not match.")

        self.all_features, self.all_labels = x, y

    def _relu(self, x: float) -> float:
        """
        Evaluates rectified linear unit function.
        Note the use of 0. rather than 0
        """
        result = x if x >= 0. else 0.
        return result

    def _relu_derivative(self, x: float) -> float:
        """
        Returns derivative of rectified linear unit function with respect to its arguments.
        """
        result = 1. if x > 0. else 0.
        return result

    def _evaluate_activation(self, layer_index: int) -> np.ndarray:
        """
        Evaluate activation function
        """

        layer_activation = self.layer_activations[layer_index]
        activation_input = self.y[layer_index]

        if layer_activation == "sigmoid":
            result = 1.0/(1.0 + np.exp(-1.0*activation_input))
        elif layer_activation == "relu":
            relu_func = np.vectorize(self._relu)
            result = relu_func(activation_input)
        elif layer_activation == "softmax":
            exp_result = np.exp(activation_input)
            result = exp_result/np.sum(exp_result, axis=0)
        elif layer_activation == "identity":
            result = activation_input
        return result

    def forward(self):
        """
        Forward pass
        """

        inp = self.features

        for ind, num in enumerate(self.layer_sizes):
            weights = self.weights[ind]
            biases = self.biases[ind]
            output = np.matmul(weights, inp) + biases
            if np.max(np.abs(output)) > 1.0e5:
                raise ValueError("Large value at layer ", ind)
            self.y[ind] = output
            self.z[ind] = self._evaluate_activation(ind)
            inp = self.z[ind]

    def _calculate_categorical_cross_entropy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate categorical cross entropy using predicted
        probabilities of belonging to each class and
        labels

        predictions must have shape (n_categories, n_samples)
        labels must be one hot encoded and transposed such its shape is (n_categories, n_samples)
        """
        log_pred = np.log(predictions)
        current_loss = -1.0*np.sum(log_pred*labels)
        return current_loss

    def _calculate_loss(self) -> float:
        """
        Calculate loss using categorical cross entropy.
        https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9
        """
        predictions = self.z[self.num_layers - 1]
        if self.loss_method == "cce":
            current_loss = self._calculate_categorical_cross_entropy(
                predictions, self.labels)
        elif self.loss_method == "mse":
            current_loss = self._evaluate_mean_square_error(
                predictions, self.labels)
            current_loss = np.mean(current_loss)

        return current_loss

    def _evaluate_gradient_activation(self, layer_index: int) -> np.ndarray:
        """
        Evaluate derivative of activation function with respect to its argument

        Returns a n by n matrix if the activation input is an n-element vector

        The value of result[i,j] is the derivative of activation_result[i] with 
        respect to activation_result[j]. 

        """
        layer_activation = self.layer_activations[layer_index]
        activation_input = self.y[layer_index]
        activation_result = self.z[layer_index]

        if layer_activation == "sigmoid":
            derivative_values = activation_result*(1. - activation_result)
            result = np.diag(derivative_values.squeeze())
        elif layer_activation == "relu":
            relu_derivative_func = np.vectorize(self._relu_derivative)
            derivative_values = relu_derivative_func(activation_input)
            result = np.diag(derivative_values.squeeze())
        elif layer_activation == "softmax":
            result = -1.0*np.matmul(activation_result, activation_result.T)
            result = np.diag(activation_result.squeeze()) + result
        elif layer_activation == "identity":
            result = np.eye(activation_input.shape[0])
        return result

    def _evaluate_derivative_cross_entropy(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculates derivative of cross-entropy with respect to the final predictions
        of the neural network.

        predictions must have shape (n_categories,1).
        labels must have shape (n_categories,1)

        Returns a matrix of shape (n_categories,1) where n_categories is equal to the 
        number of neurons in the output layer.

        """
        inv_predictions = 1.0/predictions
        result = -1.0*inv_predictions*labels
        return result

    def _evaluate_mean_square_error(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculates mean square error
        """
        error = (predictions - labels)*(predictions - labels)
        return error

    def _evaluate_derivative_mean_square_error(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculates derivative of squared error with respect to output of neurons in last layer.
        """
        derivative_error = 2.*(predictions - labels)
        return derivative_error

    def _evaluate_derivative_loss(self):
        """
        Calculates the derivative of the loss function
        """
        predictions = self.z[self.num_layers - 1]
        if self.loss_method == "cce":
            return self._evaluate_derivative_cross_entropy(predictions, self.labels)
        elif self.loss_method == "mse":
            return self._evaluate_derivative_mean_square_error(predictions, self.labels)

    def backward(self):
        """
        Perform back-propagation and calculate gradients
        """

        # Start from the output layer
        dcdz_current = self._evaluate_derivative_loss().reshape(-1, 1)
        self.dcdz[-1] += dcdz_current
        derivative_activation_current = self._evaluate_gradient_activation(
            self.num_layers - 1)
        dcdb_current = np.matmul(dcdz_current.T, derivative_activation_current)
        dcdb_current = dcdb_current.T
        self.dcdb[-1] += dcdb_current
        dcdw_current = np.matmul(dcdb_current, self.z[self.num_layers - 2].T)
        self.dcdw[-1] += dcdw_current

        derivative_activation_prev = derivative_activation_current
        dcdz_prev = dcdz_current

        for ind in range(self.num_layers-2, 0, -1):

            dcdz_current = np.matmul(dcdz_prev.T, np.matmul(
                derivative_activation_prev, self.weights[ind + 1]))
            dcdz_current = dcdz_current.T
            if np.max(np.abs(dcdz_current)) > 1.0e5:
                max_derivative = np.max(np.abs(dcdz_current))
                raise ValueError(
                    f"Derivative value is {max_derivative} at layer {ind}")
            derivative_activation_current = self._evaluate_gradient_activation(
                ind)
            dcdb_current = np.matmul(
                dcdz_current.T, derivative_activation_current)
            dcdb_current = dcdb_current.T
            dcdw_current = np.matmul(dcdb_current, self.z[ind - 1].T)
            self.dcdz[ind] += dcdz_current
            self.dcdb[ind] += dcdb_current
            self.dcdw[ind] += dcdw_current
            derivative_activation_prev = derivative_activation_current
            dcdz_prev = dcdz_current

    def _average_gradients(self, num_samples: int):
        """
        Average the values of dcdw, and dcdb across all samples
        """

        for ind in range(0, self.num_layers):
            self.dcdz[ind] /= num_samples
            self.dcdw[ind] /= num_samples
            self.dcdb[ind] /= num_samples

    def _zero_gradients(self):
        """
        Reset all gradients to zero
        """
        for ind in range(0, self.num_layers):
            self.dcdz[ind] = np.zeros((self.layer_sizes[ind], 1))
            self.dcdw[ind] = np.zeros(
                (self.layer_sizes[ind], self.layer_sizes[ind-1]))
            self.dcdb[ind] = np.zeros((self.layer_sizes[ind], 1))

    def _update_weights(self, learning_rate: float = 1.0e-3) -> None:
        """
        Update the weights and biases of all neurons using gradient descent.
        """

        if self.optimizer == 'sgd':
            for ind in range(1, self.num_layers):
                self.weights[ind] -= learning_rate*self.dcdw[ind]
                self.biases[ind] -= learning_rate*self.dcdb[ind]
        elif self.optimizer == 'adam':
            for ind in range(1, self.num_layers):
                self.adam_first_moment_weights[ind] = self.adam_beta_1 * \
                    self.adam_first_moment_weights[ind] + \
                    (1. - self.adam_beta_1)*self.dcdw[ind]
                self.adam_second_moment_weights[ind] = self.adam_beta_2*self.adam_second_moment_weights[ind] + (
                    1. - self.adam_beta_2)*self.dcdw[ind]*self.dcdw[ind]
                self.adam_first_moment_biases[ind] = self.adam_beta_1 * \
                    self.adam_first_moment_biases[ind] + \
                    (1. - self.adam_beta_1)*self.dcdb[ind]
                self.adam_second_moment_biases[ind] = self.adam_beta_2*self.adam_second_moment_biases[ind] + (
                    1. - self.adam_beta_2)*self.dcdb[ind]*self.dcdb[ind]
                self.adam_beta_1_t = self.adam_beta_1_t*self.adam_beta_1
                self.adam_beta_2_t = self.adam_beta_2_t*self.adam_beta_2
                first_moment_hat_weights = self.adam_first_moment_weights[ind] / \
                    (1. - self.adam_beta_1_t)
                second_moment_hat_weights = self.adam_second_moment_weights[ind] / \
                    (1. - self.adam_beta_2_t)
                first_moment_hat_biases = self.adam_first_moment_biases[ind] / \
                    (1. - self.adam_beta_1_t)
                second_moment_hat_biases = self.adam_second_moment_biases[ind] / \
                    (1. - self.adam_beta_2_t)
                self.weights[ind] -= learning_rate*first_moment_hat_weights / \
                    (np.sqrt(second_moment_hat_weights) + self.adam_epsilon)
                self.biases[ind] -= learning_rate*first_moment_hat_biases / \
                    (np.sqrt(second_moment_hat_biases) + self.adam_epsilon)

    def train_model(self, num_epochs: int = 5, learning_rate: float = 1.0e-3, reset_weights: bool = True) -> None:
        """
        Perform training for the specified number of epochs
        """

        if self.all_features is None or self.all_labels is None:
            logger.error(
                "Please initialize data for features and labels by calling the set_data function.")
            raise Exception(" Data not initialized.")

        if reset_weights:
            self._init_arrays(self.layer_sizes)

        logger.info("Training neural network for %d epochs", num_epochs)

        logger.info("Epoch ", " Training Loss")

        num_samples = self.all_features.shape[1]

        self.losses = np.zeros(num_epochs)

        for epoch in range(num_epochs):
            self._zero_gradients()
            for sample in range(num_samples):
                self.features = self.all_features[:, sample].reshape(-1, 1)
                self.labels = self.all_labels[:, sample].reshape(-1, 1)
                self.forward()
                self.losses[epoch] += self._calculate_loss()
                self.backward()
            self.losses[epoch] /= num_samples
            self._average_gradients(num_samples=num_samples)
            self._update_weights(learning_rate=learning_rate)
            # print(f"Completed epoch {epoch} of {num_epochs}")

        logger.info("%d  %f", epoch, self.losses[-1])
        logger.info("Training completed successfully.")

    def get_predictions(self, test_features: np.ndarray) -> np.ndarray:
        """
        Returns predictions of model for the data in test_features.
        The shape of test_features must be (n_features, n_samples)
        Returns an array of shape (n_outputs, n_samples), where n_outputs
        is the number of quantities to be predicted.
        """

        self.features = test_features
        self.forward()
        predictions = self.z[-1]
        return predictions
