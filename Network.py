import numpy as np
import matplotlib.pyplot
import time

np.set_printoptions(suppress=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class neural_network:
    """
        A class to model a Neural Network (NN).
        Attributes
        ----------
        a=0.2 : double
            Learning rate

        layers : list(int)
            List of the number of nodes in each layer of the NN

        num_layers : int
            Number of layers

        func=sigmoid : double -> double
            Activation function used in the NN

        weights : list(list(double))
            List used to store the weights of links between each layer in the NN

        input : list(list(double))
            List used to store the outputs of each layer in the NN (after the application of the activation function)

        error : list(list(double))
            List used to store the errors of each layer in the NN

        Methods -------
        train(targets_list: list(double), inputs_list: list(double))
            Trains the model by inputting the elements of inputs_list, comparing the outputs with the elements of
            targets_list, and updating weights accordingly

        get_results(inputs_list(double))
            Returns the output of the NN given existing weights and inputs from inputs_list

        fwdfeed(weights: list(double), inputs: list(double))
            Returns the result of the activation function applied to the dot product of the weights and inputs given for
            use in forward propagation of the output

        backfeed(weights: list(double), errors: list(double))
            Returns the result of the dot product of the weights and errors given for use in back propagation of the error

        forwardpropagation()
            Propagates the output forward through every layer

        backpropagation()
            Propagates the error backward through every layer

        update_weight()
            Updates weights based on the backpropagation of the error

        fprop()
            Returns the results from the output node after propagating the input instance forward
        """

    def __init__(self, layers, a=.2, func=sigmoid):

        self.a = a
        self.layers = layers
        self.num_layers = len(layers)
        self.func = func

        # Initialize the weights between every layer.
        # Center the initial weight normal distribution at 0, with a standard deviation of 1 / sqrt(incoming links).

        self.weights = [np.random.normal(0.0, 1 / np.sqrt(x), (y, x)) for x, y in zip(layers[:-1], layers[1:])]

        # Initialize the input and error matrices for each layer.
        self.input = [np.zeros((x, 1)) for x in layers]
        self.error = [np.zeros((x, 1)) for x in layers]

    def train(self, targets_list, input_list):

        self.input[0] = input_list
        self.forwardpropagation()
        self.error[0] = targets_list - self.input[-1]
        self.backpropagation()
        self.update_weight()
        self.input = [np.zeros((x, 1)) for x in self.layers]
        self.error = [np.zeros((x, 1)) for x in self.layers]

    def get_results(self, inputs_list):
        self.input = inputs_list
        return self.fprop()

    def fwdfeed(self, weights, inputs):
        return self.func(np.dot(weights, inputs))

    def backfeed(self, weights, errors):
        return np.dot(weights, errors)

    def forwardpropagation(self):
        for i in range(self.num_layers - 1):
            self.input[i + 1] = self.fwdfeed(self.weights[i], self.input[i])

    def backpropagation(self):
        for i in range(self.num_layers - 1):
            self.error[i + 1] = self.backfeed(np.transpose(self.weights[self.num_layers - i - 2]), self.error[i])

    def update_weight(self):
        for i in range(self.num_layers - 1):
            self.weights[self.num_layers - 2 - i] += self.a * np.dot(
                (self.error[i] * self.input[self.num_layers - 1 - i] * (1.0 - self.input[self.num_layers - 1 - i]))[
                    None].T, self.input[self.num_layers - 2 - i][None])

        return self.weights

    def fprop(self):
        for i in range(self.num_layers - 1):
            self.input = np.reshape(self.input, self.layers[i])
            self.input = self.fwdfeed(self.weights[i], self.input)

        return np.asarray(self.input)


if __name__ == '__main__':

    my_array = [784, 300, 10]

    n = neural_network(my_array)

    # ------------ DATA ------------
    # Index 0 of each line is the true value of the handwritten digit

    # Training data,
    training_data_file = open("C:/Users/sam/Downloads/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # Test data
    test_data_file = open("C:/Users/sam/Downloads/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(my_array[-1]) + .01
        targets[int(all_values[0])] = 0.99
        n.train(targets, inputs)


    # Testing 10 handwritten digits: 0-9
    # listy contains the indices of each digit in the testing data

    # listy = [3, 2, 1, 30, 4, 15, 11, 0, 61, 7]
    #
    # for i in listy:
    #
    #     all_values = test_data_list[i].split(',')
    #     image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    #     matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    #     matplotlib.pyplot.show()
    #     scaled_input = np.asfarray(all_values[1:]) / 255.0 * 0.99 + .01
    #
    #     print(n.get_results(scaled_input))


    # ------------ Testing the NN ------------
    scorecard = 0

    for record in test_data_list:

        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.get_results(inputs)
        label = np.argmax(outputs)
        if label == correct_label:
            scorecard += 1

    print("performance = ", scorecard / len(test_data_list))

