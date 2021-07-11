import random
from all_math import _math as solve

""" Class NeuralNetwork """
class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = [[random(), random()],
        [random(), random()]]

        self.bias = random()
        self.learning_rate = learning_rate

    def make_prediction(self, input, weights, bias, similarity):
        layer_1 = solve.Math.dot_product(input, weights) + bias
        for i in range(len(similarity) - 1):
            if (similarity[i] > similarity[i+1]):
                value = layer_1[i]
            else:
                value = layer_1[i+1]

        layer_2 = solve.Math.sigmoid(value)

        return layer_2

    def make_prediction_test(input, weights, bias, similarity):
        layer_1 = solve.Math.dot_product_test(input, weights) + bias
        value = layer_1[0]
        layer_2 = solve.Math.sigmoid(value)

        return layer_2

    def define_weights_for_derivative(weights, input_vector):
        weights_1 = [[0] * (len(weights))] * (len(input_vector))
        var_aux = 0
        var_aux = weights
        weights_1 = var_aux

        return weights_1
