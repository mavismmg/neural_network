import random
from all_math import _math as solve

""" Class NeuralNetwork """

"""     Transformar tudo p/ padrao class 
        Transformar em package [Feito]
        Documentar 
"""

class NeuralNetwork:
    def __init__(self):
        """ self.weights = [[random(), random()],
        [random(), random()]]

        self.bias = random() """
        #self.learning_rate = learning_rate

    def make_prediction(input, weights, bias, similarity):
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

    def compute_gradient(self, input, target):
        layer_1 = solve.Math.dot_product(input, self.weights) + self.bias
        value = layer_1
        layer_2 = solve.Math.sigmoid(value)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = solve.Math.derivative_sigmoid(layer_1)
        dlayer1_dbias1 = 1
        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias1
        dlayer1_dweights = (0 * self.weights) + (1 * input)
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

        return derror_dbias, derror_dweights

    def update_par(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    def train(self, input, target, iter):
        cumm_errors = []
        for current_iter in range(iter):
            random_data_index = random.randint(len(input))
            input = input[random_data_index]
            derror_dbias, derror_dweights = self.compute_gradient(input, target)
            self.update_par(derror_dbias, derror_dweights)
            if current_iter % 100 == 0:
                cumm_error = 0
                for data_instance_index in range(len(input)):
                    data_point = input[data_instance_index]
                    target_ = target[data_instance_index]
                    prediction = self.make_prediction(data_point)
                    error = solve.Math.mean_square_error(prediction, target, mse_n=(len(input)))
                    cumm_error = cumm_error + error
                
                cumm_errors.append(cumm_error)

        return cumm_errors