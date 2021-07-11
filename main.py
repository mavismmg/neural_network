import matplotlib.pyplot as plt
import neural_network as Neural
import matrix_alloc as alloc
from all_math import _math as solve

def main():
    input_vector = [2, 1.5]
    weights = [[1.45, -0.66], [0.0, 0.0]]
    """ input_vector = [5.23, 2.54, 8.99, 3.22, 9.99, 10.54]
    weights = [[1.12, 3.44, 3.14, 6.32, 8.45, 7.54], 
                [8.45, 3.57, 2.22, 7.45, 6.22, 7.77]] """
    bias = [0.0]

    similarity = solve.Math.dot_product(input_vector, weights)
    prediction = Neural.NeuralNetwork.make_prediction(input_vector, weights, bias, similarity)

    target = 0
    mse_n = len(input_vector)
    mse = solve.Math.mean_square_error(prediction, target, mse_n)

    derivative = 2 * (prediction - target)
    weights_1 = Neural.NeuralNetwork.define_weights_for_derivative(weights, input_vector)
    vector_derivative = solve.Math.derivative_vector(input_vector, derivative)
    weights_d = solve.Math.derivative_weights(weights_1, vector_derivative)

    similarity_1 = solve.Math.dot_product_test(input_vector, weights_d)
    prediction_1 = Neural.NeuralNetwork.make_prediction_test(input_vector, weights_d, bias, similarity_1)
    mse_1 = solve.Math.mean_square_error(prediction_1, target, mse_n)
    error = (prediction - target) ** 2

    print('Similaridades: ', similarity)
    print('Prediction: ', prediction)
    print('Mean squared error: ', mse)
    print('Derivative: ', derivative)

    print('Prediction_1:', prediction_1)
    print('Mse 1:', mse_1)

    derror_dprediction = 2 * (prediction - target)
    layer_1 = solve.Math.dot_product_test(input_vector, weights_d) + bias
    dprediction_dlayer1 = solve.Math.derivative_sigmoid(layer_1)
    dlayer1_dbias1 = 1
    derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias1


if __name__ == "__main__":
    main()