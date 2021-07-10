import matplotlib.pyplot as plt
import neural_network as pred
import matrix_alloc as alloc

""" derivative_vector = []
weights_1 = [[0] * (len(weights))] * (len(input_vector))
var_aux = 0
print(weights_1)
for i in range(len(weights)):
    for j in range(len(input_vector)):
        derivative_vector.append(derivative)
        var_aux = weights
        #var_aux = [[weights_1[i][j] - derivative_vector[j]] * (len(weights))] * len(input_vector)
        weights_1 = var_aux
        #weights_1 = weights_1[i][j] - derivative_vector[j]
        print(weights_1) """

""" for i in range(len(weights) - 1):  
    for j in range(len(input_vector)):
        derivative_vector.append(derivative)
        print(derivative_vector)
        #weights_1.append(weights[i][j] - derivative_vector[j])
        weights_1 = [[weights_1[i][j] - derivative_vector[j]] * len(weights)] * len(input_vector) """

def define_weights_for_derivative(weights, input_vector):
    weights_1 = [[0] * (len(weights))] * (len(input_vector))
    var_aux = 0
    var_aux = weights
    weights_1 = var_aux

    return weights_1


def derivative_vector(input, derivative):
    vector_derivative = []
    for i in range(len(input)):
        vector_derivative.append(derivative)

    return vector_derivative


def derivative_weights(weights_1, vector_derivative):
    new_weights = []
    for i in range(len(weights_1) - 1):
        for j in range(len(vector_derivative)):
            new_weights.append(weights_1[i][j] - vector_derivative[j])
    
    return new_weights


def main():
    input_vector = [2, 1.5]
    weights = [[1.45, -0.66], [0.0, 0.0]]
    """ input_vector = [5.23, 2.54, 8.99, 3.22, 9.99, 10.54]
    weights = [[1.12, 3.44, 3.14, 6.32, 8.45, 7.54], 
                [8.45, 3.57, 2.22, 7.45, 6.22, 7.77]] """
    bias = [0.0]

    similarity = pred.dot_product(input_vector, weights)
    prediction = pred.make_prediction(input_vector, weights, bias, similarity)

    target = 0
    mse_n = len(input_vector)
    mse = pred.mean_square_error(prediction, target, mse_n)
    #mse = np.square(prediction - target)

    derivative = 2 * (prediction - target)
    weights_1 = define_weights_for_derivative(weights, input_vector)
    vector_derivative = derivative_vector(input_vector, derivative)
    weights_d = derivative_weights(weights_1, vector_derivative)

    similarity_1 = pred.dot_product(input_vector, weights_d)
    prediction_1 = pred.make_prediction(input_vector, weights_d, bias, similarity_1)
    #mse_1 = np.square(prediction_1 - target)
    error = (prediction - target) ** 2

    print('Similaridades: ', similarity)
    print('Prediction: ', prediction)
    print('Mean squared error: ', mse)
    print('Derivative: ', derivative)

    print('Prediction_1:', prediction_1)
    #print('Mse 1:', mse_1)

    #plt.scatter(prediction, mse)
    plt.scatter(prediction, mse)
    plt.show()

if __name__ == "__main__":
    dinamic_array = alloc.DynamicMatrix()
    dinamic_array.append(2)
    print(dinamic_array, dinamic_array[0])
    main()