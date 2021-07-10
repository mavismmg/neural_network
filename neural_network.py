import math
import numpy as np
import matplotlib.pyplot as plt

def define_weights_for_derivative(weights):
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

""" Pegamos o valor de prediction e subtraímos de target, função resulta no erro
de prediction """
def mean_square_error(prediction, target, mse_n):
    sum = 0
    for i in range(0, mse_n):
        diff = prediction - target
        sqre = diff ** 2
        sum = sqre + sum
    
    mse = sum / mse_n

    return mse

""" Tentar implementar euler para float """
""" def euler(n):
    term_sum = 0
    for i in range(n):
        term = 1 / math.factorial(i)
        term_sum = term_sum + term

    return term_sum """

""" def round_number(a):
    int_a = int(a)
    error = a - int_a
    if (a - int_a >= 0.5):
        b = a + (1 - error)
        int_b = int(b)
        print(b)
        ver_error = 1 - error
        print(ver_error)

        return int_b
    else:
        return int_a """

def sigmoid(a):

    return 1 / (1 + math.exp(-a))

""" Usando o produto escalar, podemos verificar a similaridade entre a entrada (input) ...
e a entrada anterior (weights);
Se o produto de weights(2) * input > weights(1) * input, então weights(2) é mais similar ao input;
Consequentemente as saídas também são mais similares, a partir disso temos uma predição; """
def dot_product(input, weights):
    dot_sum = 0
    dot_product = []
    similarity = []
    for i in range(len(weights)):
        for j in range(len(input)):
            dot_product = input[j] * weights[i][j]
            value_cur_index = dot_product
            dot_sum = value_cur_index + dot_sum

        similarity.append(dot_sum)
        dot_sum = 0
    return similarity

def make_prediction(input, weights, bias, similarity):
    layer_1 = dot_product(input, weights) + bias
    for i in range(len(similarity) - 1):
        # Pegando o maior valor weights calculado:
        if (similarity[i] > similarity[i+1]):
            value = layer_1[i]
        else:
            value = layer_1[i+1]

    layer_2 = sigmoid(value)

    return layer_2

input_vector = [2, 1.5]
weights = [[1.45, -0.66], [0.0, 0.0]]
""" input_vector = [5.23, 2.54, 8.99, 3.22, 9.99, 10.54]
weights = [[1.12, 3.44, 3.14, 6.32, 8.45, 7.54], 
            [8.45, 3.57, 2.22, 7.45, 6.22, 7.77]] """
bias = [0.0]

similarity = dot_product(input_vector, weights)
prediction = make_prediction(input_vector, weights, bias, similarity)

target = 0
mse_n = len(input_vector)
mse = mean_square_error(prediction, target, mse_n)
#mse = np.square(prediction - target)

derivative = 2 * (prediction - target)
weights_1 = define_weights_for_derivative(weights)
vector_derivative = derivative_vector(input_vector, derivative)
weights_d = derivative_weights(weights_1, vector_derivative)

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

similarity_1 = dot_product(input_vector, weights_d)
prediction_1 = make_prediction(input_vector, weights_d, bias, similarity_1)
mse_1 = np.square(prediction_1 - target)
error = (prediction - target) ** 2

print('Similaridades: ', similarity)
print('Prediction: ', prediction)
print('Mean squared error: ', mse)
print('Derivative: ', derivative)

print('Prediction_1:', prediction_1)
print('Mse 1:', mse_1)

#plt.scatter(prediction, mse)
plt.scatter(prediction, mse)
plt.show()