import math

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

def euler(n):
    term_sum = 0
    for i in range(n):
        term = 1 / math.factorial(i)
        term_sum = term_sum + term

    return term_sum

def sigmoid(a):
    """ Ainda não sei se será util, mas estou tentando achar uma forma melhor para arrendondar
        os valores utilizados """
    # Estamos arredondando a, então estamos acrescentando um erro ao cálculo.
    b = 0
    int_a = int(a)
    temp = 0
    #print('Valor de similaridade: ', a,'Valor transformando a var em int: ', int_a)
    e_error = a - int_a
    if (a - int_a >= 0.5):
        b = a + (1 - e_error)
        int_b = int(b)
        print(b)
        method_error = 1 - e_error
        print('Erro calculado: ', e_error, method_error)
        return 1 / (1 + euler(-int_b))
    else: 
        return 1 / (1 + euler(-int_a))
    print('Erro calculado: ', e_error, method_error)

    # return 1 / (1 + euler(-int_b))

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

def make_prediction(input, weights, bias):
    layer_1 = dot_product(input, weights) + bias
    for i in range(len(layer_1) - 1):
        # Pegando o maior valor weights calculado:
        if (layer_1[i] > layer_1[i+1]):
            value = layer_1[i]
        else:
            value = layer_1[i+1]
    layer_2 = sigmoid(value)

    return layer_2

""" input_vector = [0.2, 1.3]
weights = [[0.2, 1.5], [1.2, 3.5]] """
input_vector = [5.23, 2.54, 8.99, 3.22, 9.99, 10.54]
weights = [[1.12, 3.44, 3.14, 6.32, 8.45, 7.54], 
            [8.45, 3.57, 2.22, 7.45, 6.22, 7.77]]
bias = [0.0]

similarity = dot_product(input_vector, weights)
prediction = make_prediction(input_vector, weights, bias)

target = 0.5
mse_n = len(input_vector)
mse = mean_square_error(prediction, target, mse_n)

print('Similaridades: ', similarity)
print('Prediction: ', prediction)
print('Mean squared error: ', mse)