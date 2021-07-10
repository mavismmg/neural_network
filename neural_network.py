import math

""" Class NeuralNetwork """

def sigmoid(a):

    return 1 / (1 + math.exp(-a))


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