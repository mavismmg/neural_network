import math

class Math:
    def __init__(self):
        """ dot_product var """
        self.dot_product_vector = []
        self.similarity = []
        """ derivative var """
        self.vector_derivative = []
        self.n_weights = []

    def sigmoid(a):

        return 1 / (1 + math.exp(-a))

    def mean_square_error(prediction, target, mse_n):
        sum = 0
        for current_iter in range(0, mse_n):
            diff = prediction - target
            sqre = diff ** 2
            sum = sqre + sum
        
        mse = sum / mse_n

        return mse

    """ Usando o produto escalar, podemos verificar a similaridade entre a entrada (input) ...
        e a entrada anterior (weights);
        Se o produto de weights(2) * input > weights(1) * input, então weights(2) é mais similar ao input;
        Consequentemente as saídas também são mais similares, a partir disso temos uma predição; 
    """
    def dot_product(input, weights):
        dot_sum = 0
        dot_product_vector = []
        similarity = []
        for line in range(len(weights)):
            for columm in range(len(input)):
                dot_product_vector = input[columm] * weights[line][columm]
                cur_index = dot_product_vector
                dot_sum = cur_index + dot_sum

            similarity.append(dot_sum)
            dot_sum = 0

        return similarity

    def dot_product_test(input, weights):
        dot_sum = 0
        dot_product_vector = []
        similarity = []
        for line in range(len(input)):
            dot_product_vector = input[line] * weights[line]
            cur_index = dot_product_vector
            dot_sum = cur_index + dot_sum
        
        similarity.append(dot_sum)
        dot_sum = 0

        return similarity

    def derivative_vector(input, derivative):
        vector_derivative = []
        for current_iter in range(len(input)):
            vector_derivative.append(derivative)

        return vector_derivative

    def derivative_weights(weights_1, vector_derivative):
        n_weights = []
        for line in range(len(weights_1) - 1):
            for columm in range(len(vector_derivative)):
                n_weights.append(weights_1[line][columm] - vector_derivative[columm])
        
        return n_weights

    def derivative_sigmoid(a):
        converted_a = a[0]

        return Math.sigmoid(converted_a) * (1 - Math.sigmoid(converted_a))