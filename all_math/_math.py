import math

class Math:
    def __init__(self):
        """ dot_product var """
        self.dot_product_vector = []
        self.similarity = []
        """ derivative var """
        self.vector_derivative = []
        self.n_weights = []

    def sigmoid(self, a):

        return 1 / (1 + math.exp(-a))

    def mean_square_error(self, prediction, target, mse_n):
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
        Consequentemente as saídas também são mais similares, a partir disso temos uma predição; 
    """
    def dot_product(self, input, weights):
        dot_sum = 0
        for i in range(len(weights)):
            for j in range(len(input)):
                self.dot_product_vector = input[j] * weights[i][j]
                cur_index = self.dot_product_vector
                dot_sum = cur_index + dot_sum

            self.similarity.append(dot_sum)
            dot_sum = 0

        return self.similarity

    def dot_product_test(self, input, weights):
        dot_sum = 0
        for i in range(len(input)):
            self.dot_product_vector = input[i] * weights[i]
            cur_index = self.dot_product_vector
            dot_sum = cur_index + dot_sum
        
        self.similarity.append(dot_sum)
        dot_sum = 0

        return self.similarity

    def derivative_vector(self, input, derivative):
        for i in range(len(input)):
            self.vector_derivative.append(derivative)

        return self.vector_derivative

    def derivative_weights(self, weights_1):
        for i in range(len(weights_1) - 1):
            for j in range(len(self.vector_derivative)):
                self.n_weights.append(weights_1[i][j] - self.vector_derivative[j])
        
        return self.n_weights

    def derivative_sigmoid(self, a):
        converted_a = a[0]

        return Math.sigmoid(converted_a) * (1 - Math.sigmoid(converted_a))