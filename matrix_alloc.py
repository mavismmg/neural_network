import ctypes

class DynamicMatrix(object):
    def __init__(self):
        self.line = 0
        self.column = 0
        self.capacity = 1
        self.matrix = self.make_matrix(self.capacity)

    def __len__(self):
            
        return self.line, self.column

    def __getitem__(self, i):
        if not 0 <= i < self.line:

            return IndexError('i is out of bounds')

        # if not 0 <= j < self.column:
            
        #     return IndexError('j is out of bounds')

        return self.matrix[i]

    def append(self, element):
        # if self.line == self.capacity:
        #     self._resize(2 * self.capacity)

        # if self.column == self.capacity:
        #     self._resize(2 * self.capacity)

        if self.line == self.capacity & self.column == self.capacity:
            for i in range(self.line):
                self._resize(2 * self.capacity)
                for j in range(self.column):
                    self._resize(2 * self.capacity)

        for i in range(self.line):
            self.matrix[self.line] = element
            self.line += 1
            for j in range(self.column):
                self.matrix[self.column] = element
                self.column += 1

        # self.matrix[self.line] = element
        # self.matrix[self.column] = element
        # self.line += 1
        # self.column += 1

    """     def insertAt(self, item, index):
            if index < 0 or index > self.size:

                return print('please enter appropriate index')

            if self.size == self.capacity:
                self._resize(2*self.capacity)

            for i in range(self.size-1, index-1, -1):
                self.matrix[i+1] = self.matrix[i]

            self.matrix[index] = item
            self.size += 1 """

    """     def delete(self):
            if self.size == 0:

                return print('removing is not possible')

            self.matrix[self.matrix-1] = 0
            self.matrix -= 1 """

    """     def removeAt(self, index):
        if self.size == 0:
            
            return print('its already empty')

        if index < 0 or index >= self.size:

            return print('index is out of bounds')

        if index == self.size-1:
            self.matrix[index] = 0
            self.size -= 1

            return

        for i in range(index, self.size-1):
            self.matrix[i] = self.matrix[i+1]

        self.matrix[self.size-1] = 0
        self.size -= 1 """

    def _resize(self, new_capacity):
        matrix_2 = self.make_matrix(new_capacity)
        for i in range(self.line):
            matrix_2[i] = self.matrix[i]
            for j in range(self.column):
                matrix_2[j] = self.matrix[j]

        self.matrix = matrix_2
        self.capacity = new_capacity

    def make_matrix(self, new_capacity):

        return (new_capacity * ctypes.py_object)()