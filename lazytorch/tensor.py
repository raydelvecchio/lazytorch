from .value import Value

class Tensor:
    def __init__(self, data):
        self.data = [[Value(x) if not isinstance(x, Value) else x for x in row] for row in data]
        self.shape = (len(data), len(data[0]))
    
    def __str__(self):
        return '\n'.join(['\t'.join([str(x) for x in row]) for row in self.data])

    def __add__(self, other):
        """
        If you pass in a Tensor, it adds the tensors together. If you pass in a scalar, it performs element wise addition of the scalar.
        """
        if isinstance(other, Tensor):
            assert self.shape == other.shape, "Shapes must match for addition"
            result = [[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Tensor(result)
        elif isinstance(other, (int, float)):
            result = [[self.data[i][j] + Value(other) for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Tensor(result)
        else:
            raise Exception("Must add either a Tensor, float, or int")

    def __sub__(self, other):
        """
        If you pass in a Tensor, it subtracts each Tensor. If you pass in a scalar, it performs element wise subtraction of the scalar.
        """
        if isinstance(other, Tensor):
            assert self.shape == other.shape, "Shapes must match for subtraction"
            result = [[self.data[i][j] - other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Tensor(result)
        elif isinstance(other, (int, float)):
            result = [[self.data[i][j] - Value(other) for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Tensor(result)
        else:
            raise Exception("Must subtract by either a Tensor, float, or int")

    def __mul__(self, other):
        """
        If you pass in a Tensor, it MATRIX MULTIPLIES them together. If you pass in a scalar, it performs element wise multiplication of the scalar.
        """
        if isinstance(other, Tensor):
            assert self.shape[1] == other.shape[0], "Shapes are not aligned for matrix multiplication"
            result = []
            for i in range(self.shape[0]):
                result_row = []
                for j in range(other.shape[1]):
                    sum_value = Value(0)
                    for k in range(self.shape[1]):
                        sum_value += self.data[i][k] * other.data[k][j]
                    result_row.append(sum_value)
                result.append(result_row)
            return Tensor(result)
        elif isinstance(other, (int, float)):
            result = [[self.data[i][j] * Value(other) for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Tensor(result)
        else:
            raise Exception("Must multiply by a Tensor, float, or int")

    def __truediv__(self, other):
        """
        Only does element-wise division of the scalar.
        """
        if isinstance(other, (int, float)):
            result = [[self.data[i][j] / Value(other) for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Tensor(result)
        else:
            raise Exception("Must divide by either a float or int")

    def backprop(self):
        for row in self.data:
            for value in row:
                value.backprop()

    def gradient(self):
        """
        Returns the gradients of this Tensor as its own Tensor! This is the Jacobian Matrix of this Tensor.
        """
        result = [[Value(x.gradient) for x in row] for row in self.data]
        return Tensor(result)
    
    def zero(self):
        """
        Zeros out this Tensor by setting the gradients of everything to zero.
        """
        for row in self.data:
            for value in row:
                value.zero()
    
    def sum(self) -> Value:
        """
        Sums all values in this 2D tensor in a gradient-friendly way.
        """
        total = Value(0)
        for row in self.data:
            for value in row:
                total += value
        return total
