from abc import ABC, abstractmethod
from .tensor import Tensor
from .value import Value
import random

class Layer(ABC):
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def zero(self):
        """
        Must be implemented to zero out every trainable parameter.
        """
        pass

    @abstractmethod
    def apply_gradients(self, lr: float):
        """
        Must be implemented to apply gradients (during a weight update) to each trainable parameter, given a learning rate.
        """
        pass

class DenseLayer(Layer):
    def __init__(self, input_size, output_size) -> None:
        self.weights = Tensor([[Value(random.uniform(-0.1, 0.1)) for _ in range(output_size)] for _ in range(input_size)])  # by default, initialize weights to random values       
        self.biases = Tensor([[Value(0.0) for _ in range(output_size)]])  # biases initialized to zero

    def __call__(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.weights.shape[0], "Input shape must match the shape of the weights"        
        output = x * self.weights        
        output = output + self.biases
        return output
    
    def zero(self):
        """
        Zeros out the weights and biases!
        """
        self.weights.zero()
        self.biases.zero()

    def apply_gradients(self, lr: float):
        """
        Subtracts the gradients of both the weights and biases from themselves, scaled by the learning rate.
        """
        weights_gradients = self.weights.gradient()
        biases_gradients = self.biases.gradient()

        # below modifies the weights directly and in-place, without messing up the gradient computation graph (although since we should zero out grads right after, it shouldn't matter)
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights.data[i][j].value -= lr * weights_gradients.data[i][j].value

        for i in range(self.biases.shape[0]):
            for j in range(self.biases.shape[1]):
                self.biases.data[i][j].value -= lr * biases_gradients.data[i][j].value
