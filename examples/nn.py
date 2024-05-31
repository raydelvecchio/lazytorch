import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lazytorch import Value, Tensor
import random
from typing import Callable
import time
import os
import pickle

def MSE_Loss(predicted: Tensor, target: Tensor) -> Value:
    """
    Calculates a batch of predicted vs target predictions via MSE loss in the form of a tensor.

    NOTE: the target Tensor will have gradients associated with it, but these are irrelevant. 
    Since it's the predicted value, there should be no gradient associated here at all. This is a quirk of not implementing
    gradient freezing, and having gradients always calculated.
    """
    assert predicted.shape == target.shape, "Shapes of predicted and target must match"
    
    squared_diff = (predicted - target) * (predicted - target)  # calculate the squared difference in an autodiff-friendly manner
    mse_loss = squared_diff.sum() / (predicted.shape[0] * predicted.shape[1])
    
    return mse_loss

def Leaky_ReLU(x: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Gradient-Safe leaky relu implementation. Used as activation function here!
    """
    result = []
    for row in x.data:
        result_row = []
        for value in row:
            if value.value > 0:
                result_row.append(value)
            else:
                result_row.append(value * alpha)
        result.append(result_row)
    return Tensor(result)  # gradients can flow through this since they're stored in the Value objects comprising the tensor, not the Tensor itself!

class DenseLayer:
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
    
class FunctionApproximatorNN:
    def __init__(self, loss_fn: Callable[[Tensor, Tensor], Value]):
        """
        Defines all the hyperparameters and layers here to approximate x^2. Learns how to predict x^2 over the range [lower, upper], and nothing else.
        Simply define this with your own differentiable loss function, then call .train() and you're good to go!
        """
        self.loss_function = loss_fn  # the loss function we want to use!
        self.l1 = DenseLayer(1, 32)
        self.l2 = DenseLayer(32, 32)
        self.l3 = DenseLayer(32, 32)
        self.l4 = DenseLayer(32, 1)
        self.layers = [self.l1, self.l2, self.l3, self.l4]
        self.training_losses = []
        self.lower_bound = -10
        self.upper_bound = 10
        self.learning_rate = 0.0001
        self.epochs = 20
        self.num_points = 5000  # number of data points in the dataset we want to train on
        
    def zero(self):
        for layer in self.layers:
            layer.zero()
    
    def apply_gradients(self):
        for layer in self.layers:
            layer.apply_gradients(lr=self.learning_rate)
    
    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass of the neural net. Calls each layer with Leaky ReLU activation function.
        """
        output = inp
        for layer in self.layers[:-1]:
            output = Leaky_ReLU(layer(output))
        output = self.layers[-1](output)
        return output
    
    def train(self):
        """
        Trains the NN to approximate x^2. Constructs a dataset to train, then trains! Every epoch we save 
        a snapshot of the NN as a .pkl file for evaluation later.
        """
        x = [i * (20 / (self.num_points - 1)) - 10 for i in range(self.num_points)]
        y = [i ** 2 for i in x]

        start_time = time.time()
        total_start_time = start_time

        for epoch in range(self.epochs):
            combined = list(zip(x, y))
            random.shuffle(combined)  # shuffle these each epoch for smoother, more comprehensive training
            x, y = zip(*combined)
            for i, (x_val, y_val) in enumerate(zip(x, y)):
                out = self.forward(Tensor([[x_val]]))
                loss = self.loss_function(out, Tensor([[y_val]]))
                self.training_losses.append(loss.value)

                loss.backprop()
                self.apply_gradients()
                self.zero()

                if (i + 1) % 50 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Time for iteration {i - 49} to {i}: {elapsed_time:.2f} seconds")
                    start_time = time.time()
                    
                    if len(self.training_losses) >= 50:
                        avg_loss = sum(self.training_losses[-50:]) / 50
                        print(f"Average loss over the last 50 iterations: {avg_loss:.4f}")
                    else:
                        print(f"Last loss value: {self.training_losses[-1]:.4f}")

                    start_time = time.time()
            
            print(f"EPOCH {epoch + 1} / {self.epochs} COMPLETE")

            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_checkpoint.pkl")

            with open(checkpoint_path, "wb") as f:
                pickle.dump(self, f)
        
        total_elapsed_time = time.time() - total_start_time
        print(f"Total time for all epochs: {total_elapsed_time:.2f} seconds")

if __name__ == "__main__":    
    fann = FunctionApproximatorNN(loss_fn=MSE_Loss)
    fann.train()
