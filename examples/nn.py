import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lazytorch import Tensor, Network, MSE_Loss, DenseLayer, Leaky_ReLU
import random
import time

class FunctionApproximatorNN(Network):
    def __init__(self):
        """
        Defines all the hyperparameters and layers here to approximate x^2. Learns how to predict x^2 over the range [lower, upper], and nothing else.
        Simply define this with your own differentiable loss function, then call .train() and you're good to go!
        """
        super().__init__(loss_fn=MSE_Loss, layers=[
            DenseLayer(input_size=1, output_size=32),
            DenseLayer(input_size=32, output_size=32),
            DenseLayer(input_size=32, output_size=32),
            DenseLayer(input_size=32, output_size=1)
        ])

        self.training_losses = []
        self.lower_bound = -10
        self.upper_bound = 10
        self.learning_rate = 0.0001
        self.epochs = 20
        self.num_points = 5000  # number of data points in the dataset we want to train on
    
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
                self.apply_gradients(self.learning_rate)
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
            self.save_checkpoint(epoch = epoch + 1)
        
        total_elapsed_time = time.time() - total_start_time
        print(f"Total time for all epochs: {total_elapsed_time:.2f} seconds")

if __name__ == "__main__":    
    fann = FunctionApproximatorNN()
    fann.train()
