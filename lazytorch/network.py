from abc import ABC, abstractmethod
from typing import List, Callable
from .tensor import Tensor
from .value import Value
from .layers import Layer
import os
import pickle

class Network(ABC):
    def __init__(self, loss_fn: Callable[[Tensor, Tensor], Value], layers: List[Layer]):
        """
        Initializes the network with a loss function and a list of layers.
        """
        self.loss_function = loss_fn
        self.layers = layers
        self.training_losses = []

    def zero(self):
        """
        Zeros out the gradients for all layers.
        """
        for layer in self.layers:
            layer.zero()

    def apply_gradients(self, learning_rate: float):
        """
        Applies gradients to all layers using the specified learning rate.
        """
        for layer in self.layers:
            layer.apply_gradients(lr=learning_rate)

    def save_checkpoint(self, epoch: int, checkpoint_dir: str = "checkpoints"):
        """
        Saves the current state of the network to a .pkl file.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_checkpoint.pkl")

        with open(checkpoint_path, "wb") as f:
            pickle.dump(self, f)

    @abstractmethod
    def forward(self, inp: Tensor) -> Tensor:
        """
        Defines the forward pass of the network.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Trains the network.
        """
        pass
