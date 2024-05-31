import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lazytorch import Tensor

def example_1():
    print("EXAMPLE 1\n")

    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([[5, 6], [7, 8]])
    results = tensor1 + tensor2

    print(tensor1)
    print()
    print(tensor2)
    print()
    print(results)
    print()

    results.backprop()

    print("GRADIENTS:\n")
    print(tensor1.gradient())
    print()
    print(tensor2.gradient())
    print()
    print(results.gradient())
    print()
    print()

def example_2():
    print("EXAMPLE 2\n")

    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([[5, 6], [7, 8]])
    results = tensor1 - tensor2

    print(tensor1)
    print()
    print(tensor2)
    print()
    print(results)
    print()

    results.backprop()

    print("GRADIENTS:\n")
    print(tensor1.gradient())
    print()
    print(tensor2.gradient())
    print()
    print(results.gradient())
    print()
    print()

def example_3():
    print("EXAMPLE 3\n")

    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([[5, 6], [7, 8]])
    results = tensor1 * tensor2

    print(tensor1)
    print()
    print(tensor2)
    print()
    print(results)
    print()

    results.backprop()

    print("GRADIENTS:\n")
    print(tensor1.gradient())
    print()
    print(tensor2.gradient())
    print()
    print(results.gradient())
    print()
    print()

if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
