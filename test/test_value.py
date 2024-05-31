import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lazytorch import Value

def example_1():
    print("EXAMPLE 1")

    a = Value(2.0)
    b = Value(3.0)

    f = b + b * a  # simple function
    print(f)  # value = 9

    f.backprop()  # computing the gradients of f, a, and b, all with respect to f

    print(f'Gradient of a: {a.gradient}')  # value = 3
    print(f'Gradient of b: {b.gradient}')  # value = 3
    print(f'Gradient of f: {f.gradient}')  # value = 1
    print()

def example_2():
    print("EXAMPLE 2")

    a = Value(2.0)
    b = Value(3.0)

    f = (b - a) * b / a
    print(f)  # value = 1.5

    f.backprop()

    print(f'Gradient of a: {a.gradient}')  # expected value = -2.25
    print(f'Gradient of b: {b.gradient}')  # expected value = 2
    print(f'Gradient of f: {f.gradient}')  # expected value = 1
    print()

def example_3():
    print("EXAMPLE 3")

    a = Value(2.0)
    b = Value(3.0)

    f = b + b * a
    z = f * b
    print(f)
    print(z)

    z.backprop()

    print(f'Gradient of a: {a.gradient}')
    print(f'Gradient of b: {b.gradient}')
    print(f'Gradient of f: {f.gradient}')
    print(f'Gradient of z: {z.gradient}')
    print()

def example_4():
    print("EXAMPLE 4")

    a = Value(2.0)
    b = Value(3.0)

    f = a + b
    print(f)  # value = 5

    f.backprop()

    print(f'Gradient of a: {a.gradient}')  # expected value = 1
    print(f'Gradient of b: {b.gradient}')  # expected value = 1
    print(f'Gradient of f: {f.gradient}')  # expected value = 1
    print()

def example_5():
    print("EXAMPLE 5")

    a = Value(2.0)
    b = 10.0

    f = a * b
    print(f)  # value = 20.0

    f.backprop()

    print(f'Gradient of a: {a.gradient}')  # expected value = 10.0
    print(f'Gradient of f: {f.gradient}')  # expected value = 1
    print()

if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    example_4()
    example_5()
