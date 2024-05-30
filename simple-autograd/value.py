class Value:
    def __init__(self, value: float):
        self.value = value
        self.gradient = 0
        self._backward = lambda: None
        self._prev = set()

    def zero(self):
        self.gradient = 0

    def __str__(self):
        return str(self.value)

    def __add__(self, other):
        out = Value(self.value + other.value)

        def add_backward():
            self.gradient += out.gradient
            other.gradient += out.gradient
        
        out._backward = add_backward
        out._prev = {self, other}
        return out
    
    def __sub__(self, other):
        out = Value(self.value - other.value)

        def sub_backward():
            self.gradient += out.gradient
            other.gradient -= out.gradient
        
        out._backward = sub_backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        out = Value(self.value * other.value)

        def mul_backward():
            self.gradient += other.value * out.gradient
            other.gradient += self.value * out.gradient

        out._backward = mul_backward
        out._prev = {self, other}
        return out
    
    def __truediv__(self, other):
        out = Value(self.value / other.value)

        def div_backward():
            self.gradient += (1 / other.value) * out.gradient
            other.gradient -= (self.value / (other.value ** 2)) * out.gradient

        out._backward = div_backward
        out._prev = {self, other}
        return out

    def backward(self) -> float:
        ordered_vals = []
        visited_vals = set()

        def build_order(val: Value):
            if val not in visited_vals:
                visited_vals.add(val)
                for v in val._prev:
                    build_order(v)
                ordered_vals.append(val)

        build_order(self)

        self.gradient = 1

        for val in reversed(ordered_vals):
            val._backward()

if __name__ == "__main__":
    def example_1():
        print("EXAMPLE 1")

        a = Value(2.0)
        b = Value(3.0)

        f = b + b * a  # simple function
        print(f)  # value = 9

        f.backward()  # computing the gradients of f, a, and b, all with respect to f

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

        f.backward()

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

        z.backward()

        print(f'Gradient of a: {a.gradient}')
        print(f'Gradient of b: {b.gradient}')
        print(f'Gradient of f: {f.gradient}')
        print(f'Gradient of z: {z.gradient}')
        print()

    example_1()
    example_2()
    example_3()
