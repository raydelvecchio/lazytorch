class Value:
    def __init__(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError("Value must be a float or an int")
        
        self.value = value
        self.gradient = 0
        self._backprop = self.placeholder_backprop
        self._dependents = set()

    def placeholder_backprop(self):
        """
        Placeholder value for backprop to allow for pickling and checkpointing (can't pickle Lambdas)!
        """
        return None

    def zero(self):
        self.gradient = 0
        self._backprop = self.placeholder_backprop

    def __str__(self):
        return str(self.value)

    def __add__(self, other):
        if isinstance(other, Value):
            out = Value(self.value + other.value)

            def add_backprop():
                self.gradient += out.gradient
                other.gradient += out.gradient
            
            out._backprop = add_backprop
            out._dependents = {self, other}
        elif isinstance(other, (int, float)):
            out = Value(self.value + other)

            def add_backprop():
                self.gradient += out.gradient
            
            out._backprop = add_backprop
            out._dependents = {self}
        else:
            raise TypeError("Unsupported type for addition")
        return out
    
    def __sub__(self, other):
        if isinstance(other, Value):
            out = Value(self.value - other.value)

            def sub_backprop():
                self.gradient += out.gradient
                other.gradient -= out.gradient

            out._backprop = sub_backprop
            out._dependents = {self, other}
        elif isinstance(other, (int, float)):
            out = Value(self.value - other)

            def sub_backprop():
                self.gradient += out.gradient

            out._backprop = sub_backprop
            out._dependents = {self}
        else:
            raise TypeError("Unsupported type for subtraction")
        return out

    def __mul__(self, other):
        if isinstance(other, Value):
            out = Value(self.value * other.value)

            def mul_backprop():
                self.gradient += other.value * out.gradient
                other.gradient += self.value * out.gradient

            out._backprop = mul_backprop
            out._dependents = {self, other}
        elif isinstance(other, (int, float)):
            out = Value(self.value * other)

            def mul_backprop():
                self.gradient += other * out.gradient

            out._backprop = mul_backprop
            out._dependents = {self}
        else:
            raise TypeError("Unsupported type for multiplication")
        return out
    
    def __truediv__(self, other):
        if isinstance(other, Value):
            out = Value(self.value / other.value)

            def div_backprop():
                self.gradient += (1 / other.value) * out.gradient
                other.gradient -= (self.value / (other.value ** 2)) * out.gradient

            out._backprop = div_backprop
            out._dependents = {self, other}
        elif isinstance(other, (int, float)):
            out = Value(self.value / other)

            def div_backprop():
                self.gradient += (1 / other) * out.gradient

            out._backprop = div_backprop
            out._dependents = {self}
        else:
            raise TypeError("Unsupported type for division")
        return out

    def backprop(self) -> float:
        ordered_vals = []
        visited_vals = set()

        def build_order(val: Value):
            if val not in visited_vals:
                visited_vals.add(val)
                for dep in val._dependents:
                    build_order(dep)
                ordered_vals.append(val)

        build_order(self)

        self.gradient = 1

        for val in reversed(ordered_vals):
            val._backprop()
