from .tensor import Tensor

def Leaky_ReLU(x: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Gradient-Safe leaky relu implementation.
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