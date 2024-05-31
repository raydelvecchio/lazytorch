from .value import Value
from .tensor import Tensor

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
