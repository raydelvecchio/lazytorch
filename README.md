# Simple Autograd
Custom autogradient implementation. Written in Raw python, no external imports.

# The `Value` Class
* Main building block for this autograd repo; smallest unit available
* Supports `add`, `subtract`, `multiply`, and `divide` operations
    * No other operations right now for simplicity
* All internal values are floats, no support for complex numbers
* All gradients automatically calculated upon operation (no `requires_grad` type toggle for simplicity)
* Calculated gradients are stored in each `Value` directly; must access `Value.gradient` to see them
* Like PyTorch, we cannot automatically zero out gradients and must use the `zero()` function to do so
* Multi-line operations are totally fine in line with the chain rule, however, changing of previous values will break the calculation graph

    * ```python
        # this is ok!
        a = Value(2.0)
        b = Value(3.0)
        f = b + b * a
        z = f * b
        z.backward()
        ```

    * ```python
        # this is not ok
        a = Value(2.0)
        b = Value(3.0)
        f = b + b * a
        b = b * a
        f.backward()
        ```

### Gradient Calculation Code Flow
We'll use addition as an example here for simplicity.
1. `Value` class instances are created with numerical values assigned to them. 
2. Upon instantiation we initialized the value itself, the gradient, a set `_prev`, which will track the gradients used to calculate `self`'s gradient, and a function `_backward`, which will be used to store the last gradient calculation function for a given operation. 
3. Two `Value` objects A and B are added together. 
4. The resulting value of a forward pass (calculation of addition) is returned as another `Value` object C.
5. The backward function for addition is set as the `_backward` value. This backward function is where we define how to calculate gradients of the `Value` A, as well as the `Value` B added to it, for addition (if we were not doing addition, it'd be for the other operation). The backward function also applies the gradient updates to both `Value` A and B. We must update the gradients of both to track the operation so they can flow through the chain rule during backpropagation.
6. The `_prev` set is updated to be equal to the `Value` A, as well as the `Value` B. This is because both `Value` objects are responsible for calculating each other's gradient.
7. We decide we want to find the gradients, so we call `.backward()` on the final output `Value` C. 
8. Starting with `Value` C, we use depth-first search to construct a list, from first operation -> last operation, of `Value` objects. This list represents all `Value` objects in the order they were used to perform the calculations that result in C, in this case, addition.
9. We set the gradient of C to be = 1. This must occur before backprop calculation, since we use this gradient in it. We are calculating all gradients with respect to the final output, which in this case is C, so this makes sense. In a deep learning scenario, this final output is typically the loss obtained from forward prop!
10. We reverse the depth-first list of `Value` objects, so it is now from last operated `Value` -> first operated `Value`. Iterating through this list, we call each `_backprop` function in each `Value`, which was set in step 5. Again, this represents the gradient calculation / setting of each gradient for the last operation occurring on the `Value`. This step is equivalent to calculating the chain rule.
11. To access the gradients of A, B, and C, all with respect to C, you will access `[A | B | C].gradient` to see it!

### Example Usage and Gradient Calculation:
```python
a = Value(2.0)
b = Value(3.0)
f = b + b * a
f.backward()
print(f'Gradient of a: {a.gradient}')
print(f'Gradient of b: {b.gradient}')
print(f'Gradient of f: {f.gradient}')
```

### Derivation:

Given the function:
$$ f = b + b \cdot a $$

Where:
- a = 2.0
- b = 3.0

First, we compute the value of f:

$$ f = b + b \cdot a $$
$$ f = 3.0 + 3.0 \cdot 2.0 $$
$$ f = 3.0 + 6.0 $$
$$ f = 9.0 $$

To find the gradient of f with respect to a, we take the partial derivative of f with respect to a:

$$ \frac{\partial f}{\partial a} = \frac{\partial}{\partial a} (b + b \cdot a) $$
$$ \frac{\partial f}{\partial a} = 0 + b $$
$$ \frac{\partial f}{\partial a} = b $$

Substituting b = 3.0:

$$ \frac{\partial f}{\partial a} = 3.0 $$

So, the gradient with respect to a is 3.0.

To find the gradient of f with respect to b, we take the partial derivative of f with respect to b:

$$ \frac{\partial f}{\partial b} = \frac{\partial}{\partial b} (b + b \cdot a) $$
$$ \frac{\partial f}{\partial b} = 1 + a $$

Substituting a = 2.0:

$$ \frac{\partial f}{\partial b} = 1 + 2.0 $$
$$ \frac{\partial f}{\partial b} = 3.0 $$

So, the gradient with respect to b is 3.0.

The gradient of f with respect to itself is 1:

$$ \frac{\partial f}{\partial f} = 1 $$

- The value of f is 9.0.
- The gradient of f with respect to a is 3.0.
- The gradient of f with respect to b is 3.0.
- The gradient of f with respect to itself is 1.
