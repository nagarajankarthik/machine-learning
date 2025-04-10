# Tensors

This module defines and implements a Tensor class. For the current purposes, a tensor is defined as a collection of 2D matrices, all of which have the same number of rows and columns. A key property of a tensor is its shape, which is an array of integers that has at least 3 entries. The last 2 entries of this array define the numbers of rows and columns of each matrix comprising the tensor. They are referred to as the non-batch dimensions. The remaining entries determine the number of matrices that make up the tensor and are known as the batch dimensions. 

For example, a tensor object of shape (5,3,2,4) consists of 15 2D matrices, each of which has 2 rows and 4 columns. 

# Tensor Operations

The objective of implementing a Tensor class is to enable operations such as addition, multiplication and the application of nonlinear functions, including the [standard logistic](https://en.wikipedia.org/wiki/Logistic_function) and [tanh](https://en.wikipedia.org/wiki/Tanh) functions, to be efficiently performed on arrays of matrices. The functions for performing these operations are implemented in the `tensor_operations.h` file. 

It also contains the corresponding functions to compute gradients with respect to the input tensors of these operations. A gradient is defined as the partial derivative of a loss with respect to an input. Gradients are calculated using [reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), as done in [PyTorch](https://pytorch.org/docs/stable/notes/autograd.html).

The various operations that can be performed on tensors are described in greater detail below:

## Addition

Adds corresponding matrices of two input tensors. The two tensors must have the same shape, unless they can be broadcast against one another, following the same rules as those used in [PyTorch](https://pytorch.org/docs/stable/notes/broadcasting.html).

## Multiplication

Multiplies corresponding matrices of two input tensors. The two tensors must have the same shape, unless they can be broadcast against one another, following the same rules as those used in [PyTorch](https://pytorch.org/docs/stable/notes/broadcasting.html).

## Activation Functions

Activation functions are applied separately to each column of every matrix comprising the input tensor. The output and input tensors have the same shape. In the following sections, $\mathbf{x}$ and $\mathbf{y}$ are used to denote the inputs and outputs of an activation function operation. They are both column vectors.

### Rectified Linear Unit (ReLU)

Forward pass:

$$
y_i = \begin{cases}
x_i & \text{if } x_i \ge 0 \\
0 & \text{otherwise}
\end{cases}
$$

Back-propagation:

$$
\frac{\partial L}{\partial x_i} = \begin{cases}
\frac{\partial L}{\partial y_i} \times 1 & \text{if } x_i \ge 0 \\
0 & \text{otherwise}
\end{cases}
$$

### Standard Logistic (also known as Sigmoid)

Forward pass:

$$
y_i = \frac{1}{1 + \exp(-x_i)}
$$

Back-propagation:


$$
y_i(1 + \exp(-x_i)) = 1
$$
$$
\frac{dy_i}{dx_i}(1 + \exp(-x_i)) + y_i(-\exp(-x_i)) = 0
$$

$$
\frac{dy_i}{dx_i}(1 + \exp(-x_i)) = y_i(\exp(-x_i))
$$

$$
\frac{dy_i}{dx_i} = y_i \frac{\exp(-x_i)}{1 + \exp(-x_i)} = y_i(1 - y_i)
$$


$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y_i} \frac{dy_i}{dx_i}
$$

### Hyperbolic Tangent

Forward pass:

$$
y_i = \tanh(x_i) = \frac{\sinh(x_i)}{\cosh(x_i)} = \frac{(\exp(x) - \exp(-x))/2}{\exp(x) + \exp(-x))/2} 
$$

Back-propagation:

$$
\frac{dy_i}{dx_i} = \frac{(\exp(x_i) + \exp(-x_i))(\exp(x_i) + \exp(-x_i)) - (\exp(x_i) - \exp(-x_i))(\exp(x_i) - \exp(-x_i))}{(\exp(x_i) + \exp(-x_i))^2} = 1 - \frac{\sinh^2(x_i)}{\cosh^2(x_i)} = 1 - \tanh^2(x_i) = 1 - y_i^2
$$

### Softmax

Forward pass:

$$
y_j = \frac{\exp(x_j)}{\sum_{k} \exp(x_k)}
$$

Back-propagation:

$$
\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \frac{\delta_{ij} \exp(x_j)\sum_{s} \exp(x_s) - \exp(x_j)\exp(x_i)}{\sum_{s} \exp(x_s) \sum_{t} \exp(x_t)} = \sum_j \frac{\partial L}{\partial y_j} (\delta_{ij} \frac{\exp(x_j)}{\sum_{t} \exp(x_t)} - \frac{\exp(x_j)}{\sum_s \exp(x_s)} \frac{\exp(x_i)}{\sum_t \exp(x_t)})
$$

$$
\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} (\delta_{ij} y_j - y_i y_j) = \frac{\partial L}{\partial y_i} y_i - \sum_j \frac{\partial L}{\partial y_j} y_i y_j = y_i(\frac{\partial L}{\partial y_i} - \sum_j \frac{\partial L}{\partial y_j} y_j)
$$

## Loss Functions

### Cross Entropy

### Mean Squared Error
