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

Activation functions are applied separately to each column of every matrix comprising the input tensor. The output and input tensors have the same shape. In the following sections, $x$ and $y$ are used to denote the inputs and outputs of an activation function operation. They are both column vectors.

### Rectified Linear Unit (ReLU)




### Standard Logistic (also known as Sigmoid)

### Hyperbolic Tangent

## Loss Functions

### Cross Entropy

### Mean Squared Error
