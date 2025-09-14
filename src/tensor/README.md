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


## Axis Norm

Implementation follows the [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf) paper by Ioffe and Szegedy. The results have been validated against the [PyTorch implementation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html). The implementation of axis norm does not include the additional linear transformation mentioned in the above paper.

Assigns the elements of a tensor to disjoint subsets and normalizes the elements in each subset by the corresponding mean and variance. The subsets are formed by extracting all elements of the tensor corresponding to each index of the specified axis.The output tensor has the same shape as the input tensor.

Forward pass:

$$
y_i = \frac{x_i - \mu}{\sqrt {\sigma^2 + \epsilon}}
$$

$$
\mu = \frac{1}{m} \sum_{i = 1}^{m} x_i
$$

$$
\sigma^2 = \frac{1}{m} \sum_{i = 1}^{m} (x_i - \mu)^2
$$

$$
z_i = \gamma y_i + \beta
$$

, where $x_i$ is the element at index $i$ of the current subset, $m$ is the number of elements, and $\mu$ and $\sigma$ denote the mean and standard deviation of the subset respectively. Also, $\gamma$ and $\beta$ are learnable parameters. The overall effect of this operation is to map each element $y_i$ to the corresponding output $z_i$.

Back-propagation:


$$
\frac{\partial L}{\partial \beta} = \sum_{i = 1}^m \frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial \beta} = \sum_{i = 1}^m \frac{\partial L}{\partial z_i}
$$

$$
\frac{\partial L}{\partial \gamma} = \sum_{i = 1}^m \frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial \gamma} = \sum_{i = 1}^m y_i \frac{\partial L}{\partial z_i}
$$

$$
\frac{\partial L}{\partial y_i} = \frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial y_i} = \gamma \frac{\partial L}{\partial z_i}
$$

$$
\frac{\partial \mu}{\partial x_j} = \frac{1}{m}
$$

$$
\frac{\partial \sigma^2}{\partial x_j} = \frac{1}{m} \sum_{i = 1}^{m} 2(x_i - \mu)(\delta_{ij} - \frac{1}{m}) = 2\frac{x_j - \mu}{m} - \frac{2}{m^2} \sum_{i = 1}^{m} (x_i - \mu) = 2\frac{x_j - \mu}{m}
$$

$$
\frac{\partial y_j}{\partial x_i} = \frac{(\delta_{ij} - \frac{1}{m})(\sqrt {\sigma^2 + \epsilon}) - (x_i - \mu)(0.5)(\sigma^2 + \epsilon)^{-1/2}(2)\frac{ x_j - \mu}{m}}{\sigma^2 + \epsilon}
$$

$$
\frac{\partial y_j}{\partial x_i} = \frac{\delta_{ij} - \frac{1}{m}}{\sqrt {\sigma^2 + \epsilon}} - \frac{(x_i - \mu)(x_j - \mu)}{m(\sigma^2 + \epsilon)^{3/2}}
$$

$$
\frac{\partial L}{\partial x_i} = \sum_{j = 1}^m \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i}
$$


$$
\frac{\partial L}{\partial x_i} = \sum_{j = 1}^m \frac{\partial L}{\partial y_j} \frac{\delta_{ij}}{\sqrt {\sigma^2 + \epsilon}} - \frac{1}{m \sqrt {\sigma^2 + \epsilon}} \sum_{j = 1}^m \frac{\partial L}{\partial y_j} -  \frac{x_i - \mu}{(\sigma^2 + \epsilon)^{3/2}} \sum_{j = 1}^m \frac{\partial L}{\partial y_j} (x_j - \mu)
$$

$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y_i} \frac{1}{\sqrt {\sigma^2 + \epsilon}} - \frac{1}{m \sqrt {\sigma^2 + \epsilon}} \sum_{j = 1}^m \frac{\partial L}{\partial y_j} -  \frac{x_i - \mu}{(\sigma^2 + \epsilon)^{3/2}} \sum_{j = 1}^m \frac{\partial L}{\partial y_j} (x_j - \mu)
$$
## Concatenation

Concatenates two tensors along the specified dimension. The two tensors must have the same size for all other dimensions.

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

The cross entropy (CE) loss is used when the target variable is categorical.

Assume there are $C$ classes for a particular categorical variable. The ground truth values are denoted as $t_i$, where $1 \le i \le C$. The value of $t_i$ is 1 for a single value of $i \in [1, C]$ and zero for all other values of $i$. A neural network typically predicts the probability of membership, $p_i$, for each category. They satisfy $\sum_{i = 1}^C p_i = 1$. The cross-entropy loss, CE, is [defined](https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5) as

$$
\text{CE} = - \sum_{i = 1}^C t_i \log(p_i)
$$

During back-propagation, one evaluates

$$
\frac{\partial \text{CE}}{\partial p_i} = - \sum_{i = 1}^C \frac{t_i}{p_i}
$$

Note that only one term comprising the sum on the right side will be non-zero.


### Mean Squared Error

The mean squared error (MSE) loss should be used when predicting the value of a continuous variable. It is defined as

$$
MSE = \sum_{i} (p_i - t_i)^2
$$

where $p_i$ denotes the predicted value of variable $i$ and $t_i$ denotes the corresponding ground truth value.

During back-propagation, one evaluates

$$
\frac{\partial \text{MSE}}{\partial p_i} =  2 (p_i - t_i)
$$

Note the order of the variables in the above equation. If $p_i > t_i$$, increasing $p_i$ results in MSE increasing. This implies $\frac{\partial \text{MSE}}{\partial p_i} > 0$. Conversely, increasing $p_i$ reduces MSE whenever $p_i < t_i$. In this case, $\frac{\partial \text{MSE}}{\partial p_i} < 0$.
