# Design considerations for neural networks module

This document explains the design considerations for the `neural_network` module. A neural network is considered to be a collection of layers. A layer receives an input [tensor](../tensor/README.md) and performs one or more operations on it to obtain an output tensor. There can be different types of layers such as linear fully connected, convolutional, recurrent and multi-head attention. 

In terms of programming implementation, the `layer.h` header file defines an abstract class for a neural network layer. Any concrete implementation of a layer must inherit from this class and override the `forward` function. Since neural network layers have been designed to work with tensors, the backward pass is automatically taken of by the back-propagation functions implemented in the `tensor_operations.h` header file.

The rest of this document provides additional details about the methods used to implement the different types of layers.

## Linear fully connected layer

A linear fully connected layer performs a linear transformation on its input tensor followed by the application of a user-specified activation function. This operation can be mathematically represented as 

$$
\mathbf{z} = f(\mathbf{W} \mathbf{x} + \mathbf{b})
$$

## Convolutional Layer

The convolution of a 2D matrix, $\mathbf{I} \in \mathbb{R}^{W \times H}$, with a 2D kernel, $\mathbf{k} \in \mathbb{R}^{F \times F}$, with stride, $S$, can be mathematically represented as

$$
\mathbf{R}(i,j) = \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \mathbf{I}(iS + a, jS + b) \mathbf{k}(a,b) + t \quad 0 \le i \le \lfloor \frac{W-F}{S} \rfloor, \quad 0 \le j \le \lfloor \frac{H-F}{S} \rfloor 
$$

, where $t$ is the bias term.

The expressions for the upper bounds of $i$ and $j$ are explained at this [link](https://cs231n.github.io/convolutional-networks/). The padding for $\mathbf{I}$ was ignored in the above equation. Back-propagation of the loss through convolutional layers is an important issue that must be considered when when implementing such functionality. The remainder of this section will show that the backward pass through a convolutional layer can itself be implemented as a convolution.

The backward pass through a convolutional layer calculates the partial derivatives $\frac{\partial L}{\partial I(x, y)}$ for all $0 \le x < W$ and $0 \le y < H$, where $L$ is the magnitude of the loss. First note that 

$$
\frac{\partial \mathbf{R}(i, j)}{\partial \mathbf{I}(x, y)} = \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \delta_{iS + a, x} \delta_{jS + b, y} \mathbf{k}(a,b) \quad 0 \le i \le \lfloor \frac{W-F}{S} \rfloor, \quad 0 \le j \le \lfloor \frac{H-F}{S} \rfloor
$$

where $\delta_{i, j}$ is the Kronecker delta function. The chain rule implies that 

$$
\frac{\partial L}{\partial \mathbf{I}(x, y)} = \sum_{i=0}^{\lfloor \frac{W-F}{S} \rfloor} \sum_{j=0}^{\lfloor \frac{H-F}{S} \rfloor} \frac{\partial L}{\partial \mathbf{R}(i, j)}\frac{\partial \mathbf{R}(i, j)}{\partial \mathbf{I}(x, y)} \\
= \sum_{i=0}^{\lfloor \frac{W-F}{S} \rfloor} \sum_{j=0}^{\lfloor \frac{H-F}{S} \rfloor} \frac{\partial L}{\partial \mathbf{R}(i, j)} \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \delta_{iS + a, x} \delta_{jS + b, y} \mathbf{k}(a, b) \\
= \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \frac{\partial L}{\partial \mathbf{R}(\frac{x - a}{S}, \frac{y - b}{S})} \mathbf{k}(a, b) \\
$$

In the last expression, the sum over $a$ and $b$ is taken only in cases where $\frac{x - a}{S}$ and $\frac{y - b}{S}$ are integers. In addition, the constraints $0 \le \frac{x - a}{S} \le \lfloor \frac{W-F}{S} \rfloor$ and $0 \le \frac{y - b}{S} < \lfloor \frac{H-F}{S} \rfloor$ must be satisfied. 

To formulate the backward pass in terms of convolutions, the following two changes will be made:

\begin{itemize}
    \item The stride, $S$, will be replaced with a dilation coefficient, $d$.
    \item A padding of $F-1$ will be added to the result of the convolution at both ends of the resulting matrix.
\end{itemize}

As per the first change, the expression for convolution can be re-written as 

$$
\mathbf{R}(i,j) = d(i, j) \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \mathbf{I}(i + a, j + b) \mathbf{k}(a,b) + t \quad 0 \le i \le \lfloor W-F \rfloor, \quad 0 \le j \le \lfloor H-F \rfloor, \quad d(i, j) = \begin{cases} 1 & \text{if } \mod(i, S) = 0 \text{ and } \mod(j, S) = 0 \\ 0 & \text{otherwise} \end{cases}
$$

The above expression indicates that the convolution is always performed with stride 1. Therefore, the output matrix $\mathbf{R}$ will always have $W - F + 1$ columns and $H - F + 1$ rows. The inclusion of the dilation factor, $d(i, j)$, ensures that entries of the output matrix that would not exist when using the actual value of stride are set to zero. In the actual implementation, the first equation that explicitly includes the stride is used in the forward pass. The presence of additional zero entries that would be obtained using the last of the above equations is accounted for in the backward pass.

With this change, the partial derivative of loss with respect to the input is given by 

$$
\frac{\partial L}{\partial \mathbf{I}(x, y)} = \sum_{i=0}^{W - F} \sum_{j = 0}^{H - F} \frac{\partial L}{\partial \mathbf{R}(i, j)}\frac{\partial \mathbf{R}(i, j)}{\partial \mathbf{I}(x, y)} \\
= \sum_{i=0}^{W - F} \sum_{j=0}^{H - F} \frac{\partial L}{\partial \mathbf{R}(i, j)} \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \delta_{i + a, x} \delta_{j + b, y} \mathbf{k}(a, b) \\
= \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \frac{\partial L}{\partial \mathbf{R}(x - a, y - b)} \mathbf{k}(a, b) \\
$$

One can introduce the change of variables $a' = (F - 1) - a$ and $b' = (F - 1) - b$ in the last expression of the above equation to obtain 

$$
\frac{\partial L}{\partial \mathbf{I}(x, y)} 
= \sum_{a' = 0}^{F - 1} \sum_{b' = 0}^{F - 1} \frac{\partial L}{\partial \mathbf{R}(x - (F - 1) + a', y - (F - 1) + b')} \mathbf{k}((F - 1) - a', (F - 1) -  b') \\
$$

The above equation still cannot be interpreted as a convolution since the sum cannot be taken over all values of $a'$ and $b'$ for certain values of $x$ and $y$. For example, $a'$ can only be $F - 1$ when $x = 0$ and $b'$ can only be $F - 1$ when $y = 0$. This shortcoming can be remedied by the introduction of padding of $F - 1$ at both ends of the matrix $\mathbf{R}$.

This implies that $x$ is replaced by $ x + (F - 1)$ and $y$ is replaced by $y + (F - 1)$. The above equation becomes

$$
\frac{\partial L}{\partial \mathbf{I}(x, y)} 
= \sum_{a = 0}^{F - 1} \sum_{b = 0}^{F - 1} \frac{\partial L}{\partial \mathbf{R}(x + a, y + b)} \mathbf{k}_r(a, b) \\
$$

, where $a'$ and $b'$ have been replaced with $a$ and $b$ for notational convenience. Also, $\mathbf{k}_r(a, b)$ is defined as $\mathbf{k}((F - 1) - a, (F - 1) - b)$.

The incorporation of padding of $F - 1$ at both ends of the matrix $\mathbf{R}$ ensures that the indices into $\frac{\partial L}{\partial \mathbf{R}}$ are always valid.
Notice that the last equation is now of the same form as the first equation in this section defining the convolution operation. 

Finally, the partial derivative of loss with respect to the kernel is given by 

$$
\frac{\partial L}{\partial \mathbf{k}(c, d)} = \sum_{i=0}^{W - F} \sum_{j=0}^{H - F} \frac{\partial L}{\partial \mathbf{R}(i, j)} \frac{\partial \mathbf{R}(i, j)}{\partial \mathbf{k}(c, d)} \\
= \sum_{i=0}^{W - F} \sum_{j=0}^{H - F} \frac{\partial L}{\partial \mathbf{R}(i, j)} \mathbf{I}(iS + c, jS + d) \\
$$

, which is a convolution of the input with the gradient of the loss with respect to the output.
