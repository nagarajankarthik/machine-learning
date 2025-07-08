# Design considerations for neural networks module

This document explains the design considerations for the `neural_network` module. A neural network is considered to be a collection of layers. A layer receives an input [tensor](../tensor/README.md) and performs one or more operations on it to obtain an output tensor. There can be different types of layers such as linear fully connected, convolutional, recurrent and multi-head attention.

In terms of programming implementation, the `layer.h` header file defines an abstract class for a neural network layer. Any concrete implementation of a layer must inherit from this class and override the `forward` function. Since neural network layers have been designed to work with tensors, the backward pass is automatically taken of by the back-propagation functions implemented in the `tensor_operations.h` header file.

The rest of this document provides additional details about the methods used to implement the different types of layers.

## Linear fully connected layer

A linear fully connected layer performs a linear transformation on its input tensor followed by the application of a user-specified activation function. This operation can be mathematically represented as

$$
\mathbf{z} = f(\mathbf{W} \mathbf{x} + \mathbf{b})
$$

## Convolutional layer with a single channel

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

1. The stride, $S$, will be replaced with a dilation coefficient, $d$.
2. A padding of $F-1$ will be added to the result of the convolution at both ends of the resulting matrix.

As per the first change, the expression for convolution can be re-written as

$$
\mathbf{R}(i,j) = \gamma(i, j) \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \mathbf{I}(i + a, j + b) \mathbf{k}(a,b) + t \quad 0 \le i \le \lfloor W-F \rfloor, \quad 0 \le j \le \lfloor H-F \rfloor, \quad \gamma(i, j) = \begin{cases} 1 & \text{if } \mod(i, S) = 0 \text{ and } \mod(j, S) = 0 \\ 0 & \text{otherwise} \end{cases}
$$

The above expression indicates that the convolution is always performed with stride 1. Therefore, the output matrix $\mathbf{R}$ will always have $W - F + 1$ columns and $H - F + 1$ rows. The inclusion of the dilation factor, $\gamma(i, j)$, ensures that entries of the output matrix that would not exist when using the actual value of stride are set to zero. In the actual implementation, the first equation that explicitly includes the stride is used in the forward pass. The presence of additional zero entries that would be obtained using the last of the above equations is accounted for in the backward pass.

With this change, the partial derivative of loss with respect to the input is given by

$$
\frac{\partial L}{\partial \mathbf{I}(x, y)} = \sum_{i=0}^{W - F} \sum_{j = 0}^{H - F} \frac{\partial L}{\partial \mathbf{R}(i, j)}\frac{\partial \mathbf{R}(i, j)}{\partial \mathbf{I}(x, y)} \\
= \sum_{i=0}^{W - F} \sum_{j=0}^{H - F} \gamma(i, j) \frac{\partial L}{\partial \mathbf{R}(i, j)} \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \delta_{i + a, x} \delta_{j + b, y} \mathbf{k}(a, b) \\
= \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} d(x - a, y - b) \frac{\partial L}{\partial \mathbf{R}(x - a, y - b)} \mathbf{k}(a, b) \\
$$

One can introduce the change of variables $a' = (F - 1) - a$ and $b' = (F - 1) - b$ in the last expression of the above equation to obtain

$$
\frac{\partial L}{\partial \mathbf{I}(x, y)}
= \sum_{a' = 0}^{F - 1} \sum_{b' = 0}^{F - 1} d(x - (F - 1) + a', y - (F - 1) + b') \frac{\partial L}{\partial \mathbf{R}(x - (F - 1) + a', y - (F - 1) + b')} \mathbf{k}((F - 1) - a', (F - 1) -  b') \\
$$

The above equation still cannot be interpreted as a convolution since the sum cannot be taken over all values of $a'$ and $b'$ for certain values of $x$ and $y$. For example, $a'$ can only be $F - 1$ when $x = 0$ and $b'$ can only be $F - 1$ when $y = 0$. This shortcoming can be remedied by the introduction of padding of $F - 1$ at both ends of the matrix $\mathbf{R}$.

This implies that $x$ is replaced by $x + (F - 1)$ and $y$ is replaced by $y + (F - 1)$. The above equation becomes

$$
\frac{\partial L}{\partial \mathbf{I}(x, y)}
= \sum_{a = 0}^{F - 1} \sum_{b = 0}^{F - 1} d(x + a, y + b) \frac{\partial L}{\partial \mathbf{R}(x + a, y + b)} \mathbf{k}_r(a, b) \\
$$

, where $a'$ and $b'$ have been replaced with $a$ and $b$ for notational convenience. Also, $\mathbf{k}_r(a, b)$ is defined as $\mathbf{k}((F - 1) - a, (F - 1) - b)$.

The incorporation of padding of $F - 1$ at both ends of the matrix $\mathbf{R}$ ensures that the indices into $\frac{\partial L}{\partial \mathbf{R}}$ are always valid.
Notice that the last equation is now of the same form as the first equation in this section defining the convolution operation.

In the actual implementation, the forward pass is performed using the first equation for the convolution operation in this sub-section that includes the stride $S$ but not the dilation factor $d$. This implies that the convolution result $R$ will have $\frac{H - F}{S} + 1$ rows and $\frac{W - F}{S} + 1$ columns. Next, dilation is performed along the width and height dimensions by inserting $S - 1$ zeros following all but the last entry. These zeros reflect the elements that were 'missed' as a result of $S$ being greater than one. Mathematically the number of elements along the width dimension changes as

$$
\frac{W - F}{S} + 1 \implies \frac{W - F}{S} \implies W - F \implies W - F + 1
$$


This procedure effectively yields the values of $d(x + a, y + b) \frac{\partial L}{\partial \mathbf{R}(x + a, y + b)}$ by setting $\frac{\partial L}{\partial \mathbf{R}(x + a, y + b)} = 0$, whenever $d(x + a, y + b) = 0$ and leaving the remaining values unchanged.

Finally, the partial derivative of loss with respect to the kernel is given by

$$
\frac{\partial L}{\partial \mathbf{k}(c, d)} = \sum_{i=0}^{\lfloor \frac{W - F}{S} \rfloor} \sum_{j=0}^{\lfloor \frac{H - F}{S} \rfloor} \frac{\partial L}{\partial \mathbf{R}(i, j)} \frac{\partial \mathbf{R}(i, j)}{\partial \mathbf{k}(c, d)} \\
= \sum_{i=0}^{\lfloor \frac{W - F}{S} \rfloor} \sum_{j=0}^{\lfloor \frac{H - F}{S} \rfloor} \frac{\partial L}{\partial \mathbf{R}(i, j)} \mathbf{I}(iS + c, jS + d)
$$

In the special case $S = 1$, the above equation reduces to

$$
\frac{\partial L}{\partial \mathbf{k}(c, d)} = \sum_{i=0}^{W - F} \sum_{j=0}^{ H - F} \frac{\partial L}{\partial \mathbf{R}(i, j)} \frac{\partial \mathbf{R}(i, j)}{\partial \mathbf{k}(c, d)} \\
= \sum_{i=0}^{W - F} \sum_{j=0}^{ H - F} \frac{\partial L}{\partial \mathbf{R}(i, j)} \mathbf{I}(c + i, d + j)
$$

, which is a convolution of the input with the partial derivative of the loss with respect to the output.

If $S > 1$, one follows the same procedure as above with the only difference being that differentiation is performed with respect to the filter instead of the input. This yields

$$
\frac{\partial L}{\partial \mathbf{k}(c, d)} = \sum_{i=0}^{W - F} \sum_{j = 0}^{H - F} \frac{\partial L}{\partial \mathbf{R}(i, j)}\frac{\partial \mathbf{R}(i, j)}{\partial \mathbf{k}(c, d)} \\
= \sum_{i=0}^{W - F} \sum_{j=0}^{H - F} \gamma(i, j) \frac{\partial L}{\partial \mathbf{R}(i, j)} \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \mathbf{I}(i + a, j + b) \delta_{a, c} \delta_{j, d} \\
= \sum_{i=0}^{W - F} \sum_{j=0}^{H - F} \gamma(i, j) \frac{\partial L}{\partial \mathbf{R}(i, j)}  \mathbf{I}(c + i, j + d) \\
$$

, which represents a convolution with $\gamma(i, j)*\frac{\partial L}{\partial \mathbf{R}(i, j)}$ as the filter and $\mathbf{I}(c + i, j + d)$ as the input.



## Convolutional layer with multiple channels

In general, the convolution input will be a tensor with shape $(N, H, W, C)$, where $N$, $H$, $W$ and $C$ denote the numbers of training instances in the current batch, rows and columns per channel, and the number of channels respectively. The tensor representing the convolution kernel will have dimensions of $(Q, F, F, C)$, where $Q$ and $F$ denote the number of filters and the filter size respectively.

In such cases, the convolution operation is defined as

$$
\mathbf{R}(v, i, j, f) = \sum_{p = 0}^{C - 1} \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \mathbf{I}(v, iS + a, jS + b, p) \mathbf{k}(f, a, b, p) + t \quad 0 \le i \le \lfloor \frac{W-F}{S} \rfloor, \quad 0 \le j \le \lfloor \frac{H-F}{S} \rfloor \quad 0 \le v < N, \quad 0 \le f < Q
$$

As per the alternative formulation, one obtains

$$
\mathbf{R}(v, i, j, f) = \gamma(i, j) \sum_{p = 0}^{C - 1} \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \mathbf{I}(v, i + a, j + b, p) \mathbf{k}(f, a, b, p) + t \quad 0 \le i \le W - F , \quad 0 \le j \le H - F , \quad \gamma(i, j) = \begin{cases} 1 & \text{if } \mod(i, S) = 0 \text{ and } \mod(j, S) = 0 \\ 0 & \text{otherwise} \end{cases}
$$

The gradients are given by:

$$
\begin{flalign*}
\frac{\partial L}{\partial \mathbf{I}(v, x, y, r)} &= \sum_{w = 0}^{N - 1} \sum_{i=0}^{W - F} \sum_{j = 0}^{H - F} \sum_{g = 0}^{Q - 1} \frac{\partial L}{\partial \mathbf{R}(w, i, j, g)}\frac{\partial \mathbf{R}(w, i, j, g)}{\partial \mathbf{I}(v, x, y, r)} \\
&= \sum_{w = 0}^{N - 1} \sum_{i=0}^{W - F} \sum_{j=0}^{H - F} \sum_{g = 0}^{Q - 1} \gamma(i, j) \frac{\partial L}{\partial \mathbf{R}(w, i, j, g)} \sum_{p = 0}^{C - 1} \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \delta_{wv} \delta_{i + a, x} \delta_{j + b, y} \delta_{p, r} \mathbf{k}(g, a, b, p) \\
&= \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \sum_{g=0}^{Q-1} \gamma(x - a, y - b) \frac{\partial L}{\partial \mathbf{R}(v, x - a, y - b, g)} \mathbf{k}(g, a, b, r)
\end{flalign*}
$$


Performing the same steps as above yields

$$
\frac{\partial L}{\partial \mathbf{I}(v, x, y, r)}
= \sum_{a = 0}^{F - 1} \sum_{b = 0}^{F - 1} \sum_{g=0}^{Q-1} \gamma(x + a, y + b) \frac{\partial L}{\partial \mathbf{R}(v, x + a, y + b, g)} \mathbf{k}_r(g, a, b, r) \\
$$

which can again be interpreted as a convolution with a flipped kernel.


Similarly,

$$
\begin{flalign*}
\frac{\partial L}{\partial \mathbf{k}(h, c, d, r)} &= \sum_{w = 0}^{N - 1} \sum_{i=0}^{W - F} \sum_{j = 0}^{H - F} \sum_{g=0}^{Q-1} \frac{\partial L}{\partial \mathbf{R}(w, i, j, g)}\frac{\partial \mathbf{R}(w, i, j, g)}{\partial \mathbf{k}(h, c, d, r)} \\
&= \sum_{w = 0}^{N - 1} \sum_{i=0}^{W - F} \sum_{j=0}^{H - F} \sum_{g=0}^{Q-1} \gamma(i, j) \frac{\partial L}{\partial \mathbf{R}(w, i, j, g)} \sum_{p = 0}^{C - 1}\sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \mathbf{I}(i + a, j + b, p) \delta_{a, c} \delta_{b, d} \delta_{p, r} \delta_{gh} \\
&= \sum_{w = 0}^{N - 1} \sum_{i=0}^{W - F} \sum_{j=0}^{H - F} \gamma(i, j) \frac{\partial L}{\partial \mathbf{R}(w, i, j, g)}  \mathbf{I}(c + i, j + d, r) \\
\end{flalign*}
$$

References:

- [Carnegie Mellon University Notes](https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf)
- [Stanford CS231n lecture notes (html)](https://cs231n.github.io/convolutional-networks/)
- [Stanford CS231n lecture notes (slides)](https://cs231n.stanford.edu/slides/2016/winter1516_lecture11.pdf)
- [Medium Article](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)
