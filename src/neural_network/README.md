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
\mathbf{R}(i,j) = \sum_{a=0}^{F-1} \sum_{b=0}^{F-1} \mathbf{I}(iS + a, jS + b) \mathbf{k}(a,b) \quad 0 \le i \le \frac{W-F}{S}, \quad 0 \le j \le \frac{H-F}{S} 
$$
