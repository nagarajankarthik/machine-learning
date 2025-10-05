#include "../tensor/tensor.h"
#include "../tensor/tensor_operations.h"
#include "../utils/logging.h"
#include <cassert>
#include <random>
#include <variant>
using namespace std;

namespace ml {

struct ForwardParams {
  shared_ptr<Tensor> input{nullptr};
  bool train{false};
};

class Layer {
public:
  // Random seed
  int random_seed = 0;

  // Random generator
  std::mt19937 random_generator;

  // Number of values in weights tensor
  int number_weights = 0;

  // Number of values in bias tensor
  int number_biases = 0;

  // Activation function. Default is linear.
  string activation = "linear";

  // Logger
  shared_ptr<Logger> logger = nullptr;

  // Map of activation functions
  unordered_map<string, function<shared_ptr<Tensor>(shared_ptr<Tensor>)>>
      _activation_functions = {
          {"linear", [](shared_ptr<Tensor> x) { return x; }},
          {"relu", [](shared_ptr<Tensor> x) { return relu_forward(x); }},
          {"tanh", [](shared_ptr<Tensor> x) { return tanh_forward(x); }},
          {"sigmoid", [](shared_ptr<Tensor> x) { return sigmoid_forward(x); }},
          {"softmax", [](shared_ptr<Tensor> x) { return softmax_forward(x); }}};

  Layer() = default;

  Layer(shared_ptr<Logger> logger) : logger(logger) {}

  Layer(int seed, int number_weights, int number_biases,
        vector<int> weights_shape, string init_method, string activation,
        shared_ptr<Logger> logger)
      : random_seed(seed), number_weights(number_weights),
        number_biases(number_biases), activation(activation), logger(logger) {
    random_generator = std::mt19937(random_seed);

    // Initialize weights
    vector<double> weights_values(number_weights, 0.0);
    vector<double> bias_values(number_biases, 0.0);
    weights = make_shared<Tensor>(weights_values, weights_shape, logger);
    bias =
        make_shared<Tensor>(bias_values, vector<int>{1, number_biases}, logger);

    if (_init_methods.find(init_method) == _init_methods.end()) {
      throw invalid_argument("Unknown init method: " + init_method);
    }

    if (init_method != "pytorch")
      logger->log(WARNING,
                  "Using non-pytorch initialization method. Correctness "
                  "of initialization is not guaranteed.");

    if (_activation_functions.find(activation) == _activation_functions.end()) {
      throw invalid_argument("Unknown activation function: " + activation);
    }

    _init_methods[init_method]();
  };

  virtual shared_ptr<Tensor> forward(ForwardParams params) = 0;

  shared_ptr<Tensor> weights = nullptr;
  shared_ptr<Tensor> bias = nullptr;

private:
  // Map of initialization methods
  unordered_map<string, function<void()>> _init_methods = {
      {"glorot_normal", [this]() { _initialize_glorot_normal(); }},
      {"glorot_uniform", [this]() { _initialize_glorot_uniform(); }},
      {"he_normal", [this]() { _initialize_he_normal(); }},
      {"he_uniform", [this]() { _initialize_he_uniform(); }},
      {"pytorch", [this]() { _initialize_pytorch(); }}};

  // Map of activation function gains
  unordered_map<string, double> _activation_gain = {
      {"linear", 1.0},
      {"relu", sqrt(2.0)},
      {"leaky_relu", sqrt(2.0 / (1 + 0.01 * 0.01))},
      {"tanh", 5.0 / 3.0},
      {"sigmoid", 1.0}};

  // Get fan in and fan out
  pair<int, int> _get_fan_in_fan_out() {
    int fan_in = weights->shape[0];
    int fan_out = weights->shape[1];
    int receptive_field_size = 1;
    // Convolutional layer case
    if (weights->shape.size() == 4) {
      fan_in = weights->shape[3];
      fan_out = weights->shape[0];
      for (int i = 1; i < weights->shape.size() - 1; i++) {
        receptive_field_size *= weights->shape[i];
      }
    }
    return make_pair(fan_in * receptive_field_size,
                     fan_out * receptive_field_size);
  }

  // Fill tensor with normal distribution
  void _fill_normal(shared_ptr<Tensor> tensor, double mean, double stddev) {
    normal_distribution<double> distribution(mean, stddev);
    for (int i = 0; i < tensor->values.size(); i++) {
      tensor->values[i] = distribution(random_generator);
    }
  }

  // Fill tensor with uniform distribution
  void _fill_uniform(shared_ptr<Tensor> tensor, double lower, double upper) {
    uniform_real_distribution<double> distribution(lower, upper);
    for (int i = 0; i < tensor->values.size(); i++) {
      tensor->values[i] = distribution(random_generator);
    }
  }

  void _initialize_glorot_normal() {
    int input_features = weights->shape[0];
    int output_features = weights->shape[1];
    double stddev = sqrt(2.0 / (input_features + output_features));
    _fill_normal(weights, 0.0, stddev);
  }

  void _initialize_he_normal() {
    int input_features = weights->shape[0];
    double stddev = sqrt(2.0 / input_features);
    _fill_normal(weights, 0.0, stddev);
  }

  void _initialize_glorot_uniform() {
    int input_features = weights->shape[0];
    int output_features = weights->shape[1];
    double limit = sqrt(6.0 / (input_features + output_features));
    _fill_uniform(weights, -limit, limit);
  }

  void _initialize_he_uniform() {
    int input_features = weights->shape[0];
    double limit = sqrt(6.0 / input_features);
    _fill_uniform(weights, -limit, limit);
  }

  void _initialize_pytorch() {
    pair<int, int> fan_in_out = _get_fan_in_fan_out();
    int fan_in = fan_in_out.first;
    assert(fan_in > 0);
    double inv_sqrt_fan_in = 1.0 / sqrt(fan_in);
    double gain = _activation_gain[activation];
    double std = gain * inv_sqrt_fan_in;
    double bound = sqrt(3.0) * std;
    _fill_uniform(weights, -bound, bound);
    _fill_uniform(bias, -inv_sqrt_fan_in, inv_sqrt_fan_in);
  }
};

/**
 * Method of initialization of weights and biases follows the implementation
 * in https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
 * and https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
 * */

class FullConnectedLayer : public Layer {
public:
  FullConnectedLayer(int random_seed, int number_inputs, int number_outputs,
                     string init_method, string activation,
                     shared_ptr<Logger> logger)
      : Layer(random_seed, number_inputs * number_outputs, number_outputs,
              vector<int>{number_inputs, number_outputs}, init_method,
              activation, logger) {}

  /**
   * Function to calculate outputs
   * @param input: Input tensor to layer
   * @return: Output tensor
   */
  shared_ptr<Tensor> forward(ForwardParams forward_params) override {
    shared_ptr<Tensor> input = forward_params.input;
    shared_ptr<Tensor> product_result = batch_matmul_forward(input, weights);
    shared_ptr<Tensor> linear_result = add_tensor_forward(product_result, bias);
    return _activation_functions[activation](linear_result);
  }
};

class ConvolutionalLayer : public Layer {
public:
  // Kernel height
  // Assuming square kernels, kernel_height == kernel_width
  int kernel_height = 0;
  // Kernel width
  int kernel_width = 0;
  // Number of input channels
  int input_channels = 0;
  // Number of output channels (or filters)
  int output_channels = 0;
  // Stride of the convolution
  int stride = 1;
  // Padding added to input
  int padding = 0;
  // Dilation of the kernel in convolution
  int dilation_kernel = 1;

  ConvolutionalLayer(int random_seed, int input_channels, int output_channels,
                     int kernel_height, int kernel_width, int stride,
                     int padding, int dilation_kernel, string init_method,
                     string activation, shared_ptr<Logger> logger)
      : Layer(random_seed,
              input_channels * output_channels * kernel_height * kernel_width,
              output_channels,
              vector<int>{output_channels, kernel_height, kernel_width,
                          input_channels},
              init_method, activation, logger),
        kernel_height(kernel_height), kernel_width(kernel_width),
        input_channels(input_channels), output_channels(output_channels),
        stride(stride), padding(padding), dilation_kernel(dilation_kernel) {
    if (kernel_height <= 0 || kernel_width <= 0) {
      throw invalid_argument("Kernel height and width must be positive");
    }

    if (kernel_height != kernel_width) {
      throw invalid_argument(
          "Only square kernels are currently supported. To relax this "
          "constraint, the convolution function in tensor_operations.h needs "
          "to be modified to allow the different values of padding in height "
          "and width dimensions.");
    }

    if (input_channels <= 0) {
      throw invalid_argument("Number of input channels must be positive");
    }
    if (output_channels <= 0) {
      throw invalid_argument("Number of output channels must be positive");
    }
  }

  /**
   * Function to calculate outputs
   * @param input: Input tensor to layer with shape (batch_size,
   * height_input, width_input, input_channels)
   * @return: Output tensor with shape (batch_size, height_output,
   * width_output, output_channels)
   */
  shared_ptr<Tensor> forward(ForwardParams forward_params) override {
    shared_ptr<Tensor> input = forward_params.input;
    return _activation_functions[activation](
        convolution(input, weights, bias, stride, padding, 1, dilation_kernel));
  }
};

class PoolingLayer : public Layer {
public:
  // Kernel height
  int kernel_height = 0;
  // Kernel width
  int kernel_width = 0;
  // Stride of the pooling operation
  int stride = 1;
  // Padding added to input. The values of padding grid cells are negative
  // infinity and zero for max and average pooling respectively.
  int padding = 0;
  // Dilation of the kernel in pooling.
  int dilation_kernel = 1;
  // Type of pooling operation
  string pooling_type = "max";

  PoolingLayer(int kernel_height, int kernel_width, int stride, int padding,
               int dilation_kernel, string pooling_type,
               shared_ptr<Logger> logger)
      : Layer(logger), kernel_height(kernel_height), kernel_width(kernel_width),
        stride(stride), padding(padding), dilation_kernel(dilation_kernel),
        pooling_type(pooling_type) {
    if (kernel_height <= 0 || kernel_width <= 0) {
      throw invalid_argument("Kernel height and width must be positive");
    }
  }

  /**
   * Function to calculate outputs
   * @param input: Input tensor to layer with shape (batch_size,
   * height_input, width_input, input_channels)
   * @return: Output tensor with shape (batch_size, height_output,
   * width_output, input_channels)
   */
  shared_ptr<Tensor> forward(ForwardParams forward_params) override {
    shared_ptr<Tensor> input = forward_params.input;
    assert(input->shape.size() == 4);
    if (pooling_type == "max") {
      return max_pool(input, kernel_height, kernel_width, stride, padding,
                      dilation_kernel);
    } else if (pooling_type == "average") {
      return average_pool(input, kernel_height, kernel_width, stride, padding,
                          dilation_kernel);
    } else {
      throw invalid_argument("Pooling type must be max or average");
    }
  }
};

class ReshapeLayer : public Layer {
public:
  // Target shape after reshaping
  // The target shape should be compatible with the input shape.
  vector<int> target_shape{};
  ReshapeLayer(vector<int> target_shape, shared_ptr<Logger> logger)
      : Layer(logger), target_shape(target_shape) {}

  /**
   * Function to calculate outputs
   * @param input: Input tensor to layer with arbitrary shape
   * @return: Output tensor whose shape matches target_shape
   */
  shared_ptr<Tensor> forward(ForwardParams forward_params) override {
    shared_ptr<Tensor> input = forward_params.input;
    assert(input->values.size() == std::accumulate(target_shape.begin(),
                                                   target_shape.end(), 1,
                                                   std::multiplies<int>()));
    input->reshape(target_shape);
    return input;
  }
};

class BatchNormLayer : public Layer {
public:
  // Number of features
  int number_features = 0;

  // Number of batches processed thus far
  int number_batches = 0;

  // Axis over which slices of tensor are extracted for normalization
  int axis = 1;

  // Running average for batch means
  double running_average_batch_mean = 0.;

  // Running average for batch standard deviation
  double running_average_batch_std = 0.;

  // Momentum for moving average. Should be between 0 and 1. Otherwise, a
  // simple averaging is performed. See
  // https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
  // for more details. This number refers to the weight applied to the
  // statistics obtained for the most recent batch, when updating the moving
  // average.
  double momentum = 0.1;

  BatchNormLayer(int number_features, double momentum, int axis,
                 shared_ptr<Logger> logger)
      : Layer(logger), number_features(number_features), momentum(momentum),
        axis(axis) {

    vector<double> normalization_parameters_values(2 * number_features, 0.0);
    fill(normalization_parameters_values.begin(),
         normalization_parameters_values.begin() + number_features, 1.0);
    vector<int> weights_bias_shape{1, number_features};
    vector<double> weights_values(number_features, 1.0);
    vector<double> bias_values(number_features, 0.0);
    weights = make_shared<Tensor>(weights_values, weights_bias_shape, logger);
    bias = make_shared<Tensor>(bias_values, weights_bias_shape, logger);
  }

  /**
   * Function to calculate outputs
   * @param input: Input tensor to layer with arbitrary shape
   * @param compute_mean_variance: Boolean indicating whether to compute mean
   * and variance or use the existing moving averages
   * @return: Output tensor after normalization is applied over the specified
   * axis
   */
  shared_ptr<Tensor> forward(ForwardParams forward_params) {

    shared_ptr<Tensor> input = forward_params.input;
    bool train = forward_params.train;

    if (input->shape[axis] != number_features) {
      throw invalid_argument("Error in BatchNormLayer: Input shape does not "
                             "match number of features");
    }

    number_batches += number_features;
    vector<double> batch_means(number_features, running_average_batch_mean);
    vector<double> batch_std(number_features, running_average_batch_std);
    double average_factor = momentum;
    if (average_factor < 0) {
      average_factor = 1. / number_batches;
    }

    int batch_size = 1;
    for (int i = 0; i < input->shape.size(); i++) {
      int multiplier = i == axis ? 1 : input->shape[i];
      batch_size *= multiplier;
    }

    shared_ptr<Tensor> normalized_input = axis_norm_forward(
        input, axis, weights, bias, batch_means, batch_std, train);

    if (train) {
      double batch_means_average =
          std::accumulate(batch_means.begin(), batch_means.end(), 0.0) /
          number_features;
      double batch_std_average =
          std::accumulate(batch_std.begin(), batch_std.end(), 0.0) /
          number_features;
      batch_std_average *= 1. * batch_size / (batch_size - 1);

      // Special treatment for the first set of batches??
      if (number_batches == number_features) {
        running_average_batch_mean = batch_means_average;
        running_average_batch_std = batch_std_average;
      } else {
        running_average_batch_mean =
            average_factor * batch_means_average +
            (1 - average_factor) * running_average_batch_mean;
        running_average_batch_std =
            average_factor * batch_std_average +
            (1 - average_factor) * running_average_batch_std;
      }
    }

    return normalized_input;
  }
};
} // namespace ml
