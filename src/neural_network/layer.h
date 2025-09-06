#include "../tensor/tensor.h"
#include "../tensor/tensor_operations.h"
#include "../utils/logging.h"
#include <cassert>
#include <random>
#include <variant>
using namespace std;

namespace ml {

/**
 * Abstract class defining template for neural network layers
 */
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

  // Activation function
  string activation = "sigmoid";

  // Map of activation functions
  unordered_map<string, function<shared_ptr<Tensor>>> _activation_functions = {
      {"linear", [](shared_ptr<Tensor> x) { return x; }},
      {"relu", [](shared_ptr<Tensor> x) { return relu_forward(x); }},
      {"tanh", [](shared_ptr<Tensor> x) { return tanh_forward(x); }},
      {"sigmoid", [](shared_ptr<Tensor> x) { return sigmoid_forward(x); }}};

  Layer(int seed, int number_weights, int number_biases,
        vector<int> weights_shape, string init_method, string activation,
        shared_ptr<Logger> logger)
      : random_seed(seed), number_weights(number_weights),
        number_biases(number_biases), activation(activation) {
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

    if (activation != "linear" && activation != "relu" &&
        activation != "leaky_relu" && activation != "tanh" &&
        activation != "sigmoid") {
      throw invalid_argument("Unknown activation function: " + activation);
    }

    _init_methods[init_method]();
  };

  virtual shared_ptr<Tensor> forward(shared_ptr<Tensor> input) = 0;

private:
  unordered_map<string, function<void()>> _init_methods = {
      {"glorot_normal", [this]() { _initialize_glorot_normal(); }},
      {"glorot_uniform", [this]() { _initialize_glorot_uniform(); }},
      {"he_normal", [this]() { _initialize_he_normal(); }},
      {"he_uniform", [this]() { _initialize_he_uniform(); }},
      {"pytorch", [this]() { _initialize_pytorch(); }}};

  unordered_map<string, double> _activation_gain = {
      {"linear", 1.0},
      {"relu", sqrt(2.0)},
      {"leaky_relu", sqrt(2.0 / (1 + 0.01 * 0.01))},
      {"tanh", 5.0 / 3.0},
      {"sigmoid", 1.0}};

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

  void _fill_normal(shared_ptr<Tensor> tensor, double mean, double stddev) {
    normal_distribution<double> distribution(mean, stddev);
    for (int i = 0; i < tensor->values.size(); i++) {
      tensor->values[i] = distribution(random_generator);
    }
  }

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

protected:
  shared_ptr<Tensor> weights = nullptr;
  shared_ptr<Tensor> bias = nullptr;
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
  shared_ptr<Tensor> forward(shared_ptr<Tensor> input) override {
    shared_ptr<Tensor> product_result = batch_matmul_forward(input, weights);
    shared_ptr<Tensor> linear_result = add_tensor_forward(product_result, bias);
    return _activation_functions[activation](linear_result);
  }
};

} // namespace ml
