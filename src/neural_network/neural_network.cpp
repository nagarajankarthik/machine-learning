#include "neural_network.h"
#include <memory>
#include <vector>

using namespace std;

namespace ml {
NeuralNetwork::NeuralNetwork(nlohmann::json parameters,
                             shared_ptr<Logger> logger)::BaseModel(parameters,
                                                                   logger) {

  if (parameters.contains("batch_size")) {
    int batch_size_input = parameters["batch_size"];
    if (batch_size_input > 0)
      batch_size = batch_size_input;
    else
      logger->log(
          WARNING,
          "The specified batch size, " + to_string(batch_size_input) +
              " is invalid. The default batch size of 1 will be used instead.");
  }
  input_shape = parameters["input_shape"].get<vector<int>>();
  labels_shape = parameters["labels_shape"].get<vector<int>>();
  nlohmann::json layer_specifications = input_parameters["layers"];
  vector<shared_ptr<Tensor>> optimize_params{};
  for (nlohmann::json::iterator it = layer_specifications.begin();
       it != layer_specifications.end(); ++it) {
    nlohmann::json layer_parameters = *it;
    string layer_type = layer_parameters["type"];
    shared_ptr<Layer> layer = nullptr;
    switch (layer_type) {
    case "fully_connected":
      layer = make_shared<FullConnectedLayer>(
          random_seed, layer_parameters["number_inputs"],
          layer_parameters["number_outputs"], layer_parameters["init_method"],
          layer_parameters["activation"], logger);
      optimize_params.push_back(layer->weights);
      optimize_params.push_back(layer->bias);
      break;
    case "convolution":
      layer = make_shared<ConvolutionalLayer>(
          random_seed, layer_parameters["input_channels"],
          layer_parameters["output_channels"],
          layer_parameters["kernel_height"], layer_parameters["kernel_width"],
          layer_parameters["stride"], layer_parameters["padding"],
          layer_parameters["dilation_kernel"], layer_parameters["init_method"],
          layer_parameters["activation"], logger);
      optimize_params.push_back(layer->weights);
      optimize_params.push_back(layer->bias);
      break;
    case "pooling":
      layer = make_shared<PoolingLayer>(
          layer_parameters["kernel_height"], layer_parameters["kernel_width"],
          layer_parameters["stride"], layer_parameters["padding"],
          layer_parameters["dilation_kernel"], layer_parameters["pooling_type"],
          logger);
      break;
    case "reshape":
      layer =
          make_shared<ReshapeLayer>(layer_parameters["target_shape"], logger);
      break;
    case "batch_norm":
      layer = make_shared<BatchNormLayer>(layer_parameters["momentum"],
                                          layer_parameters["number_features"], ,
                                          logger);
      optimize_params.push_back(layer->weights);
      optimize_params.push_back(layer->bias);
      break;
    default:
      logger->log(WARNING, "Unknown layer type: " + layer_type);
      break;
    }
    if (layer != nullptr) {
      layers.push_back(layer);
    }
  }
  nlohmann::json optimizer_type = input_parameters["optimizer"];
  switch (optimizer_type) {
  case "sgd":
    optimizer = make_shared<SGDOptimizer>(optimize_params, logger);
    break;
  default:
    logger->log(WARNING,
                "Unknown optimizer type: " + to_string(optimizer_type));
    logger->log(WARNING, "Defaulting to SGD optimizer.");
    optimizer = make_shared<SGDOptimizer>(optimize_params, logger);
    break;
  }
}

void NeuralNetwork::prepare_input(vector<vector<double>> features,
                                  vector<vector<double>> labels) {

  if (features.size() != labels.size()) {
    logger->log(
        ERROR,
        "Features and Labels datasets have different numbers of records.");
    exit(EXIT_FAILURE);
  }
  int number_training_examples = features.size();
  int number_features = features[0].size();
  int number_outputs = labels[0].size();
  int number_batches = number_training_examples / batch_size;
  int first_batch_size = batch_size + (number_training_examples % batch_size);

  // prepare input tensor for first batch
  vector<double> first_batch_input(first_batch_size * number_features, 0.0);
  vector<double> first_batch_labels(first_batch_size * number_outputs, 0.0);
  for (int i = 0; i < first_batch_size; i++) {
    for (int j = 0; j < number_features; j++) {
      first_batch_input[i * number_features + j] = features[i][j];
      first_batch_labels[i * number_outputs + j] = labels[i][j];
    }
  }
  vector<int> first_input_shape(input_shape.begin(), input_shape.end());
  vector<int> first_labels_shape(labels_shape.begin(), labels_shape.end());
  first_input_shape[0] = first_batch_size;
  first_labels_shape[0] = first_batch_size;
  shared_ptr<Tensor> first_input_tensor =
      make_shared<Tensor>(first_batch_input, first_input_shape, logger);
  shared_ptr<Tensor> first_labels_tensor =
      make_shared<Tensor>(first_batch_labels, first_labels_shape, logger);
  inputs.push_back(first_input_tensor);
  labels.push_back(first_labels_tensor);

  // prepare input tensor for subsequent batches
  for (int i = 1; i < number_batches; i++) {
    vector<double> batch_input(batch_size * number_features, 0.0);
    vector<double> batch_labels(batch_size * number_outputs, 0.0);
    for (int j = 0; j < batch_size; j++) {
      for (int k = 0; k < number_features; k++) {
        batch_input[j * number_features + k] = features[i * batch_size + j][k];
        batch_labels[j * number_outputs + k] = labels[i * batch_size + j][k];
      }
    }
    shared_ptr<Tensor> input_tensor =
        make_shared<Tensor>(batch_input, input_shape, logger);
    shared_ptr<Tensor> labels_tensor =
        make_shared<Tensor>(batch_labels, labels_shape, logger);
    inputs.push_back(input_tensor);
    labels.push_back(labels_tensor);
  }
}

void NeuralNetwork::train_step() {
  shared_ptr<Tensor> current_value = nullptr;

  for (auto &batch_input : inputs) {
    current_value = batch_input;
    for (auto &layer : layers) {
      current_value = layer->forward(current_value);
    }
  }
}

} // namespace ml
