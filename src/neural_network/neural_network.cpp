#include "neural_network.h"
#include <memory>
#include <vector>

using namespace std;

namespace ml {
NeuralNetwork::NeuralNetwork(nlohmann::json parameters,
                             shared_ptr<Logger> logger)
    : BaseModel(parameters, logger) {

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
  if (parameters.contains("number_epochs")) {
    int number_epochs_input = parameters["number_epochs"];
    if (number_epochs_input > 0)
      number_epochs = number_epochs_input;
    else
      logger->log(WARNING, "The specified number of epochs, " +
                               to_string(number_epochs_input) +
                               " is invalid. The default number of epochs, 1, "
                               "will be used instead.");
  }
  input_shape = parameters["input_shape"].get<vector<int>>();
  labels_shape = parameters["labels_shape"].get<vector<int>>();
  nlohmann::json layer_specifications = parameters["layers"];
  vector<shared_ptr<Tensor>> optimize_params{};
  for (nlohmann::json::iterator it = layer_specifications.begin();
       it != layer_specifications.end(); ++it) {
    nlohmann::json layer_parameters = *it;
    string layer_type = layer_parameters["type"];
    shared_ptr<Layer> layer = nullptr;
    if (layer_type == "fully_connected") {
      layer = make_shared<FullConnectedLayer>(
          random_seed, layer_parameters["number_inputs"],
          layer_parameters["number_outputs"], layer_parameters["init_method"],
          layer_parameters["activation"], logger);
      optimize_params.push_back(layer->weights);
      optimize_params.push_back(layer->bias);
    } else if (layer_type == "convolution") {
      layer = make_shared<ConvolutionalLayer>(
          random_seed, layer_parameters["input_channels"],
          layer_parameters["output_channels"],
          layer_parameters["kernel_height"], layer_parameters["kernel_width"],
          layer_parameters["stride"], layer_parameters["padding"],
          layer_parameters["dilation_kernel"], layer_parameters["init_method"],
          layer_parameters["activation"], logger);
      optimize_params.push_back(layer->weights);
      optimize_params.push_back(layer->bias);
    } else if (layer_type == "pooling") {
      layer = make_shared<PoolingLayer>(
          layer_parameters["kernel_height"], layer_parameters["kernel_width"],
          layer_parameters["stride"], layer_parameters["padding"],
          layer_parameters["dilation_kernel"], layer_parameters["pooling_type"],
          logger);
    } else if (layer_type == "reshape") {
      layer =
          make_shared<ReshapeLayer>(layer_parameters["target_shape"], logger);
    } else if (layer_type == "batch_norm") {
      layer = make_shared<BatchNormLayer>(layer_parameters["number_features"],
                                          layer_parameters["momentum"],
                                          layer_parameters["axis"], logger);
      optimize_params.push_back(layer->bias);
    } else {
      logger->log(WARNING, "Unknown layer type: " + layer_type);
    }
    if (layer != nullptr) {
      layers.push_back(layer);
    }
  }
  nlohmann::json optimizer_type = parameters["optimizer"]["type"];
  if (optimizer_type == "sgd") {
    optimizer = make_shared<SGDOptimizer>(
        optimize_params, logger, parameters["optimizer"]["learning_rate"],
        parameters["optimizer"]["momentum"]);
  } else {
    logger->log(WARNING,
                "Unknown optimizer type: " + to_string(optimizer_type));
    logger->log(WARNING, "Defaulting to SGD optimizer.");
    optimizer = make_shared<SGDOptimizer>(optimize_params, logger);
  }
  string loss_type = parameters["loss"];
  loss_function = _loss_functions[loss_type];
}

void NeuralNetwork::prepare_inference_input(
    const vector<vector<double>> &features,
    const vector<vector<double>> &labels) {
  if (features.size() != labels.size()) {
    logger->log(
        ERROR,
        "Features and Labels datasets have different numbers of records.");
    exit(EXIT_FAILURE);
  }
  vector<double> input_values(features.size() * features[0].size(), 0.0);
  vector<double> label_values(labels.size() * labels[0].size(), 0.0);
  for (int i = 0; i < features.size(); i++) {
    for (int j = 0; j < features[0].size(); j++) {
      input_values[i * features[0].size() + j] = features[i][j];
    }
    for (int j = 0; j < labels[0].size(); j++) {
      label_values[i * labels[0].size() + j] = labels[i][j];
    }
  }
  vector<int> inference_input_shape(input_shape.begin(), input_shape.end());
  vector<int> inference_labels_shape(labels_shape.begin(), labels_shape.end());
  inference_input_shape[0] = features.size();
  inference_labels_shape[0] = labels.size();
  inference_inputs =
      make_shared<Tensor>(input_values, inference_input_shape, logger);
  inference_labels =
      make_shared<Tensor>(label_values, inference_labels_shape, logger);
}

void NeuralNetwork::prepare_train_input(const vector<vector<double>> &features,
                                        const vector<vector<double>> &labels) {

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
    }
    for (int j = 0; j < number_outputs; j++) {
      first_batch_labels[i * number_outputs + j] = labels[i][j];
    }
  }
  vector<int> first_input_shape(input_shape.begin(), input_shape.end());
  vector<int> first_labels_shape(labels_shape.begin(), labels_shape.end());
  first_input_shape[0] = first_batch_size;
  first_labels_shape[0] = first_batch_size;
  shared_ptr<Tensor> test_tensor =
      make_shared<Tensor>(vector<double>(4, 0.), vector<int>(2, 2), logger);
  shared_ptr<Tensor> first_input_tensor =
      make_shared<Tensor>(first_batch_input, first_input_shape, logger);
  shared_ptr<Tensor> first_labels_tensor =
      make_shared<Tensor>(first_batch_labels, first_labels_shape, logger);
  train_inputs.push_back(first_input_tensor);
  train_labels.push_back(first_labels_tensor);

  // prepare input tensor for subsequent batches
  for (int i = 1; i < number_batches; i++) {
    vector<double> batch_input(batch_size * number_features, 0.0);
    vector<double> batch_labels(batch_size * number_outputs, 0.0);
    for (int j = 0; j < batch_size; j++) {
      int ind = first_batch_size + (i - 1) * batch_size + j;
      for (int k = 0; k < number_features; k++) {
        batch_input[j * number_features + k] = features[ind][k];
      }
      for (int k = 0; k < number_outputs; k++) {
        batch_labels[j * number_outputs + k] = labels[ind][k];
      }
    }
    shared_ptr<Tensor> input_tensor =
        make_shared<Tensor>(batch_input, input_shape, logger);
    shared_ptr<Tensor> labels_tensor =
        make_shared<Tensor>(batch_labels, labels_shape, logger);
    train_inputs.push_back(input_tensor);
    train_labels.push_back(labels_tensor);
  }
}

void NeuralNetwork::set_data(TrainTestData &&train_test) {
  prepare_train_input(train_test.train_features, train_test.train_labels);
  prepare_inference_input(train_test.test_features, train_test.test_labels);
}

void NeuralNetwork::train_epoch(int current_epoch) {
  shared_ptr<Tensor> current_value = nullptr;
  shared_ptr<Tensor> loss = nullptr;
  for (int i = 0; i < train_inputs.size(); i++) {
    optimizer->zero_gradients();
    current_value = train_inputs[i];
    ForwardParams forward_params{current_value, true};
    for (auto &layer : layers) {
      current_value = layer->forward(forward_params);
      forward_params.input = current_value;
    }
    loss = loss_function(current_value, train_labels[i]);
    double total_loss = 0.0;
    for (int i = 0; i < loss->values.size(); i++) {
      if (loss->values[i] < 0.0)
        logger->log(ERROR, to_string(i) + ": " + to_string(loss->values[i]));
      total_loss += loss->values[i];
    }
    logger->log(INFO, "Training loss at epoch " + to_string(current_epoch) +
                          " and batch " + to_string(i + 1) + ": " +
                          to_string(total_loss));
    loss->backward();
    optimizer->step();
  }
}

shared_ptr<Tensor> NeuralNetwork::validate(int current_epoch) {
  shared_ptr<Tensor> current_value = inference_inputs;
  shared_ptr<Tensor> loss = nullptr;
  ForwardParams forward_params{current_value, false};

  for (auto &layer : layers) {
    current_value = layer->forward(forward_params);
    forward_params.input = current_value;
  }
  loss = loss_function(current_value, inference_labels);
  double total_loss = 0.0;
  for (int i = 0; i < loss->values.size(); i++) {
    if (loss->values[i] < 0.0)
      logger->log(ERROR, to_string(i) + ": " + to_string(loss->values[i]));
    total_loss += loss->values[i];
    // logger->log(INFO, to_string(loss->values[i]));
  }
  logger->log(INFO, "Validation loss at epoch " + to_string(current_epoch) +
                        ": " + to_string(total_loss));
  return current_value;
}

void NeuralNetwork::fit() {
  for (int i = 0; i < number_epochs; i++) {
    train_epoch(i + 1);
    validate(i + 1);
  }
}

// TODO: Currently included as a placeholder to enable compilation.
vector<vector<double>> NeuralNetwork::predict() {
  shared_ptr<Tensor> current_value = nullptr;
  shared_ptr<Tensor> loss = nullptr;
  ForwardParams forward_params{inference_inputs, false};
  for (auto &layer : layers) {
    current_value = layer->forward(forward_params);
    forward_params.input = current_value;
  }
  loss = loss_function(current_value, inference_labels);
  vector<vector<double>> predictions{};
  return predictions;
}

// TODO: Add support for predicting multiple output categories
vector<vector<double>>
NeuralNetwork::get_categories(shared_ptr<Tensor> tensor) {
  vector<vector<double>> categories(tensor->shape[0], vector<double>(1, 0.));
  int number_outputs = tensor->shape[tensor->shape.size() - 1];
  for (int i = 0; i < tensor->values.size(); i += number_outputs) {
    double max_value = -1.0;
    int predicted_category = 0;

    for (int j = i; j < i + number_outputs; j++) {
      if (tensor->values[j] > max_value) {
        max_value = tensor->values[j];
        predicted_category = j - i;
      }
    }
    int index = i / number_outputs;
    categories[index][0] = 1. * predicted_category;
  }
  return categories;
}

void NeuralNetwork::evaluate() {
  shared_ptr<Tensor> predicted_tensor = validate(number_epochs);
  vector<vector<double>> predictions_categories =
      get_categories(predicted_tensor);
  vector<vector<double>> actual_categories = get_categories(inference_labels);
  // TODO: Consider including the actual training labels for the last argument
  get_confusion_matrices(predictions_categories, actual_categories,
                         actual_categories);
}
} // namespace ml
