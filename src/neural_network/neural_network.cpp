#include "neural_network.h"
#include <memory>
#include <vector>

using namespace std;

namespace ml {
NeuralNetwork::NeuralNetwork(nlohmann::json parameters,
                             shared_ptr<Logger> logger)
    : BaseModel(parameters, logger) {

  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  logger->log(INFO, "Global rank: " + to_string(global_rank));
  logger->log(INFO, "Number of processes: " + to_string(world_size));
  if (parameters.contains("global_batch_size")) {
    int global_batch_size_input = parameters["global_batch_size"];
    if (global_batch_size_input > 0)
      micro_batch_size = global_batch_size_input / world_size;
    else
      logger->log(
          WARNING,
          "The specified batch size, " + to_string(global_batch_size_input) +
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
  if (parameters.contains("max_batches_per_epoch")) {
    int max_batches_per_epoch_input = parameters["max_batches_per_epoch"];
    if (max_batches_per_epoch_input > 0)
      max_batches_per_epoch = max_batches_per_epoch_input;
    else
      logger->log(WARNING, "The specified number of batches per epoch, " +
                               to_string(max_batches_per_epoch_input) +
                               " is invalid. The default number of batches per "
                               "epoch, 1, will be used instead.");
  }
  input_shape = parameters["input_shape"].get<vector<int>>();
  labels_shape = parameters["labels_shape"].get<vector<int>>();
  nlohmann::json layer_specifications = parameters["layers"];
  vector<shared_ptr<Tensor>> optimize_params{};
  int gradient_buffer_size = 0;
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
      vector<int> target_shape =
          layer_parameters["target_shape"].get<vector<int>>();
      target_shape[0] = target_shape[0] / world_size;

      layer = make_shared<ReshapeLayer>(target_shape, logger);
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
      if (layer->weights == nullptr)
        continue;
      gradient_buffer_size += layer->weights->gradients.size();
      gradient_buffer_size += layer->bias->gradients.size();
    }
  }
  gradient_buffer.resize(gradient_buffer_size, 0.0);
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

void NeuralNetwork::collect_gradients() {
  int gradient_buffer_index = 0;
  for (auto &layer : layers) {
    if (layer->weights == nullptr)
      continue;
    for (int i = 0; i < layer->weights->gradients.size(); i++) {
      gradient_buffer[gradient_buffer_index++] = layer->weights->gradients[i];
    }
    for (int i = 0; i < layer->bias->gradients.size(); i++) {
      gradient_buffer[gradient_buffer_index++] = layer->bias->gradients[i];
    }
  }
}

void NeuralNetwork::update_gradients() {
  int gradient_buffer_index = 0;
  for (auto &layer : layers) {
    if (layer->weights == nullptr)
      continue;
    for (int i = 0; i < layer->weights->gradients.size(); i++) {
      layer->weights->gradients[i] = gradient_buffer[gradient_buffer_index++];
    }
    for (int i = 0; i < layer->bias->gradients.size(); i++) {
      layer->bias->gradients[i] = gradient_buffer[gradient_buffer_index++];
    }
  }
}

void NeuralNetwork::communicate_gradients() {
  collect_gradients();
  MPI_Allreduce(MPI_IN_PLACE, gradient_buffer.data(), gradient_buffer.size(),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  update_gradients();
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
  // inference_inputs =
  //     make_shared<Tensor>(input_values, inference_input_shape, logger);
  // inference_labels =
  //     make_shared<Tensor>(label_values, inference_labels_shape, logger);
}
void NeuralNetwork::prepare_inputs_labels(
    const vector<vector<double>> &features,
    const vector<vector<double>> &labels,
    vector<shared_ptr<Tensor>> &prepared_inputs,
    vector<shared_ptr<Tensor>> &prepared_labels) {

  if (features.size() != labels.size()) {
    logger->log(
        ERROR,
        "Features and Labels datasets have different numbers of records.");
    exit(EXIT_FAILURE);
  }
  int number_instances = features.size();
  int number_features = features[0].size();
  int number_outputs = labels[0].size();
  int number_batches = number_instances / micro_batch_size;
  int first_micro_batch_size =
      micro_batch_size + (number_instances % micro_batch_size);

  logger->log(INFO, "Number of instances: " + to_string(number_instances));
  logger->log(INFO, "Number of features: " + to_string(number_features));
  logger->log(INFO, "Number of outputs: " + to_string(number_outputs));
  logger->log(INFO, "Number of batches: " + to_string(number_batches));
  logger->log(INFO,
              "First micro batch size: " + to_string(first_micro_batch_size));

  // prepare input tensor for first batch
  vector<double> first_batch_input(first_micro_batch_size * number_features,
                                   0.0);
  vector<double> first_batch_labels(first_micro_batch_size * number_outputs,
                                    0.0);
  for (int i = 0; i < first_micro_batch_size; i++) {
    for (int j = 0; j < number_features; j++) {
      first_batch_input[i * number_features + j] = features[i][j];
    }
    for (int j = 0; j < number_outputs; j++) {
      first_batch_labels[i * number_outputs + j] = labels[i][j];
    }
  }
  vector<int> first_input_shape(input_shape.begin(), input_shape.end());
  vector<int> first_labels_shape(labels_shape.begin(), labels_shape.end());
  first_input_shape[0] = first_micro_batch_size;
  first_labels_shape[0] = first_micro_batch_size;
  shared_ptr<Tensor> first_input_tensor =
      make_shared<Tensor>(first_batch_input, first_input_shape, logger);
  shared_ptr<Tensor> first_labels_tensor =
      make_shared<Tensor>(first_batch_labels, first_labels_shape, logger);
  prepared_inputs.push_back(first_input_tensor);
  prepared_labels.push_back(first_labels_tensor);

  // prepare input tensor for subsequent batches
  for (int i = 1; i < number_batches; i++) {
    vector<double> batch_input(micro_batch_size * number_features, 0.0);
    vector<double> batch_labels(micro_batch_size * number_outputs, 0.0);
    for (int j = 0; j < micro_batch_size; j++) {
      int ind = first_micro_batch_size + (i - 1) * micro_batch_size + j;
      for (int k = 0; k < number_features; k++) {
        batch_input[j * number_features + k] = features[ind][k];
      }
      for (int k = 0; k < number_outputs; k++) {
        batch_labels[j * number_outputs + k] = labels[ind][k];
      }
    }
    vector<int> train_input_shape(input_shape.begin(), input_shape.end());
    vector<int> train_labels_shape(labels_shape.begin(), labels_shape.end());
    train_input_shape[0] = micro_batch_size;
    train_labels_shape[0] = micro_batch_size;
    shared_ptr<Tensor> input_tensor =
        make_shared<Tensor>(batch_input, train_input_shape, logger);
    shared_ptr<Tensor> labels_tensor =
        make_shared<Tensor>(batch_labels, train_labels_shape, logger);
    prepared_inputs.push_back(input_tensor);
    prepared_labels.push_back(labels_tensor);
  }
}

void NeuralNetwork::prepare_train_input(const vector<vector<double>> &features,
                                        const vector<vector<double>> &labels) {

  if (features.size() != labels.size()) {
    logger->log(
        ERROR,
        "Features and Labels datasets have different numbers of records.");
    exit(EXIT_FAILURE);
  }
  int number_instances = features.size();
  int number_features = features[0].size();
  int number_outputs = labels[0].size();
  int number_batches = number_instances / micro_batch_size;
  int first_micro_batch_size =
      micro_batch_size + (number_instances % micro_batch_size);

  logger->log(INFO, "Number of instances: " + to_string(number_instances));
  logger->log(INFO, "Number of features: " + to_string(number_features));
  logger->log(INFO, "Number of outputs: " + to_string(number_outputs));
  logger->log(INFO, "Number of batches: " + to_string(number_batches));
  logger->log(INFO,
              "First micro batch size: " + to_string(first_micro_batch_size));
  // prepare input tensor for first batch

  vector<double> first_batch_input(first_micro_batch_size * number_features,
                                   0.0);
  vector<double> first_batch_labels(first_micro_batch_size * number_outputs,
                                    0.0);
  for (int i = 0; i < first_micro_batch_size; i++) {
    for (int j = 0; j < number_features; j++) {
      first_batch_input[i * number_features + j] = features[i][j];
    }
    for (int j = 0; j < number_outputs; j++) {
      first_batch_labels[i * number_outputs + j] = labels[i][j];
    }
  }
  vector<int> first_input_shape(input_shape.begin(), input_shape.end());
  vector<int> first_labels_shape(labels_shape.begin(), labels_shape.end());
  first_input_shape[0] = first_micro_batch_size;
  first_labels_shape[0] = first_micro_batch_size;
  shared_ptr<Tensor> first_input_tensor =
      make_shared<Tensor>(first_batch_input, first_input_shape, logger);
  shared_ptr<Tensor> first_labels_tensor =
      make_shared<Tensor>(first_batch_labels, first_labels_shape, logger);
  train_inputs.push_back(first_input_tensor);
  train_labels.push_back(first_labels_tensor);

  // prepare input tensor for subsequent batches
  for (int i = 1; i < number_batches; i++) {
    vector<double> batch_input(micro_batch_size * number_features, 0.0);
    vector<double> batch_labels(micro_batch_size * number_outputs, 0.0);
    for (int j = 0; j < micro_batch_size; j++) {
      int ind = first_micro_batch_size + (i - 1) * micro_batch_size + j;
      for (int k = 0; k < number_features; k++) {
        batch_input[j * number_features + k] = features[ind][k];
      }
      for (int k = 0; k < number_outputs; k++) {
        batch_labels[j * number_outputs + k] = labels[ind][k];
      }
    }
    vector<int> train_input_shape(input_shape.begin(), input_shape.end());
    vector<int> train_labels_shape(labels_shape.begin(), labels_shape.end());
    train_input_shape[0] = micro_batch_size;
    train_labels_shape[0] = micro_batch_size;
    shared_ptr<Tensor> input_tensor =
        make_shared<Tensor>(batch_input, train_input_shape, logger);
    shared_ptr<Tensor> labels_tensor =
        make_shared<Tensor>(batch_labels, train_labels_shape, logger);
    train_inputs.push_back(input_tensor);
    train_labels.push_back(labels_tensor);
  }
}

void NeuralNetwork::prepare_inference_categories(
    const vector<vector<double>> &validation_labels) {
  inference_categories.resize(validation_labels.size());
  fill(inference_categories.begin(), inference_categories.end(),
       vector<double>{0.});
  for (int i = 0; i < validation_labels.size(); i++) {
    int current_category =
        max_element(validation_labels[i].begin(), validation_labels[i].end()) -
        validation_labels[i].begin();
    inference_categories[i][0] = 1. * current_category;
  }
}

void NeuralNetwork::set_data(TrainTestData &&train_test) {
  prepare_inputs_labels(train_test.train_features, train_test.train_labels,
                        train_inputs, train_labels);
  prepare_inputs_labels(train_test.test_features, train_test.test_labels,
                        inference_inputs, inference_labels);
  prepare_inference_categories(train_test.test_labels);
}

void NeuralNetwork::train_epoch() {
  shared_ptr<Tensor> current_value = nullptr;
  shared_ptr<Tensor> loss = nullptr;
  for (int i = 0; i < min(max_batches_per_epoch, (int)train_inputs.size());
       i++) {
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
    MPI_Barrier(MPI_COMM_WORLD);
    communicate_gradients();
    optimizer->step();
  }
}

shared_ptr<Tensor> NeuralNetwork::validate() {
  shared_ptr<Tensor> current_value = nullptr;
  shared_ptr<Tensor> loss = nullptr;
  shared_ptr<Tensor> predicted_tensor = nullptr;
  double total_loss = 0.0;
  for (int i = 0; i < inference_inputs.size(); i++) {
    current_value = inference_inputs[i];
    ForwardParams forward_params{current_value, false};

    for (auto &layer : layers) {
      current_value = layer->forward(forward_params);
      forward_params.input = current_value;
    }
    predicted_tensor =
        i == 0 ? current_value
               : concatenate_forward(predicted_tensor, current_value, 0);
    loss = loss_function(current_value, inference_labels[i]);
    for (int i = 0; i < loss->values.size(); i++) {
      if (loss->values[i] < 0.0)
        logger->log(ERROR, to_string(i) + ": " + to_string(loss->values[i]));
      total_loss += loss->values[i];
    }
  }
  logger->log(INFO, "Validation loss at epoch " + to_string(current_epoch) +
                        " on rank " + to_string(global_rank) + ": " +
                        to_string(total_loss));
  return predicted_tensor;
}

void NeuralNetwork::fit() {
  for (int i = 0; i < number_epochs; i++) {
    current_epoch = i + 1;
    train_epoch();
    evaluate();
  }
}

// TODO: Currently included as a placeholder to enable compilation.
vector<vector<double>> NeuralNetwork::predict() {
  shared_ptr<Tensor> current_value = nullptr;
  shared_ptr<Tensor> loss = nullptr;
  ForwardParams forward_params{inference_inputs[0], false};
  for (auto &layer : layers) {
    current_value = layer->forward(forward_params);
    forward_params.input = current_value;
  }
  loss = loss_function(current_value, inference_labels[0]);
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
  shared_ptr<Tensor> predicted_tensor = validate();
  vector<vector<double>> predictions_categories =
      get_categories(predicted_tensor);
  int number_outputs =
      predicted_tensor->shape[predicted_tensor->shape.size() - 1];
  vector<vector<double>> all_categories(number_outputs, vector<double>{0.});
  for (int i = 0; i < number_outputs; i++) {
    all_categories[i][0] = 1. * i;
  }
  // TODO: Consider including the actual training labels for the last argument
  logger->log(INFO, "Confusion matrices for epoch " + to_string(current_epoch) +
                        " on rank " + to_string(global_rank));
  get_confusion_matrices(predictions_categories, inference_categories,
                         all_categories);
}
} // namespace ml
