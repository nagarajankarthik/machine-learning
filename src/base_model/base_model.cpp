#include "base_model.h"

namespace ml {

BaseModel::BaseModel(nlohmann::json model_parameters, shared_ptr<Logger> logger)
    : logger(logger) {

  random_seed = 0;
  if (model_parameters.contains("random_seed")) {
    int random_seed_input = model_parameters["random_seed"];
    if (random_seed_input > -1)
      random_seed = random_seed_input;
  }

  random_generator = std::mt19937(random_seed);
}

vector<vector<int>>
BaseModel::double_to_int(const vector<vector<double>> &data) {
  int m = data.size();
  int n = data[0].size();

  vector<vector<int>> data_int{};

  for (int i = 0; i < m; i++) {
    vector<int> tmp{};
    for (int j = 0; j < n; j++) {
      int int_element = (int)data[i][j];
      tmp.push_back(int_element);
    }
    data_int.push_back(tmp);
  }
  return data_int;
}

vector<unordered_set<int>>
BaseModel::get_unique_classes(vector<vector<int>> outputs) {

  int number_outputs = outputs[0].size();

  unordered_set<int> tmp{};

  vector<unordered_set<int>> unique_classes(number_outputs, tmp);

  for (int i = 0; i < outputs.size(); i++) {
    for (int j = 0; j < outputs[i].size(); j++) {
      int current_class = outputs[i][j];
      unique_classes[j].insert(current_class);
    }
  }
  return unique_classes;
}

vector<vector<int>>
BaseModel::get_confusion_matrix_single(const vector<vector<int>> &predictions,
                                       const vector<vector<int>> &test_outputs,
                                       unordered_set<int> unique_classes,
                                       int index_output) {
  int number_instances = test_outputs.size();
  int number_classes = unique_classes.size();
  vector<int> tmp(number_classes, 0);
  vector<vector<int>> confusion_matrix(number_classes, tmp);
  vector<int> unique_classes_array(unique_classes.begin(),
                                   unique_classes.end());
  sort(unique_classes_array.begin(), unique_classes_array.end());
  unordered_map<int, int> class_index{};
  for (int i = 0; i < number_classes; i++)
    class_index.insert(make_pair(unique_classes_array[i], i));

  for (int i = 0; i < number_instances; i++) {
    int predicted_class = predictions[i][index_output];
    int ground_truth_class = test_outputs[i][index_output];
    int predicted_index = class_index[predicted_class];
    int ground_truth_index = class_index[ground_truth_class];
    confusion_matrix[ground_truth_index][predicted_index]++;
  }

  return confusion_matrix;
}

template <class T>
string BaseModel::array_2d_to_string(vector<vector<T>> matrix) {

  string result = "";
  int m = matrix.size();
  int n = matrix[0].size();
  for (int i = 0; i < m; i++) {
    string row = "";
    for (int j = 0; j < n; j++) {
      row += to_string(matrix[i][j]) + ", ";
    }
    result += row + "\n";
  }
  return result;
}

// Explicit instantiation of the template for int
template string BaseModel::array_2d_to_string<int>(vector<vector<int>> matrix);

void BaseModel::get_confusion_matrices(
    const vector<vector<double>> &test_predictions_double,
    const vector<vector<double>> &test_labels_double,
    const vector<vector<double>> &train_labels_double) {
  vector<vector<int>> test_predictions = double_to_int(test_predictions_double);
  vector<vector<int>> test_labels = double_to_int(test_labels_double);
  vector<vector<int>> train_labels_int = double_to_int(train_labels_double);

  vector<unordered_set<int>> unique_classes =
      get_unique_classes(train_labels_int);
  vector<unordered_set<int>> unique_classes_test =
      get_unique_classes(test_labels);
  for (int i = 0; i < unique_classes.size(); i++)
    unique_classes[i].insert(unique_classes_test[i].begin(),
                             unique_classes_test[i].end());
  int number_outputs = test_labels[0].size();
  for (int j = 0; j < number_outputs; j++) {
    vector<vector<int>> confusion_matrix = get_confusion_matrix_single(
        test_predictions, test_labels, unique_classes[j], j);
    logger->log(INFO, "Confusion matrix for variable " + to_string(j));
    logger->log(INFO, array_2d_to_string(confusion_matrix));
  }
}

} // namespace ml
