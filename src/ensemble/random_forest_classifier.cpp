#include "random_forest_classifier.h"

namespace ml {

RandomForestClassifier::RandomForestClassifier(nlohmann::json parameters,
                                               shared_ptr<Logger> logger)
    : BaseModel(parameters, logger), parameters(parameters) {

  if (parameters.contains("number_trees")) {
    int number_trees_input = parameters["number_trees"];
    if (number_trees_input > 0)
      number_trees = number_trees_input;
  }

  random_generator = std::mt19937(random_seed);
}

void RandomForestClassifier::get_bootstrap_sample(
    vector<vector<double>> &features_sample,
    vector<vector<double>> &outputs_sample) {
  int number_instances = train_features.size();
  vector<int> data_indices(number_instances, 0);

  std::uniform_int_distribution<int> distribution_uniform(0,
                                                          number_instances - 1);

  for (int i = 0; i < number_instances; i++) {
    auto random_integer = distribution_uniform(random_generator);
    vector<double> features_random = train_features[random_integer];
    vector<double> outputs_random = train_labels[random_integer];
    features_sample[i].assign(features_random.begin(), features_random.end());
    outputs_sample[i].assign(outputs_random.begin(), outputs_random.end());
  }
}

/**
 * Note the use of std::move(object).member instead of move(object.member)
 * See https://oliora.github.io/2016/02/12/where-to-put-std-move.html
 */
void RandomForestClassifier::set_data(TrainTestData &&train_test) {
  this->train_features = std::move(train_test).train_features;
  this->train_labels = std::move(train_test).train_labels;
  this->test_features = std::move(train_test).test_features;
  this->test_labels = std::move(train_test).test_labels;
}

void RandomForestClassifier::fit() {

  trees.clear();

  int number_instances = train_features.size();
  vector<vector<double>> features_sample{};
  features_sample.assign(train_features.begin(), train_features.end());
  vector<vector<double>> labels_sample{};
  labels_sample.assign(train_labels.begin(), train_labels.end());

  std::vector<std::shared_ptr<DecisionTreeClassifier>> local_trees(
      number_trees);

#pragma omp parallel for
  for (int i = 0; i < number_trees; i++) {
    shared_ptr<DecisionTreeClassifier> classifier_tree(
        new DecisionTreeClassifier(parameters, logger));
    get_bootstrap_sample(features_sample, labels_sample);
    TrainTestData train_test_sample{features_sample, labels_sample,
                                    test_features, test_labels};
    classifier_tree->set_data(std::move(train_test_sample));
    classifier_tree->fit();
    local_trees[i] = classifier_tree;
  }

  trees = std::move(local_trees);
}

vector<vector<double>> RandomForestClassifier::predict() {

  unordered_map<double, int, DoubleHash, DoubleEqual> tmp{};
  int number_test_instances = test_features.size();
  int number_outputs = train_labels[0].size();
  int list_map_size = number_test_instances * number_outputs;
  vector<unordered_map<double, int, DoubleHash, DoubleEqual>> list_freq_map(
      list_map_size, tmp);
  for (auto tree : trees) {
    vector<vector<double>> current_predictions = tree->predict();
    for (int i = 0; i < number_test_instances; i++) {
      for (int j = 0; j < number_outputs; j++) {
        double predicted_class = current_predictions[i][j];
        int index = i * number_outputs + j;
        unordered_map<double, int, DoubleHash, DoubleEqual> current_map =
            list_freq_map.at(index);
        if (current_map.count(predicted_class))
          current_map[predicted_class]++;
        else
          current_map.insert(make_pair(predicted_class, 1));
        list_freq_map[index] = current_map;
      }
    }
  }
  vector<double> tmp2(number_outputs, 0);
  vector<vector<double>> test_predictions(number_test_instances, tmp2);

  for (int index = 0; index < list_map_size; index++) {
    unordered_map<double, int, DoubleHash, DoubleEqual> freq_map =
        list_freq_map[index];
    int max_freq = 0;
    double max_freq_class = -1.0;
    for (auto it = freq_map.begin(); it != freq_map.end(); it++) {
      int current_class = it->first;
      int current_freq = it->second;
      if (current_freq > max_freq) {
        max_freq_class = current_class;
        max_freq = current_freq;
      }
    }
    if (max_freq_class == -1.0)
      logger->log(ERROR,
                  "Maximum frequency class could not be found correctly.");

    int j = index % number_outputs;
    int i = (index - j) / number_outputs;
    test_predictions[i][j] = max_freq_class;
  }

  return test_predictions;
}

void RandomForestClassifier::evaluate() {
  vector<vector<double>> test_predictions = predict();
  get_confusion_matrices(test_predictions, test_labels, train_labels);
}

} // namespace ml
