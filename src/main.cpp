#include "ensemble/random_forest_classifier.h"
#include "neural_network/neural_network.h"
#include "utils/utils.h"
#include <cstdlib>
#include <memory>
#include <mpi.h>

using namespace std;
using namespace ml;

std::unique_ptr<BaseModel> create_model(nlohmann::json model_parameters,
                                        shared_ptr<Logger> logger) {
  string model = model_parameters["model"];
  if (model == "decision_tree_classifier") {
    return std::make_unique<DecisionTreeClassifier>(model_parameters, logger);
  } else if (model == "random_forest_classifier") {
    return std::make_unique<RandomForestClassifier>(model_parameters, logger);
  } else if (model == "neural_network") {
    return std::make_unique<NeuralNetwork>(model_parameters, logger);
  } else {
    return nullptr;
  }
}

void run_model(Utilities &utils, nlohmann::json model_parameters,
               shared_ptr<Logger> logger) {
  std::unique_ptr<BaseModel> model = create_model(model_parameters, logger);
  string model_type = model_parameters["model"];
  if (model == nullptr) {
    logger->log(ERROR, "The specified model type, " + model_type +
                           " is currently unsupported.");
    return;
  }
  TrainTestData train_test = utils.get_train_test_data(model_parameters);
  if (model_parameters.contains("data")) {
    string data_path = model_parameters["data"];
    logger->log(INFO, "Data file used was " + data_path);
  } else {
    string train_data_path = model_parameters["train_data"];
    string test_data_path = model_parameters["test_data"];
    logger->log(INFO, "Train data file used was " + train_data_path);
    logger->log(INFO, "Test data file used was " + test_data_path);
  }
  model->set_data(std::move(train_test));
  logger->log(INFO, "Training " + model_type);
  model->fit();
  logger->log(INFO, "Evaluating " + model_type + " on test data");
  model->evaluate();
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  string input_file = "input.json";
  if (argc > 1)
    input_file = argv[1];

  Utilities utils;
  auto input_parameters = utils.read_json(input_file);

  string log_file = input_parameters["general"]["logfile"];

  // Create logger instance
  shared_ptr<Logger> logger = make_shared<Logger>("../logs/" + log_file);
  logger->log(DEBUG, "Reading input parameters from " + input_file);

  utils.logger = logger;

  nlohmann::json model_specifications = input_parameters["models"];

  for (nlohmann::json::iterator it = model_specifications.begin();
       it != model_specifications.end(); ++it) {
    nlohmann::json model_parameters = *it;
    run_model(utils, model_parameters, logger);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
