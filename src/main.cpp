#include "utils/utils.h"
#include "ensemble/random_forest_classifier.h"
#include <memory>

using namespace std;
using namespace ml;

std::unique_ptr<BaseModel> create_model(nlohmann::json model_parameters,
                                        shared_ptr<Logger> logger) {
	string model= model_parameters["model"];
    if (model == "decision_tree_classifier") {
        return std::make_unique<DecisionTreeClassifier>(model_parameters, logger);
    } else if (model == "random_forest_classifier") {
        return std::make_unique<RandomForestClassifier>(model_parameters, logger);
    } else {
        return nullptr;
    }
}

void run_model(Utilities & utils, nlohmann::json model_parameters, shared_ptr<Logger> logger) {
	std::unique_ptr<BaseModel> model = create_model(model_parameters, logger);
	string model_type = model_parameters["model"];
	if (model == nullptr) {
		logger->log(ERROR, "The specified model type, " + model_type + " is currently unsupported.");
		return;
	}
	TrainTestData train_test = utils.get_train_test_data(model_parameters); 
	string data_path = model_parameters["data"];
	logger->log(INFO, "Training " + model_type);
	logger->log(INFO,"Data file used was " + data_path);
	model->fit(std::move(train_test).train_features, std::move(train_test).train_labels);
	logger->log(INFO, "Evaluating " + model_type + " on test data");
	model->evaluate(std::move(train_test).test_features, std::move(train_test).test_labels);
}


int main(int argc, char *argv[])
{
    string input_file = "input.json";
    if (argc > 1)
        input_file = argv[1];


    Utilities utils;
    auto input_parameters = utils.read_json(input_file);

    string log_file = input_parameters["general"]["logfile"];


    // Create logger instance
    shared_ptr<Logger> logger = make_shared<Logger>("../logs/" + log_file);
    logger->log(DEBUG,"Reading input parameters from " + input_file);

    utils.logger = logger;

    nlohmann::json model_specifications = input_parameters["models"];

	for (nlohmann::json::iterator it = model_specifications.begin(); it != model_specifications.end(); ++it) {
		nlohmann::json model_parameters = *it;
		run_model(utils, model_parameters, logger);
	}


    return 0;
}
