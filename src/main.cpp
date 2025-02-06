#include "utils/utils.h"
#include "ensemble/decision_tree_classifier.h"
#include "ensemble/random_forest_classifier.h"

using namespace std;
using namespace ml;

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
		string data_path = model_parameters["data"];
		int random_seed = model_parameters["random_seed"];
		vector<vector<double>> features {};
		vector<vector<double>> outputs {};
		utils.read_data(data_path, features, outputs);
	        vector<vector<double>> train_features {};	
	        vector<vector<double>> test_features {};
		vector<vector<double>> train_outputs {};	
		vector<vector<double>> test_outputs {};
		bool shuffle_data = true;
		double train_ratio = 0.75;
		if (model_parameters.contains("shuffle_data")) shuffle_data = model_parameters["shuffle_data"];
		if (model_parameters.contains("train_ratio")) train_ratio = model_parameters["train_ratio"];
		// train-test split
		utils.train_test_split(features, outputs, train_features, train_outputs, test_features, test_outputs, train_ratio, shuffle_data, random_seed);
		// type conversion of outputs for classification algorithms	
		vector<vector<int>> train_outputs_int = utils.double_to_int(train_outputs) ;
		vector<vector<int>> test_outputs_int = utils.double_to_int(test_outputs) ;
		string model_type = model_parameters["model"];
		if (model_type == "decision_tree_classifier") {
			DecisionTreeClassifier decision_tree(model_parameters, logger);
			logger->log(INFO, "Training " + model_type);
			logger->log(INFO,"Data file used was " + data_path);
			decision_tree.fit(train_features, train_outputs_int);
			vector<vector<int>> predictions = decision_tree.predict(train_outputs_int, test_features);
			vector<unordered_set<int>> unique_classes = utils.get_unique_classes(train_outputs_int);
			vector<unordered_set<int>> unique_classes_test = utils.get_unique_classes(test_outputs_int);
			for (int i = 0; i < unique_classes.size(); i++) unique_classes[i].insert(unique_classes_test[i].begin(), unique_classes_test[i].end());
			vector<vector<int>> confusion_matrix = utils.get_confusion_matrix(predictions, test_outputs_int, unique_classes[0], 0);
			logger->log(INFO, "Confusion matrix for decision tree classifier");
			logger->log(INFO, utils.array_2d_to_string(confusion_matrix));
		} else if (model_type == "random_forest_classifier") {
			RandomForestClassifier random_forest(model_parameters, logger);
			logger->log(INFO, "Training " + model_type);
			logger->log(INFO,"Data file used was " + data_path);
			random_forest.fit(train_features, train_outputs_int);
			vector<vector<int>> predictions = random_forest.predict(train_outputs_int, test_features);
			vector<unordered_set<int>> unique_classes = utils.get_unique_classes(train_outputs_int);
			vector<unordered_set<int>> unique_classes_test = utils.get_unique_classes(test_outputs_int);
			for (int i = 0; i < unique_classes.size(); i++) unique_classes[i].insert(unique_classes_test[i].begin(), unique_classes_test[i].end());
			vector<vector<int>> confusion_matrix = utils.get_confusion_matrix(predictions, test_outputs_int, unique_classes[0], 0);
			logger->log(INFO, "Confusion matrix for random forest classifier");
			logger->log(INFO, utils.array_2d_to_string(confusion_matrix));
		} else if (model_type == "neural_network_regressor") {
			logger->log(INFO, "Neural network is not currently implemented but will be coming soon :).");
		} else {
			logger->log(ERROR, "The specified model type, " + model_type + ", is currently unsupported.");
		}

	}


    return 0;
}
