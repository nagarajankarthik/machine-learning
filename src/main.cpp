#include "utils/read_input.cpp"
#include "utils/misc.cpp"
#include "ensemble/decision_tree_classifier.h"
#include "ensemble/random_forest_classifier.h"
#include <string>

using namespace std;
using namespace ml;

int main(int argc, char *argv[])
{
    string inputFileName = "input.json";
    if (argc > 1)
        inputFileName = argv[1];
    ReadInput inputReader;
    auto inputParameters = inputReader.readJson(inputFileName);

    string logFileName = inputParameters["general"]["logfile"];


    // Create logger instance
    Logger logger("../logs/" + logFileName);

    nlohmann::json modelSpecifications = inputParameters["models"];

    // iterate the array
	for (nlohmann::json::iterator it = modelSpecifications.begin(); it != modelSpecifications.end(); ++it) {
		nlohmann::json model_parameters = *it;
		string data_path = model_parameters["data"];
		vector<vector<double>> features {};
		vector<vector<double>> outputs {};
		inputReader.readData(data_path, features, outputs);
	        vector<vector<double>> train_features {};	
	        vector<vector<double>> test_features {};
		vector<vector<double>> train_outputs {};	
		vector<vector<double>> test_outputs {};
		bool shuffle_data = true;
		double train_ratio = 0.75;
		if (model_parameters.contains("shuffle_data")) shuffle_data = model_parameters["shuffle_data"];
		if (model_parameters.contains("train_ratio")) train_ratio = model_parameters["train_ratio"];
		// train-test split
		train_test_split(features, outputs, train_features, train_outputs, test_features, test_outputs, train_ratio, shuffle_data, &logger);
		// type conversion of outputs for classification algorithms	
		vector<vector<int>> train_outputs_int = double_to_int(train_outputs) ;
		vector<vector<int>> test_outputs_int = double_to_int(test_outputs) ;
		string model_type = model_parameters["model"];
		if (model_type == "decision_tree_classifier") {
			DecisionTreeClassifier decision_tree(model_parameters, &logger);
			logger.log(INFO, "Training " + model_type);
			decision_tree.fit(train_features, train_outputs_int);
			vector<vector<int>> predictions = decision_tree.predict(test_features);
			vector<vector<int>> confusion_matrix = get_confusion_matrix(predictions, test_outputs, 0);
		} else if (model_type == "neural_network_regressor") {
			logger.log(INFO, "Neural network is not currently implemented but will be coming soon :).");
		} else {
			logger.log(ERROR, "The specified model type is currently unsupported.");
		}

	}


    return 0;
}
