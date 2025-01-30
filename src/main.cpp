#include "utils/read_input.cpp"
#include "utils/logging.cpp"
#include "ensemble/decision_tree_classifier.h"
#include "ensemble/random_forest_classifier.h"
#include <string>
#include <random>
using namespace ml;
using namespace std;

/** Split the input data for features and outputs into trainigna nd test sets.
 *
 */
void train_test_split(const vector<vector<double>> & features, const vector<vector<double>> & outputs, vector<vector<double>> & train_features, vector<vector<double>> & train_outputs,vector<vector<double>> & test_features,vector<vector<double>> & test_outputs, double train_ratio, bool shuffle, Logger * logger) {

	if (features.size() != outputs.size()) {
		logger->log(ERROR, "Features and Outputs datasets have different numbers of records.") ;
		exit(EXIT_FAILURE) ;
	}


	int total_instances = features.size();
	int number_train = round(train_ratio*total_instances);
	int number_test = total_instances - number_train ;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	vector<int> data_indices {};
	for (int i = 0; i < total_instances; i++) data_indices.push_back(i);

	if (shuffle) std::shuffle(data_indices.begin(), data_indices.end(), std::default_random_engine(seed));

	for (int i = 0; i < number_train; i++) {
	       	train_features.push_back(features[i]);
		train_outputs.push_back(outputs[i]);
	}

	for (int i = 0; i < number_test;i++) {
		test_features.push_back(features[i+number_train]);
		test_outputs.push_back(outputs[i+number_train]);
	}

}

vector<vector<int>> double_to_int(const vector<vector<double>> & data) {
	int m = data.size();
	int n = data[0].size();

	vector<vector<int>> data_int{};

	 for (int i = 0; i < m; i++) {
		vector<int> tmp {} ;
		for (int j = 0; j < n; j++) {
		    int int_element = (int) data[i][j];
		    tmp.push_back(int_element);
		}
		data_int.push_back(tmp);
	 }
	 return data_int;
}

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
		bool shuffle = true;
		double train_ratio = 0.75;
		if (model_parameters.contains("shuffle_data")) shuffle = model_parameters["shuffle_data"];
		if (model_parameters.contains("train_ratio")) train_ratio = model_parameters["train_ratio"];
		// train-test split
		train_test_split(features, outputs, train_features, train_outputs, test_features, test_outputs, train_ratio, shuffle, &logger);	
		string model_type = model_parameters["model"];
		if (model_type == "decision_tree_classifier") {
			DecisionTreeClassifier decision_tree(model_parameters, &logger);
			logger.log(INFO, "Training " + model_type);
			vector<vector<int>> train_outputs_int = double_to_int(train_outputs) ;
			vector<vector<int>> test_outputs_int = double_to_int(test_outputs) ;
			decision_tree.fit(train_features, train_outputs_int);
		} else if (model_type == "neural_network_regressor") {
			logger.log(INFO, "Neural network is not currently implemented but will be coming soon :).");
		} else {
			logger.log(ERROR, "The specified model type is currently unsupported.");
		}

	}

    // Example usage of the logger
    logger.log(INFO, "Program started.");
    logger.log(DEBUG, "Debugging information.");
    logger.log(ERROR, "An error occurred.");

    return 0;
}
