#include <algorithm>
#include <vector>
#include <chrono>
#include <random>
#include "logging.cpp"

using namespace std;

namespace ml {

/** Split the input data for features and outputs into trainigna nd test sets.
 *
 */
void train_test_split(const vector<vector<double>> & features, const vector<vector<double>> & outputs, vector<vector<double>> & train_features, vector<vector<double>> & train_outputs,vector<vector<double>> & test_features,vector<vector<double>> & test_outputs, double train_ratio, bool shuffle_data, Logger * logger) {

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

	if (shuffle_data) std::shuffle(data_indices.begin(), data_indices.end(), std::default_random_engine(seed));

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

}
