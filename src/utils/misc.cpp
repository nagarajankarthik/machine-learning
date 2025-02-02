#include <algorithm>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <unordered_set>
#include "logging.cpp"

using namespace std;

namespace ml {

/** Split the input data for features and outputs into training and test sets.
 *
 */
void train_test_split(const vector<vector<double>> & features, const vector<vector<double>> & outputs, vector<vector<double>> & train_features, vector<vector<double>> & train_outputs,vector<vector<double>> & test_features,vector<vector<double>> & test_outputs, double train_ratio, bool shuffle_data, shared_ptr<Logger> logger) {

	if (features.size() != outputs.size()) {
		logger->log(ERROR, "Features and Outputs datasets have different numbers of records.") ;
		exit(EXIT_FAILURE) ;
	}


	int total_instances = features.size();
	int number_train = round(train_ratio*total_instances);
	int number_test = total_instances - number_train ;

	if (number_train < 1) {
		logger->log(ERROR, "Training data must have at least one instance.");
		exit(EXIT_FAILURE);
	}


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

/**
 * Convert input 2D array into a single string.
 */
template <class T>
string array_2d_to_string(vector<vector<T>> matrix) {

	string result("");		
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

vector<unordered_set<int>> get_unique_classes(vector<vector<int>> outputs) {

	int number_outputs = outputs[0].size();

	unordered_set<int> tmp {};

	vector<unordered_set<int>> unique_classes(number_outputs, tmp);

	for (int i = 0; i < outputs.size(); i++) {
		for (int j = 0; j < outputs[i].size(); j++) {
			int current_class = outputs[i][j];
			unique_classes[j].insert(current_class);
		}
	}
	return unique_classes;
}
				


}
