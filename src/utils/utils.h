#include <algorithm>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <unordered_set>
#include "logging.h"
#include "json.hpp"

using namespace std;

namespace ml
{

	class Utilities
	{
	public:
		/**
		 * Constructor for utilities class
		 */
		Utilities() {};
		/**
		 * Destructor for utilities class
		 */
		~Utilities() {};

		/** Split the input data for features and outputs into training and test sets.
		 */
		void train_test_split(const vector<vector<double>> &features, const vector<vector<double>> &outputs, vector<vector<double>> &train_features, vector<vector<double>> &train_outputs, vector<vector<double>> &test_features, vector<vector<double>> &test_outputs, double train_ratio, bool shuffle_data, shared_ptr<Logger> logger);

		/**
		 * Convert a 2D array containing elements of type double to a 2D array containing elements of type int.
		 */
		vector<vector<int>> double_to_int(const vector<vector<double>> &data);

		/**
		 * Convert input 2D array into a single string.
		 */
		template <class T>
		string array_2d_to_string(vector<vector<T>> matrix);

		/**
		 * Get unique entries in each column of an input 2D vector containing integers.
		 */
		vector<unordered_set<int>> get_unique_classes(vector<vector<int>> outputs);

		/** Get confusion matrix using predictions of classification algorithm and ground truth
		 */
		vector<vector<int>> get_confusion_matrix(const vector<vector<int>> &predictions, const vector<vector<int>> &test_outputs, unordered_set<int> unique_classes, int index_output);

		/**
		 *Read input parameters from a file in JSON format.
		 */
		nlohmann::json_abi_v3_11_3::json read_json(std::string inputFileName);

		/**
		 * Read data for features and outputs from a user-specified data file
		 * The first line of the file must being with two integers specifying
		 * the number of columns containing features and outputs respectively.
		 *
		 * @param dataFileName: Absolute path to data file.
		 * @param features: Array to store data for features.
		 * @param outputs: Array to store data for outputs.
		 */
		void read_data(std::string dataFileName,
					   std::vector<std::vector<double>> &features,
					   std::vector<std::vector<double>> &outputs);
	};
} // namespace ml
