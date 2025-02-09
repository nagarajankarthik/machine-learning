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

	struct TrainTestData {
	
		/**
		 * Features of training data
		 */
		vector<vector<double>> train_features {} ;

		/**
		 * Labels of training data
		 */
		vector<vector<double>> train_labels {};

		/**
		 * Features of test data
		 */
		vector<vector<double>> test_features {} ;

		/**
		 * Labels of training data
		 */
		vector<vector<double>> test_labels {};
	};

	class Utilities
	{
	public:

		/**
		 * Proportion of input data to be used for training
		 */
		double train_ratio = 0.75;

		/**
		 * Whether the data should be randomly shuffled before splitting into train and test sets.
		 */
		bool shuffle_data = true;

		/**
		 * random seed for shuffling data
		 */
		int random_seed = 1;

		/**
		 * Default constructor for utilities
		 */
		Utilities() {};

		/**
		 * Constructor for utilities class
		 */
		Utilities(shared_ptr<Logger> logger) ;
		/**
		 * Destructor for utilities class
		 */
		~Utilities() {};

		/**
		 * Pointer to a Logger instance
		 */
		shared_ptr<Logger> logger;

		/** Split the input data for features and outputs into training and test sets.
		 */
		TrainTestData train_test_split(const vector<vector<double>> &features, const vector<vector<double>> &outputs);

		/**
		 * Read data from input file and split it into training and test sets.
		 */
		TrainTestData get_train_test_data(nlohmann::json model_parameters);


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
		void read_data(std::string data_file,
					   std::vector<std::vector<double>> &features,
					   std::vector<std::vector<double>> &outputs, char delimiter = ',');
	};
} // namespace ml
