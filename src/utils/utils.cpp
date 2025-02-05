#include "utils.h"
#include <iterator>

namespace ml
{


	Utilities::Utilities(shared_ptr<Logger> logger):logger(logger) {} ;	

	vector<vector<int>> Utilities::get_confusion_matrix(const vector<vector<int>> &predictions, const vector<vector<int>> &test_outputs, unordered_set<int> unique_classes, int index_output)
	{
		int number_instances = test_outputs.size();
		int number_classes = unique_classes.size();
		vector<int> tmp(number_classes, 0);
		vector<vector<int>> confusion_matrix(number_classes, tmp);
		vector<int> unique_classes_array(unique_classes.begin(), unique_classes.end());
		sort(unique_classes_array.begin(), unique_classes_array.end());
		unordered_map<int, int> class_index{};
		for (int i = 0; i < number_classes; i++)
			class_index.insert(make_pair(unique_classes_array[i], i));

		for (int i = 0; i < number_instances; i++)
		{
			int predicted_class = predictions[i][index_output];
			int ground_truth_class = test_outputs[i][index_output];
			int predicted_index = class_index[predicted_class];
			int ground_truth_index = class_index[ground_truth_class];
			confusion_matrix[ground_truth_index][predicted_index]++;
		}

		return confusion_matrix;
	}

	void Utilities::train_test_split(const vector<vector<double>> &features, const vector<vector<double>> &outputs, vector<vector<double>> &train_features, vector<vector<double>> &train_outputs, vector<vector<double>> &test_features, vector<vector<double>> &test_outputs, double train_ratio, bool shuffle_data)
	{

		if (features.size() != outputs.size())
		{
			logger->log(ERROR, "Features and Outputs datasets have different numbers of records.");
			exit(EXIT_FAILURE);
		}

		int total_instances = features.size();
		int number_train = round(train_ratio * total_instances);
		int number_test = total_instances - number_train;

		if (number_train < 1)
		{
			logger->log(ERROR, "Training data must have at least one instance.");
			exit(EXIT_FAILURE);
		}

		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		vector<int> data_indices{};
		for (int i = 0; i < total_instances; i++)
			data_indices.push_back(i);

		if (shuffle_data)
			std::shuffle(data_indices.begin(), data_indices.end(), std::default_random_engine(seed));

		for (int i = 0; i < number_train; i++)
		{
			train_features.push_back(features[data_indices[i]]);
			train_outputs.push_back(outputs[data_indices[i]]);
		}

		for (int i = 0; i < number_test; i++)
		{
			test_features.push_back(features[data_indices[i + number_train]]);
			test_outputs.push_back(outputs[data_indices[i + number_train]]);
		}
	}

	vector<vector<int>> Utilities::double_to_int(const vector<vector<double>> &data)
	{
		int m = data.size();
		int n = data[0].size();

		vector<vector<int>> data_int{};

		for (int i = 0; i < m; i++)
		{
			vector<int> tmp{};
			for (int j = 0; j < n; j++)
			{
				int int_element = (int)data[i][j];
				tmp.push_back(int_element);
			}
			data_int.push_back(tmp);
		}
		return data_int;
	}

	template <class T>
	string Utilities::array_2d_to_string(vector<vector<T>> matrix)
	{

		string result = "";
		int m = matrix.size();
		int n = matrix[0].size();
		for (int i = 0; i < m; i++)
		{
			string row = "";
			for (int j = 0; j < n; j++)
			{
				row += to_string(matrix[i][j]) + ", ";
			}
			result += row + "\n";
		}
		return result;
	}

	// Explicit instantiation of the template for int
	template string Utilities::array_2d_to_string<int>(vector<vector<int>> matrix);

	// Explicit instantiation of the template for double (if needed)
	template string Utilities::array_2d_to_string<double>(vector<vector<double>> matrix);

	vector<unordered_set<int>> Utilities::get_unique_classes(vector<vector<int>> outputs)
	{

		int number_outputs = outputs[0].size();

		unordered_set<int> tmp{};

		vector<unordered_set<int>> unique_classes(number_outputs, tmp);

		for (int i = 0; i < outputs.size(); i++)
		{
			for (int j = 0; j < outputs[i].size(); j++)
			{
				int current_class = outputs[i][j];
				unique_classes[j].insert(current_class);
			}
		}
		return unique_classes;
	}

	nlohmann::json_abi_v3_11_3::json Utilities::read_json(std::string inputFileName)
	{

		std::ifstream inp;
		inp.open(inputFileName);
		std::stringstream buffer;
		buffer << inp.rdbuf();
		auto inputParameters = nlohmann::json::parse(buffer.str());
		inp.close();
		return inputParameters;
	}

	void Utilities::read_data(std::string data_file,
							  std::vector<std::vector<double>> &features,
							  std::vector<std::vector<double>> &outputs, char delimiter)
	{
		std::ifstream inp;
		inp.open(data_file);
		std::string line, value;
		getline(inp, line);
		std::stringstream ss(line);
		int number_features = 0;
		int number_outputs = 0;
		getline(ss, value, delimiter);
		number_features = std::stoi(value);
		getline(ss, value, delimiter);
		number_outputs = std::stoi(value);
		
		logger->log(INFO, "Number of features = " + to_string(number_features));
		logger->log(INFO, "Number of outputs = " + to_string(number_outputs));

		while (getline(inp, line))
		{
			std::stringstream ssd(line);
			std::vector<double> row_features(number_features, 0.0);
			std::vector<double> row_outputs(number_outputs, 0.0);
			for (int i = 0; i < number_features; i++) {
				getline(ssd, value, delimiter);
				row_features[i] = std::stod(value);
			}
			for (int i = 0; i < number_outputs; i++) {
				getline(ssd, value, delimiter);
				row_outputs[i] = std::stod(value);
			}

			features.push_back(row_features);
			outputs.push_back(row_outputs);
		}

		inp.close();

		logger->log(INFO, "Number of instances = " +to_string(features.size()));
	}


} // namespace ml
