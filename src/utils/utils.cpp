#include "utils.h"

namespace ml {

Utilities::Utilities(shared_ptr<Logger> logger) : logger(logger) {};

void Utilities::train_test_split(const vector<vector<double>> &features,
                                 const vector<vector<double>> &outputs,
                                 TrainTestData &train_test) {

  if (features.size() != outputs.size()) {
    logger->log(
        ERROR,
        "Features and Outputs datasets have different numbers of records.");
    exit(EXIT_FAILURE);
  }

  int total_instances = features.size();
  int number_train = round(train_ratio * total_instances);
  int number_test = total_instances - number_train;

  if (number_train < 1) {
    logger->log(ERROR, "Training data must have at least one instance.");
    exit(EXIT_FAILURE);
  }

  vector<int> data_indices{};
  for (int i = 0; i < total_instances; i++)
    data_indices.push_back(i);

  if (shuffle_data)
    std::shuffle(data_indices.begin(), data_indices.end(),
                 std::default_random_engine(random_seed));

  for (int i = 0; i < number_train; i++) {
    train_test.train_features.push_back(features[data_indices[i]]);
    train_test.train_labels.push_back(outputs[data_indices[i]]);
  }

  for (int i = 0; i < number_test; i++) {
    train_test.test_features.push_back(
        features[data_indices[i + number_train]]);
    train_test.test_labels.push_back(outputs[data_indices[i + number_train]]);
  }
}

void Utilities::one_hot_encoding(vector<vector<double>> &data) {

  double max_category = -1.;

  for (int i = 0; i < data.size(); i++) {
    max_category = max(data[i][0], max_category);
  }
  vector<double> tmp(max_category + 1, 0.);
  vector<vector<double>> one_hot_data(data.size(), tmp);
  for (int i = 0; i < data.size(); i++) {
    int ind = (int)data[i][0];
    one_hot_data[i][ind] = 1;
  }
  swap(data, one_hot_data);
}

TrainTestData Utilities::get_train_test_data(nlohmann::json model_parameters) {

  TrainTestData train_test{};

  if (model_parameters.contains("data")) {
    string data_path = model_parameters["data"];
    vector<vector<double>> features{};
    vector<vector<double>> outputs{};
    string data_format = data_path.substr(data_path.find_last_of(".") + 1);
    if (data_format == "csv")
      read_data_csv(data_path, features, outputs);
    else if (data_format == "bin")
      read_data_bin(data_path, features, outputs);

    if (model_parameters.contains("shuffle_data"))
      shuffle_data = model_parameters["shuffle_data"];
    if (model_parameters.contains("train_ratio"))
      train_ratio = model_parameters["train_ratio"];
    if (model_parameters.contains("random_seed"))
      random_seed = model_parameters["random_seed"];
    train_test_split(features, outputs, train_test);
  } else if (model_parameters.contains("train_data") &&
             model_parameters.contains("test_data")) {
    string train_data_path = model_parameters["train_data"];
    string test_data_path = model_parameters["test_data"];
    string data_format =
        train_data_path.substr(train_data_path.find_last_of(".") + 1);
    if (data_format == "csv") {
      read_data_csv(train_data_path, train_test.train_features,
                    train_test.train_labels);
      read_data_csv(test_data_path, train_test.test_features,
                    train_test.test_labels);
    } else if (data_format == "bin") {
      read_data_bin(train_data_path, train_test.train_features,
                    train_test.train_labels);
      read_data_bin(test_data_path, train_test.test_features,
                    train_test.test_labels);
    }
  }
  if (model_parameters.contains("one_hot_labels")) {
    one_hot_encoding(train_test.train_labels);
    one_hot_encoding(train_test.test_labels);
  }
  return train_test;
}

nlohmann::json_abi_v3_11_3::json Utilities::read_json(std::string input_file) {

  std::ifstream inp;
  inp.open(input_file);
  std::stringstream buffer;
  buffer << inp.rdbuf();
  auto inputParameters = nlohmann::json::parse(buffer.str());
  inp.close();
  return inputParameters;
}

void Utilities::read_data_csv(std::string data_file,
                              std::vector<std::vector<double>> &features,
                              std::vector<std::vector<double>> &outputs,
                              char delimiter) {
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

  while (getline(inp, line)) {
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

  logger->log(INFO, "Number of instances = " + to_string(features.size()));
}

void Utilities::read_data_bin(std::string data_file,
                              std::vector<std::vector<double>> &features,
                              std::vector<std::vector<double>> &outputs,
                              bool distributed) {
  int number_features = features[0].size();
  int number_outputs = outputs[0].size();
  int number_columns = number_features + number_outputs;

  int global_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_File fh;
  MPI_Status status;
  char *read_buffer;
  MPI_Offset offset;
  MPI_Datatype filetype;

  if (distributed) {
    int ierr = MPI_File_open(MPI_COMM_WORLD, data_file.c_str(), MPI_MODE_RDONLY,
                             MPI_INFO_NULL, &fh);
    if (ierr != MPI_SUCCESS) {
      char error_string[MPI_MAX_ERROR_STRING];
      int length_of_error_string;
      MPI_Error_string(ierr, error_string, &length_of_error_string);
      std::cerr << "Rank " << global_rank
                << ": Error opening file: " << error_string << std::endl;
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
    MPI_Offset total_file_size;
    // Only rank 0 needs to get the file size, then broadcast it to all other
    // ranks
    if (global_rank == 0) {
      MPI_File_get_size(fh, &total_file_size);
    }
    // Broadcast the total file size to all ranks
    MPI_Bcast(&total_file_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    const size_t element_size_bytes =
        sizeof(double); // Assuming np.float64 was used in Python
    MPI_Offset bytes_per_rank = total_file_size / world_size;
    MPI_Offset start_offset = global_rank * bytes_per_rank;
    MPI_Offset end_offset = start_offset + bytes_per_rank;
    // Allocate a buffer to hold the segment data
    std::vector<double> buffer(bytes_per_rank / element_size_bytes);

    // Perform the read operation
    ierr = MPI_File_read_at(fh, start_offset, buffer.data(), bytes_per_rank,
                            MPI_BYTE, &status);
    if (ierr != MPI_SUCCESS) {
      char error_string[MPI_MAX_ERROR_STRING];
      int length_of_error_string;
      MPI_Error_string(ierr, error_string, &length_of_error_string);
      std::cerr << "Rank " << global_rank << ": Error reading file at offset "
                << start_offset << ": " << error_string << std::endl;
      MPI_File_close(&fh);
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
    // Close the file
    MPI_File_close(&fh);

    // Copy the data from the buffer to the features and outputs vectors
    int bytes_per_instance = number_columns * element_size_bytes;
    features.resize(bytes_per_rank / bytes_per_instance);
    outputs.resize(bytes_per_rank / bytes_per_instance);
    for (int i = 0; i < bytes_per_rank / element_size_bytes;
         i += number_columns) {
      std::vector<double> row_features(number_features, 0.0);
      std::vector<double> row_outputs(number_outputs, 0.0);
      for (int j = 0; j < number_features; j++) {
        row_features[j] = buffer[i + j];
      }
      for (int j = 0; j < number_outputs; j++) {
        row_outputs[j] = buffer[i + number_features + j];
      }
      features[i / number_columns] = row_features;
      outputs[i / number_columns] = row_outputs;
    }
  }
}

} // namespace ml
