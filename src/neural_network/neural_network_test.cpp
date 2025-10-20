#include "neural_network.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace std;
using namespace ml;

class NeuralNetworkTest : public testing::Test {
protected:
  shared_ptr<Logger> logger;
  
  NeuralNetworkTest() {
    logger = make_shared<Logger>("test_neural_network.log");
  }
};

TEST_F(NeuralNetworkTest, ProfilingTest) {
  // Create a simple neural network configuration
  nlohmann::json parameters;
  parameters["batch_size"] = 2;
  parameters["number_epochs"] = 2;
  parameters["input_shape"] = {2, 1, 2};  // batch_size, 1, features
  parameters["labels_shape"] = {2, 1, 2};  // batch_size, 1, classes
  
  // Define layers
  nlohmann::json layers = nlohmann::json::array();
  nlohmann::json layer1;
  layer1["type"] = "fully_connected";
  layer1["number_inputs"] = 2;
  layer1["number_outputs"] = 2;
  layer1["activation"] = "softmax";
  layer1["init_method"] = "pytorch";
  layers.push_back(layer1);
  parameters["layers"] = layers;
  
  // Define optimizer
  parameters["optimizer"]["type"] = "sgd";
  parameters["optimizer"]["learning_rate"] = 0.01;
  parameters["optimizer"]["momentum"] = 0.9;
  
  parameters["loss"] = "cross_entropy";
  parameters["random_seed"] = 42;
  
  // Create neural network
  NeuralNetwork nn(parameters, logger);
  
  // Create simple training data
  vector<vector<double>> train_features = {{1.0, 0.0}, {0.0, 1.0}};
  vector<vector<double>> train_labels = {{1.0, 0.0}, {0.0, 1.0}};
  vector<vector<double>> test_features = {{1.0, 0.0}, {0.0, 1.0}};
  vector<vector<double>> test_labels = {{1.0, 0.0}, {0.0, 1.0}};
  
  nn.prepare_train_input(train_features, train_labels);
  nn.prepare_inference_input(test_features, test_labels);
  
  // Train the network
  nn.fit();
  
  // Check that profiling statistics were collected
  EXPECT_GT(nn.forward_pass_count, 0);
  EXPECT_GT(nn.backward_pass_count, 0);
  EXPECT_GT(nn.total_forward_time_ms, 0.0);
  EXPECT_GT(nn.total_backward_time_ms, 0.0);
  
  // Check that the number of forward and backward passes match
  EXPECT_EQ(nn.forward_pass_count, nn.backward_pass_count);
  
  // Check that we have the expected number of passes (batches * epochs)
  int expected_passes = 1 * 2;  // 1 batch * 2 epochs
  EXPECT_EQ(nn.forward_pass_count, expected_passes);
  
  logger->log(INFO, "Test passed - profiling statistics collected successfully");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
