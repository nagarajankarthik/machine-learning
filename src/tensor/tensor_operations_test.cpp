#include "tensor.h"
#include "tensor_operations.h"
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
using namespace std;
using namespace ml;

class TensorOpsTest : public testing::Test {
protected:
  shared_ptr<Logger> logger;
  shared_ptr<Tensor> a;
  shared_ptr<Tensor> b;
  double tol = 1.0e-12;

  TensorOpsTest() {

    logger = make_shared<Logger>("test_tensor.log");

    vector<int> shape = {1, 2, 2};
    vector<double> values{1., 1., 1., 1.};
    a = make_shared<Tensor>(values, shape, logger);
    logger->log(INFO, "Initialized tensor a");
    shape[0] = 2;
    values.insert(values.end(), {1., 1., 1., 1.});
    b = make_shared<Tensor>(values, shape, logger);
  }
};
TEST_F(TensorOpsTest, AddForwardTest) {
  shared_ptr<Tensor> c = add_batch_forward(a, b);
  logger->log(INFO, "Created tensor c by adding a and b.");
  vector<int> position{0};
  vector<vector<double>> matrix = c->get_matrix(position, "values");
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[0].size(); j++) {
      ASSERT_FLOAT_EQ(matrix[i][j], 2.);
    }
  }
}

TEST_F(TensorOpsTest, AddBackwardTest) {
  shared_ptr<Tensor> c = add_batch_forward(a, b);
  logger->log(INFO, "Created tensor c by adding a and b.");
  fill(c->gradients.begin(), c->gradients.end(), 0.);
  fill(c->gradients.begin() + 4, c->gradients.end(), 1.);
  c->backward();

  vector<int> position{0};
  vector<vector<double>> grad = a->get_matrix(position, "gradients");
  for (int i = 0; i < grad.size(); i++) {
    for (int j = 0; j < grad[0].size(); j++) {
      ASSERT_FLOAT_EQ(grad[i][j], 1.);
    }
  }

  grad = b->get_matrix(position, "gradients");
  for (int i = 0; i < grad.size(); i++) {
    for (int j = 0; j < grad[0].size(); j++) {
      ASSERT_FLOAT_EQ(grad[i][j], 0.);
    }
  }

  position[0] = 1;

  grad = b->get_matrix(position, "gradients");
  for (int i = 0; i < grad.size(); i++) {
    for (int j = 0; j < grad[0].size(); j++) {
      ASSERT_FLOAT_EQ(grad[i][j], 1.);
    }
  }
}

TEST_F(TensorOpsTest, MatmulForwardTest) {
  shared_ptr<Tensor> c = batch_matmul_forward(a, b);
  logger->log(INFO, "Created tensor c by multiplying a and b.");
  vector<int> position{0};
  vector<vector<double>> matrix = c->get_matrix(position, "values");
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[0].size(); j++) {
      ASSERT_TRUE(fabs(matrix[i][j] - 2.) < tol);
    }
  }
}

TEST_F(TensorOpsTest, MatmulBackwardTest) {
  shared_ptr<Tensor> c = batch_matmul_forward(a, b);
  logger->log(INFO, "Created tensor c by multiplying a and b.");
  vector<int> position{0};
  fill(c->gradients.begin(), c->gradients.end(), 1.);
  c->backward();

  vector<vector<double>> grad = a->get_matrix(position, "gradients");
  for (int i = 0; i < grad.size(); i++) {
    for (int j = 0; j < grad[0].size(); j++) {
      ASSERT_FLOAT_EQ(grad[i][j], 4.);
    }
  }

  grad = b->get_matrix(position, "gradients");
  for (int i = 0; i < grad.size(); i++) {
    for (int j = 0; j < grad[0].size(); j++) {
      ASSERT_FLOAT_EQ(grad[i][j], 2.);
    }
  }
}

// Perform concatenation on batch dimension
TEST_F(TensorOpsTest, ConcatenateBatchForwardTest) {
  shared_ptr<Tensor> c = concatenate_forward(a, b, 0);
  logger->log(INFO, "Created tensor c by concatenating a and b.");
  ASSERT_EQ(c->values.size(), 12);
  vector<int> c_shape = c->shape;
  ASSERT_EQ(c_shape.size(), 3);
  ASSERT_EQ(c_shape[0], 3);
  ASSERT_EQ(c_shape[1], 2);
  ASSERT_EQ(c_shape[2], 2);
  vector<int> position{0};
  vector<vector<double>> matrix = c->get_matrix(position, "values");
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[0].size(); j++) {
      ASSERT_FLOAT_EQ(matrix[i][j], 1.);
    }
  }
}

TEST_F(TensorOpsTest, ConcatenateBatchBackwardTest) {
  shared_ptr<Tensor> c = concatenate_forward(a, b);
  logger->log(
      INFO, "Created tensor c by concatenating a and b along batch dimension.");
  fill(c->gradients.begin(), c->gradients.end(), 1.);
  c->backward();

  vector<int> position{0};
  vector<vector<double>> grad = a->get_matrix(position, "gradients");
  for (int i = 0; i < grad.size(); i++) {
    for (int j = 0; j < grad[0].size(); j++) {
      ASSERT_FLOAT_EQ(grad[i][j], 1.);
    }
  }

  grad = b->get_matrix(position, "gradients");
  for (int i = 0; i < grad.size(); i++) {
    for (int j = 0; j < grad[0].size(); j++) {
      ASSERT_FLOAT_EQ(grad[i][j], 1.);
    }
  }
}

// Perform concatenation along rows of matrix
TEST_F(TensorOpsTest, ConcatenateRowForwardTest) {

  shared_ptr<Tensor> c = concatenate_forward(a, a, 1);

  logger->log(INFO, "Created tensor c by concatenating a with itself.");
  ASSERT_EQ(c->values.size(), 8);
  vector<int> c_shape = c->shape;
  ASSERT_EQ(c_shape.size(), 3);
  ASSERT_EQ(c_shape[0], a->shape[0]);
  ASSERT_EQ(c_shape[1], 2 * a->shape[1]);
  ASSERT_EQ(c_shape[2], a->shape[2]);
  vector<int> position{0};
  vector<vector<double>> matrix = c->get_matrix(position, "values");
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[0].size(); j++) {
      ASSERT_FLOAT_EQ(matrix[i][j], 1.);
    }
  }
}

TEST_F(TensorOpsTest, ConcatenateRowBackwardTest) {

  shared_ptr<Tensor> c = concatenate_forward(a, a, 1);

  logger->log(INFO, "Created tensor c by concatenating a with itself.");

  vector<int> position{0};
  fill(c->gradients.begin(), c->gradients.end(), 1.);
  c->backward();

  vector<vector<double>> grad = a->get_matrix(position, "gradients");
  for (int i = 0; i < grad.size(); i++) {
    for (int j = 0; j < grad[0].size(); j++) {
      ASSERT_FLOAT_EQ(grad[i][j], 2.);
    }
  }
}

// Perform concatenation along columns of matrix
TEST_F(TensorOpsTest, ConcatenateColumnForwardTest) {

  shared_ptr<Tensor> c = concatenate_forward(a, a, 2);

  logger->log(INFO, "Created tensor c by concatenating a with itself.");
  ASSERT_EQ(c->values.size(), 8);
  vector<int> c_shape = c->shape;
  ASSERT_EQ(c_shape.size(), 3);
  ASSERT_EQ(c_shape[0], a->shape[0]);
  ASSERT_EQ(c_shape[1], a->shape[1]);
  ASSERT_EQ(c_shape[2], 2 * a->shape[2]);
  vector<int> position{0};
  vector<vector<double>> matrix = c->get_matrix(position, "values");
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[0].size(); j++) {
      ASSERT_FLOAT_EQ(matrix[i][j], 1.);
    }
  }
}

TEST_F(TensorOpsTest, ConcatenateColumnBackwardTest) {

  shared_ptr<Tensor> c = concatenate_forward(a, a, 2);

  logger->log(INFO, "Created tensor c by concatenating a with itself.");

  vector<int> position{0};
  fill(c->gradients.begin(), c->gradients.end(), 1.);
  c->backward();

  vector<vector<double>> grad = a->get_matrix(position, "gradients");
  for (int i = 0; i < grad.size(); i++) {
    for (int j = 0; j < grad[0].size(); j++) {
      ASSERT_FLOAT_EQ(grad[i][j], 2.);
    }
  }
}

// Testing activation functions

TEST_F(TensorOpsTest, ReluForwardTest) {

  a->values = {-1., -1., 1., 1.};
  shared_ptr<Tensor> c = relu_forward(a);
  logger->log(INFO, "Created tensor c by applying relu to a.");
  ASSERT_EQ(c->values.size(), 4);
  vector<int> c_shape = c->shape;
  ASSERT_EQ(c_shape.size(), 3);
  ASSERT_EQ(c_shape[0], a->shape[0]);
  ASSERT_EQ(c_shape[1], a->shape[1]);
  ASSERT_EQ(c_shape[2], a->shape[2]);
  vector<int> position{0};
  vector<vector<double>> matrix = c->get_matrix(position, "values");
  for (int j = 0; j < matrix[0].size(); j++) {
    ASSERT_FLOAT_EQ(matrix[0][j], 0.);
  }
  for (int j = 0; j < matrix[0].size(); j++) {
    ASSERT_FLOAT_EQ(matrix[1][j], 1.);
  }
}

TEST_F(TensorOpsTest, ReluBackwardTest) {

  a->values = {-1., -1., 1., 1.};
  shared_ptr<Tensor> c = relu_forward(a);
  logger->log(INFO, "Created tensor c by applying relu to a.");
  vector<int> position{0};
  fill(c->gradients.begin(), c->gradients.end(), 1.);
  c->backward();
  vector<vector<double>> grad = a->get_matrix(position, "gradients");
  for (int j = 0; j < grad[0].size(); j++) {
    ASSERT_FLOAT_EQ(grad[0][j], 0.);
  }
  for (int j = 0; j < grad[0].size(); j++) {
    ASSERT_FLOAT_EQ(grad[1][j], 1.);
  }
}

TEST_F(TensorOpsTest, SigmoidForwardTest) {

  a->values = {-1., -1., 1., 1.};
  shared_ptr<Tensor> c = sigmoid_forward(a);
  logger->log(INFO, "Created tensor c by applying sigmoid function to a.");
  ASSERT_EQ(c->values.size(), 4);
  vector<int> c_shape = c->shape;
  ASSERT_EQ(c_shape.size(), 3);
  ASSERT_EQ(c_shape[0], a->shape[0]);
  ASSERT_EQ(c_shape[1], a->shape[1]);
  ASSERT_EQ(c_shape[2], a->shape[2]);
  vector<int> position{0};
  vector<vector<double>> matrix = c->get_matrix(position, "values");
  for (int j = 0; j < matrix[0].size(); j++) {
    ASSERT_FLOAT_EQ(matrix[0][j], 0.26894143);
  }
  for (int j = 0; j < matrix[0].size(); j++) {
    ASSERT_FLOAT_EQ(matrix[1][j], 0.7310586);
  }
}

TEST_F(TensorOpsTest, SigmoidBackwardTest) {

  a->values = {-1., -1., 1., 1.};
  shared_ptr<Tensor> c = sigmoid_forward(a);
  logger->log(INFO, "Created tensor c by applying sigmoid to a.");
  fill(c->gradients.begin(), c->gradients.end(), 1.);
  c->backward();

  for (int i = 0; i < c->values.size(); i++) {
    double val = c->values[i];
    ASSERT_FLOAT_EQ(a->gradients[i], val * (1. - val));
  }
}

TEST_F(TensorOpsTest, TanhForwardTest) {

  a->values = {-1., -1., 1., 1.};
  shared_ptr<Tensor> c = tanh_forward(a);
  logger->log(INFO, "Created tensor c by applying tanh function to a.");
  ASSERT_EQ(c->values.size(), 4);
  vector<int> c_shape = c->shape;
  ASSERT_EQ(c_shape.size(), 3);
  ASSERT_EQ(c_shape[0], a->shape[0]);
  ASSERT_EQ(c_shape[1], a->shape[1]);
  ASSERT_EQ(c_shape[2], a->shape[2]);
  vector<int> position{0};
  vector<vector<double>> matrix = c->get_matrix(position, "values");
  for (int j = 0; j < matrix[0].size(); j++) {
    ASSERT_FLOAT_EQ(matrix[0][j], -0.7615942);
  }
  for (int j = 0; j < matrix[0].size(); j++) {
    ASSERT_FLOAT_EQ(matrix[1][j], 0.7615942);
  }
}

TEST_F(TensorOpsTest, TanhBackwardTest) {

  a->values = {-1., -1., 1., 1.};
  shared_ptr<Tensor> c = tanh_forward(a);
  logger->log(INFO, "Created tensor c by applying tanh to a.");
  fill(c->gradients.begin(), c->gradients.end(), 1.);
  c->backward();

  for (int i = 0; i < c->values.size(); i++) {
    double val = c->values[i];
    ASSERT_FLOAT_EQ(a->gradients[i], 1. - val * val);
  }
}

TEST_F(TensorOpsTest, SoftmaxForwardTest) {

  a->values = {-1., -1., 1., 1.};
  shared_ptr<Tensor> c = softmax_forward(a);
  logger->log(INFO, "Created tensor c by applying softmax function to a.");
  ASSERT_EQ(c->values.size(), 4);
  vector<int> c_shape = c->shape;
  ASSERT_EQ(c_shape.size(), 3);
  ASSERT_EQ(c_shape[0], a->shape[0]);
  ASSERT_EQ(c_shape[1], a->shape[1]);
  ASSERT_EQ(c_shape[2], a->shape[2]);
  vector<int> position{0};
  vector<vector<double>> matrix = c->get_matrix(position, "values");
  double e2 = exp(2.);
  for (int j = 0; j < matrix[0].size(); j++) {
    ASSERT_FLOAT_EQ(matrix[0][j], 1. / (1. + e2));
  }
  for (int j = 0; j < matrix[0].size(); j++) {
    ASSERT_FLOAT_EQ(matrix[1][j], e2 / (1. + e2));
  }
}

TEST_F(TensorOpsTest, SoftmaxBackwardTest) {

  a->values = {-1., -1., 1., 1.};
  shared_ptr<Tensor> c = softmax_forward(a);
  logger->log(INFO, "Created tensor c by applying softmax to a.");
  vector<int> position{0};
  fill(c->gradients.begin(), c->gradients.end(), 1.);
  c->backward();

  vector<vector<double>> softmax_values = c->get_matrix(position, "values");
  vector<vector<double>> softmax_derivatives =
      c->get_matrix(position, "gradients");
  vector<vector<double>> calculated_derivatives =
      a->get_matrix(position, "gradients");
  for (int j = 0; j < calculated_derivatives[0].size(); j++) {
    for (int i = 0; i < calculated_derivatives.size(); i++) {
      double expected_derivative =
          softmax_values[i][j] *
          (softmax_derivatives[i][j] -
           matrix_col_sum(elementwise_multiplication(
               softmax_derivatives, softmax_values, logger))[j]);
      ASSERT_FLOAT_EQ(calculated_derivatives[i][j], expected_derivative);
    }
  }
}

TEST_F(TensorOpsTest, CrossEntropyForwardTest) {
  vector<int> test_shape{1, 2, 2};
  vector<double> predicted_values{0.8, 0.1, 0.2, 0.9};
  vector<double> ground_truth_values{1., 0., 0., 1.};
  shared_ptr<Tensor> predicted =
      make_shared<Tensor>(predicted_values, test_shape, logger);
  shared_ptr<Tensor> ground_truth =
      make_shared<Tensor>(ground_truth_values, test_shape, logger);
  shared_ptr<Tensor> loss =
      categorical_cross_entropy_forward(predicted, ground_truth);
  logger->log(INFO, "Calculated cross entropy loss based on ground truth "
                    "and predicted values.");
  ASSERT_EQ(loss->values.size(), 2);
  vector<int> loss_shape = loss->shape;
  ASSERT_EQ(loss_shape.size(), 3);
  ASSERT_EQ(loss_shape[0], predicted->shape[0]);
  ASSERT_EQ(loss_shape[1], 1);
  ASSERT_EQ(loss_shape[2], predicted->shape[2]);
  vector<int> position{0};
  vector<vector<double>> loss_values = loss->get_matrix(position, "values");
  ASSERT_FLOAT_EQ(loss_values[0][0], -log(0.8));
  ASSERT_FLOAT_EQ(loss_values[0][1], -log(0.9));
}

TEST_F(TensorOpsTest, CrossEntropyBackwardTest) {

  vector<int> test_shape{1, 2, 2};
  vector<double> predicted_values{0.8, 0.1, 0.2, 0.9};
  vector<double> ground_truth_values{1., 0., 0., 1.};
  shared_ptr<Tensor> predicted =
      make_shared<Tensor>(predicted_values, test_shape, logger);
  shared_ptr<Tensor> ground_truth =
      make_shared<Tensor>(ground_truth_values, test_shape, logger);
  shared_ptr<Tensor> loss =
      categorical_cross_entropy_forward(predicted, ground_truth);
  logger->log(INFO, "Calculated cross entropy loss based on ground truth "
                    "and predicted values.");
  fill(loss->gradients.begin(), loss->gradients.end(), 1.);
  loss->backward();

  vector<double> expected_derivatives{
      -ground_truth_values[0] / predicted_values[0],
      -ground_truth_values[1] / predicted_values[1],
      -ground_truth_values[2] / predicted_values[2],
      -ground_truth_values[3] / predicted_values[3],
  };

  for (int i = 0; i < predicted->gradients.size(); i++) {
    double pred_grad = predicted->gradients[i];
    double expected_grad = expected_derivatives[i];
    ASSERT_FLOAT_EQ(pred_grad, expected_grad);
  }
}

TEST_F(TensorOpsTest, MeanSquareErrorForwardTest) {
  vector<int> test_shape{1, 2, 2};
  vector<double> predicted_values{0.9, 1.1, 0.2, 0.9};
  vector<double> ground_truth_values{0.8, 1.5, 0.1, 0.8};
  shared_ptr<Tensor> predicted =
      make_shared<Tensor>(predicted_values, test_shape, logger);
  shared_ptr<Tensor> ground_truth =
      make_shared<Tensor>(ground_truth_values, test_shape, logger);
  shared_ptr<Tensor> loss = mean_squared_error_forward(predicted, ground_truth);
  logger->log(INFO, "Calculated mean squared error loss based on ground "
                    "truth and predicted values.");
  ASSERT_EQ(loss->values.size(), 4);
  vector<int> loss_shape = loss->shape;
  ASSERT_EQ(loss_shape.size(), 3);
  ASSERT_EQ(loss_shape[0], predicted->shape[0]);
  ASSERT_EQ(loss_shape[1], predicted->shape[1]);
  ASSERT_EQ(loss_shape[2], predicted->shape[2]);
  for (int i = 0; i < loss->values.size(); i++) {
    double expected_loss = (predicted->values[i] - ground_truth->values[i]) *
                           (predicted->values[i] - ground_truth->values[i]);
    ASSERT_FLOAT_EQ(loss->values[i], expected_loss);
  }
}

TEST_F(TensorOpsTest, MeanSquareErrorBackwardTest) {

  vector<int> test_shape{1, 2, 2};
  vector<double> predicted_values{0.9, 1.1, 0.2, 0.9};
  vector<double> ground_truth_values{0.8, 1.5, 0.1, 0.8};
  shared_ptr<Tensor> predicted =
      make_shared<Tensor>(predicted_values, test_shape, logger);
  shared_ptr<Tensor> ground_truth =
      make_shared<Tensor>(ground_truth_values, test_shape, logger);
  shared_ptr<Tensor> loss = mean_squared_error_forward(predicted, ground_truth);
  logger->log(INFO, "Calculated mean squared error loss based on ground truth "
                    "and predicted values.");
  fill(loss->gradients.begin(), loss->gradients.end(), 1.);
  loss->backward();

  vector<double> expected_derivatives{
      2.0 * (predicted_values[0] - ground_truth_values[0]),
      2.0 * (predicted_values[1] - ground_truth_values[1]),
      2.0 * (predicted_values[2] - ground_truth_values[2]),
      2.0 * (predicted_values[3] - ground_truth_values[3]),
  };

  for (int i = 0; i < predicted->gradients.size(); i++) {
    double pred_grad = predicted->gradients[i];
    double expected_grad = expected_derivatives[i];
    ASSERT_FLOAT_EQ(pred_grad, expected_grad);
  }
}

TEST_F(TensorOpsTest, FlipKernelTest) {
  vector<int> kernel_shape{1, 3, 3, 1};
  int num_elements = 1;
  for (int i = 0; i < kernel_shape.size(); i++) {
    num_elements *= kernel_shape[i];
  }
  vector<double> kernel_values(num_elements, 1.);
  // Populate the kernel with random values
  for (int i = 0; i < num_elements; i++) {
    kernel_values[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  shared_ptr<Tensor> kernel =
      make_shared<Tensor>(kernel_values, kernel_shape, logger);
  shared_ptr<Tensor> kernel_copy =
      make_shared<Tensor>(kernel_values, kernel_shape, logger);
  flip_kernel(kernel);
  for (int f = 0; f < kernel_shape[0]; f++) {
    for (int c = 0; c < kernel_shape[3]; c++) {
      for (int j = 0; j < kernel_shape[1]; j++) {
        for (int i = 0; i < kernel_shape[2]; i++) {
          if ((kernel_shape[2] - 1) - i < i) {
            ASSERT_FLOAT_EQ(kernel->get_element(vector<int>{f, j, i, c}),
                            kernel_copy->get_element(
                                vector<int>{f, (kernel_shape[1] - 1) - j,
                                            (kernel_shape[2] - 1) - i, c}));
          }
        }
      }
    }
  }
}

TEST_F(TensorOpsTest, GetValuesIndexTest) {

  // Create input tensor with shape (batch_size, height, width, channels)
  vector<int> input_shape{2, 3, 3, 2};
  int num_elements = 1;
  for (int i = 0; i < input_shape.size(); i++) {
    num_elements *= input_shape[i];
  }
  vector<double> input_values(num_elements, 1.);
  shared_ptr<Tensor> input_tensor =
      make_shared<Tensor>(input_values, input_shape, logger);

  for (int j = 0; j < input_shape[1]; j++) {
    for (int i = 0; i < input_shape[2]; i++) {
      input_tensor->set_element(vector<int>{0, j, i, 0}, 1.);
      input_tensor->set_element(vector<int>{0, j, i, 1}, 2.);
      input_tensor->set_element(vector<int>{1, j, i, 0}, 3.);
      input_tensor->set_element(vector<int>{1, j, i, 1}, 4.);
    }
  }

  // Get values at index for dilation_input = 1, padding = 0
  logger->log(INFO, "Get values at index for dilation_input = 1, padding = 0");
  vector<double> values =
      get_values_at_index(0, 1, 1, input_tensor, 2, 2, 0, 1);
  ASSERT_EQ(values.size(), 8);
  for (int i = 0; i < values.size(); i++) {
    if (i % 2) {
      ASSERT_FLOAT_EQ(values[i], 2.);
    } else {
      ASSERT_FLOAT_EQ(values[i], 1.);
    }
  }

  // Get values at index for dilation_input = 2, padding = 1
  logger->log(INFO, "Get values at index for dilation_input = 2, padding = 0");
  values = get_values_at_index(0, 1, 1, input_tensor, 2, 2, 0, 2);
  ASSERT_EQ(values.size(), 8);
  for (int i = 0; i < 6; i++) {
    ASSERT_FLOAT_EQ(values[i], 0.);
  }
  ASSERT_FLOAT_EQ(values[6], 1.);
  ASSERT_FLOAT_EQ(values[7], 2.);

  // Get values at index for dilation_input = 2, padding = 2
  logger->log(INFO, "Get values at index for dilation_input = 2, padding = 2");
  values = get_values_at_index(0, 3, 3, input_tensor, 2, 2, 2, 2);
  ASSERT_EQ(values.size(), 8);
  for (int i = 0; i < 6; i++) {
    ASSERT_FLOAT_EQ(values[i], 0.);
  }
  ASSERT_FLOAT_EQ(values[6], 1.);
  ASSERT_FLOAT_EQ(values[7], 2.);
}

TEST_F(TensorOpsTest, ConvolutionForwardTest) {

  // Create input tensor with shape (batch_size, height, width, channels)
  vector<int> input_shape{2, 4, 4, 2};
  int num_elements = 1;
  for (int i = 0; i < input_shape.size(); i++) {
    num_elements *= input_shape[i];
  }
  vector<double> input_values(num_elements, 1.);
  shared_ptr<Tensor> input_tensor =
      make_shared<Tensor>(input_values, input_shape, logger);

  for (int j = 0; j < input_shape[1]; j++) {
    for (int i = 0; i < input_shape[2]; i++) {
      input_tensor->set_element(vector<int>{0, j, i, 0}, 1.);
      input_tensor->set_element(vector<int>{0, j, i, 1}, 2.);
      input_tensor->set_element(vector<int>{1, j, i, 0}, 3.);
      input_tensor->set_element(vector<int>{1, j, i, 1}, 4.);
    }
  }

  // Create kernel tensor with shape (number_filters, kernel_height,
  // kernel_width, channels)
  vector<int> kernel_shape{2, 2, 2, 2};
  num_elements = 1;
  for (int i = 0; i < kernel_shape.size(); i++) {
    num_elements *= kernel_shape[i];
  }
  vector<double> kernel_values(num_elements, 1.);
  for (int i = 0; i < num_elements; i++) {
    if (i % 2) {
      kernel_values[i] = 2.;
    } else {
      kernel_values[i] = 1.;
    }
  }
  for (int i = 8; i < kernel_values.size(); i++) {
    kernel_values[i] += 2.;
  }

  shared_ptr<Tensor> kernel =
      make_shared<Tensor>(kernel_values, kernel_shape, logger);

  // Initialize bias tensor with shape (1, number_filters).
  vector<double> bias_values(kernel_shape[0], 1.);
  shared_ptr<Tensor> bias =
      make_shared<Tensor>(bias_values, vector<int>{1, kernel_shape[0]}, logger);

  int stride = 2, padding = 1, dilation_input = 2, dilation_kernel = 2;

  int number_filters = kernel->shape[0];
  int kernel_height = kernel->shape[1];
  int kernel_width = kernel->shape[2];
  int channels = input_tensor->shape[3];
  int batch_size = input_tensor->shape[0];
  int height_input = input_tensor->shape[1];
  int width_input = input_tensor->shape[2];
  int width_effective = 1 + (width_input - 1) * dilation_input + 2 * padding;
  int height_effective = 1 + (height_input - 1) * dilation_input + 2 * padding;
  int dilated_kernel_height = 1 + dilation_kernel * (kernel_height - 1);
  int dilated_kernel_width = 1 + dilation_kernel * (kernel_width - 1);
  int width_output = 1 + (width_effective - dilated_kernel_width) / stride;
  int height_output = 1 + (height_effective - dilated_kernel_height) / stride;

  shared_ptr<Tensor> convolution_result =
      convolution(input_tensor, kernel, bias, stride, padding, dilation_input,
                  dilation_kernel);

  vector<int> expected_shape{batch_size, height_output, width_output,
                             number_filters};
  vector<int> result_shape = convolution_result->shape;
  ASSERT_EQ(result_shape.size(), expected_shape.size());
  for (int i = 0; i < expected_shape.size(); i++) {
    ASSERT_EQ(result_shape[i], expected_shape[i]);
  }

  // Check the values of the convolution result
  for (int b = 0; b < expected_shape[0]; b++) {
    for (int j = 0; j < expected_shape[1]; j++) {
      for (int i = 0; i < expected_shape[2]; i++) {
        for (int c = 0; c < expected_shape[3]; c++) {
          double expected_value = 0.;
          for (int v = 0; v < dilated_kernel_height; v++) {
            for (int u = 0; u < dilated_kernel_width; u++) {
              for (int ch = 0; ch < channels; ch++) {
                int row_index = j * stride + v;
                int col_index = i * stride + u;
                double input_elem = 0.;
                if (row_index < padding ||
                    row_index > padding + (input_tensor->shape[1] - 1) *
                                              dilation_input ||
                    col_index < padding ||
                    col_index > padding + (input_tensor->shape[2] - 1) *
                                              dilation_input) {
                  input_elem = 0.;
                } else if ((row_index - padding) % dilation_input != 0 ||
                           (col_index - padding) % dilation_input != 0) {
                  input_elem = 0.;
                } else {
                  int row_index_input = (row_index - padding) / dilation_input;
                  int col_index_input = (col_index - padding) / dilation_input;
                  input_elem = input_tensor->get_element(
                      vector<int>{b, row_index_input, col_index_input, ch},
                      "values");
                }
                double kernel_elem = 0.;
                if (v % dilation_kernel != 0 || u % dilation_kernel != 0) {
                  kernel_elem = 0.;
                } else {
                  int v_kernel = v / dilation_kernel;
                  int u_kernel = u / dilation_kernel;
                  kernel_elem = kernel->get_element(
                      vector<int>{c, v_kernel, u_kernel, ch}, "values");
                }
                expected_value += input_elem * kernel_elem;
              }
            }
          }
          ASSERT_FLOAT_EQ(expected_value + bias_values[c],
                          convolution_result->get_element(
                              vector<int>{b, j, i, c}, "values"));
        }
      }
    }
  }
}

TEST_F(TensorOpsTest, ConvolutionBackwardTest) {

  // Create input tensor with shape (batch_size, height, width, channels)
  vector<int> input_shape{2, 5, 5, 2};
  int num_elements = 1;
  for (int i = 0; i < input_shape.size(); i++) {
    num_elements *= input_shape[i];
  }
  vector<double> input_values(num_elements, 1.);
  shared_ptr<Tensor> input_tensor =
      make_shared<Tensor>(input_values, input_shape, logger);

  for (int j = 0; j < input_shape[1]; j++) {
    for (int i = 0; i < input_shape[2]; i++) {
      input_tensor->set_element(vector<int>{0, j, i, 0}, 1.);
      input_tensor->set_element(vector<int>{0, j, i, 1}, 2.);
      input_tensor->set_element(vector<int>{1, j, i, 0}, 3.);
      input_tensor->set_element(vector<int>{1, j, i, 1}, 4.);
    }
  }

  // Create kernel tensor with shape (number_filters, kernel_height,
  // kernel_width, channels)
  vector<int> kernel_shape{2, 2, 2, 2};
  num_elements = 1;
  for (int i = 0; i < kernel_shape.size(); i++) {
    num_elements *= kernel_shape[i];
  }
  vector<double> kernel_values(num_elements, 1.);
  for (int i = 0; i < num_elements; i++) {
    if (i % 2) {
      kernel_values[i] = 2.;
    } else {
      kernel_values[i] = 1.;
    }
  }
  for (int i = 8; i < kernel_values.size(); i++) {
    kernel_values[i] += 2.;
  }

  shared_ptr<Tensor> kernel =
      make_shared<Tensor>(kernel_values, kernel_shape, logger);

  // Initialize bias tensor with shape (1, number_filters).
  vector<double> bias_values(kernel_shape[0], 1.);
  shared_ptr<Tensor> bias =
      make_shared<Tensor>(bias_values, vector<int>{1, kernel_shape[0]}, logger);

  int stride = 2;
  int padding = 1;
  int dilation_input = 1;
  int dilation_kernel = 2;
  int dilated_kernel_height = 1 + dilation_kernel * (kernel_shape[1] - 1);
  int dilated_kernel_width = 1 + dilation_kernel * (kernel_shape[2] - 1);
  shared_ptr<Tensor> convolution_result =
      convolution(input_tensor, kernel, bias, stride, padding, dilation_input,
                  dilation_kernel);
  for (int i = 0; i < convolution_result->gradients.size(); i++) {
    convolution_result->gradients[i] = 1.;
  }
  convolution_result->backward();

  // Check the gradients with respect to the input tensor
  for (int b = 0; b < input_shape[0]; b++) {
    for (int j = 0; j < input_shape[1]; j++) {
      for (int i = 0; i < input_shape[2]; i++) {
        for (int c = 0; c < input_shape[3]; c++) {
          double expected_gradient = 0.;
          for (int v = 0; v < dilated_kernel_height; v++) {
            for (int u = 0; u < dilated_kernel_width; u++) {
              int jp = j + padding;
              int ip = i + padding;
              if ((jp - v) >= 0 &&
                  jp + (dilated_kernel_height - 1 - v) <
                      input_shape[1] + 2 * padding &&
                  (ip - u) >= 0 &&
                  ip + (dilated_kernel_width - 1 - u) <
                      input_shape[2] + 2 * padding &&
                  (jp - v) % stride == 0 && (ip - u) % stride == 0 &&
                  (v % dilation_kernel == 0) && (u % dilation_kernel == 0)) {
                for (int f = 0; f < kernel->shape[0]; f++)
                  expected_gradient +=
                      kernel->get_element(vector<int>{f, v / dilation_kernel,
                                                      u / dilation_kernel, c},
                                          "values");
              }
            }
          }
          ASSERT_FLOAT_EQ(
              input_tensor->get_element(vector<int>{b, j, i, c}, "gradients"),
              expected_gradient);
        }
      }
    }
  }

  // Check the gradients with respect to the convolution kernel. Assume cells
  // added to the input_tensor as part of padding contain zeros.
  for (int f = 0; f < kernel_shape[0]; f++) {
    for (int j = 0; j < kernel_shape[1]; j++) {
      for (int i = 0; i < kernel_shape[2]; i++) {
        for (int c = 0; c < kernel_shape[3]; c++) {
          double expected_gradient = 0.;
          for (int v = padding; v < input_shape[1] + padding; v++) {
            for (int u = padding; u < input_shape[2] + padding; u++) {
              if ((v - j * dilation_kernel) >= 0 &&
                  v + (dilated_kernel_width - 1 - j * dilation_kernel) <
                      input_shape[1] + 2 * padding &&
                  (u - i * dilation_kernel) >= 0 &&
                  u + (dilated_kernel_width - 1 - i * dilation_kernel) <
                      input_shape[2] + 2 * padding &&
                  (v - j * dilation_kernel) % stride == 0 &&
                  (u - i * dilation_kernel) % stride == 0) {
                for (int b = 0; b < input_shape[0]; b++) {
                  expected_gradient += input_tensor->get_element(
                      vector<int>{b, v - padding, u - padding, c}, "values");
                }
              }
            }
          }
          ASSERT_FLOAT_EQ(
              kernel->get_element(vector<int>{f, j, i, c}, "gradients"),
              expected_gradient);
        }
      }
    }
  }

  // Check the gradients with respect to the bias tensor
  int number_elements_per_filter = convolution_result->shape[0] *
                                   convolution_result->shape[1] *
                                   convolution_result->shape[2];
  for (int f = 0; f < bias->shape[1]; f++) {
    ASSERT_FLOAT_EQ(bias->get_element(vector<int>{0, f}, "gradients"),
                    number_elements_per_filter);
  }
}
