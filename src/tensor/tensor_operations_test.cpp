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

TEST_F(TensorOpsTest, ConvolutionTest) {

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

  ASSERT_FLOAT_EQ(bias_values[0], 1.);
  shared_ptr<Tensor> convolution_result =
      convolution(input_tensor, kernel, bias, 1, 0, 1, 1);

  vector<int> expected_shape{2, 2, 2, 2};
  vector<int> result_shape = convolution_result->shape;
  ASSERT_EQ(result_shape.size(), expected_shape.size());
  for (int i = 0; i < expected_shape.size(); i++) {
    ASSERT_EQ(result_shape[i], expected_shape[i]);
  }

  unordered_map<int, double> expected_results;

  for (int i = 0; i < expected_shape[0]; i++) {
    for (int j = 0; j < expected_shape[1]; j++) {
      // Calculate the index using szudzik's pairing function
      int index = i >= j ? i * i + i + j : i + j * j;
      if (i == 0 != j == 0)
        expected_results[index] = 45.;
      else if (i == 0 && j == 0)
        expected_results[index] = 21.;
      else
        expected_results[index] = 101.;
    }
  }

  // Check the values of the convolution result

  for (int b = 0; b < expected_shape[0]; b++) {
    for (int j = 0; j < expected_shape[1]; j++) {
      for (int i = 0; i < expected_shape[2]; i++) {
        for (int c = 0; c < expected_shape[3]; c++) {
          int index = b >= c ? b * b + b + c : b + c * c;
          ASSERT_FLOAT_EQ(
              convolution_result->get_element(vector<int>{b, j, i, c}),
              expected_results[index]);
        }
      }
    }
  }
}

TEST_F(TensorOpsTest, ConvolutionBackwardTest) {

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

  shared_ptr<Tensor> convolution_result =
      convolution(input_tensor, kernel, bias, 1, 0, 1, 1);
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
          for (int v = 0; v < kernel->shape[1]; v++) {
            for (int u = 0; u < kernel->shape[2]; u++) {
              if ((j - v) >= 0 &&
                  j + (kernel->shape[1] - 1 - v) < input_shape[1] &&
                  (i - u) >= 0 &&
                  i + (kernel->shape[2] - 1 - u) < input_shape[2]) {
                for (int f = 0; f < kernel->shape[0]; f++)
                  expected_gradient +=
                      kernel->get_element(vector<int>{f, v, u, c}, "values");
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

  // Check the gradients with respect to the convolution kernel
  for (int f = 0; f < kernel_shape[0]; f++) {
    for (int j = 0; j < kernel_shape[1]; j++) {
      for (int i = 0; i < kernel_shape[2]; i++) {
        for (int c = 0; c < kernel_shape[3]; c++) {
          double expected_gradient = 0.;
          for (int v = 0; v < input_shape[1]; v++) {
            for (int u = 0; u < input_shape[2]; u++) {
              if ((v - j) >= 0 &&
                  v + (kernel_shape[1] - 1 - j) < input_shape[1] &&
                  (u - i) >= 0 &&
                  u + (kernel_shape[2] - 1 - i) < input_shape[2]) {
                for (int b = 0; b < input_shape[0]; b++) {
                  expected_gradient += input_tensor->get_element(
                      vector<int>{b, v, u, c}, "values");
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
  // for (int f = 0; f < bias->shape[1]; f++) {
  //   ASSERT_FLOAT_EQ(bias->get_element(vector<int>{0, f}, "gradients"), 4.);
  // }
}

TEST_F(TensorOpsTest, ConvolutionBackwardStrideTwoTest) {

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

  int stride = 2;
  shared_ptr<Tensor> convolution_result =
      convolution(input_tensor, kernel, bias, stride, 0, 1, 1);
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
          for (int v = 0; v < kernel->shape[1]; v++) {
            for (int u = 0; u < kernel->shape[2]; u++) {
              if ((j - v) >= 0 &&
                  j + (kernel->shape[1] - 1 - v) < input_shape[1] &&
                  (i - u) >= 0 &&
                  i + (kernel->shape[2] - 1 - u) < input_shape[2] &&
                  (j - v) % stride == 0 && (i - u) % stride == 0) {
                for (int f = 0; f < kernel->shape[0]; f++)
                  expected_gradient +=
                      kernel->get_element(vector<int>{f, v, u, c}, "values");
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

  // Check the gradients with respect to the convolution kernel
  for (int f = 0; f < kernel_shape[0]; f++) {
    for (int j = 0; j < kernel_shape[1]; j++) {
      for (int i = 0; i < kernel_shape[2]; i++) {
        for (int c = 0; c < kernel_shape[3]; c++) {
          double expected_gradient = 0.;
          for (int v = 0; v < input_shape[1]; v++) {
            for (int u = 0; u < input_shape[2]; u++) {
              if ((v - j) >= 0 &&
                  v + (kernel_shape[1] - 1 - j) < input_shape[1] &&
                  (u - i) >= 0 &&
                  u + (kernel_shape[2] - 1 - i) < input_shape[2] &&
                  (v - j) % stride == 0 && (u - i) % stride == 0) {
                for (int b = 0; b < input_shape[0]; b++) {
                  expected_gradient += input_tensor->get_element(
                      vector<int>{b, v, u, c}, "values");
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
  // for (int f = 0; f < bias->shape[1]; f++) {
  //   ASSERT_FLOAT_EQ(bias->get_element(vector<int>{0, f}, "gradients"), 4.);
  // }
}
