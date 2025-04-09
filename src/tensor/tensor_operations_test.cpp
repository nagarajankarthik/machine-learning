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
	    INFO,
	    "Created tensor c by concatenating a and b along batch dimension.");
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
	logger->log(INFO,
		    "Created tensor c by applying sigmoid function to a.");
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
	logger->log(INFO,
		    "Created tensor c by applying softmax function to a.");
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

	vector<vector<double>> softmax_values =
	    c->get_matrix(position, "values");
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
				 softmax_derivatives, softmax_values,
				 logger))[j]);
			ASSERT_FLOAT_EQ(calculated_derivatives[i][j],
					expected_derivative);
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
	logger->log(
	    INFO,
	    "Calculated loss based on ground truth and predicted values.");
	ASSERT_EQ(loss->values.size(), 2);
	vector<int> loss_shape = loss->shape;
	ASSERT_EQ(loss_shape.size(), 3);
	ASSERT_EQ(loss_shape[0], predicted->shape[0]);
	ASSERT_EQ(loss_shape[1], 1);
	ASSERT_EQ(loss_shape[2], predicted->shape[2]);
	vector<int> position{0};
	vector<vector<double>> loss_values =
	    loss->get_matrix(position, "values");
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
	logger->log(
	    INFO,
	    "Calculated loss based on ground truth and predicted values.");
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
