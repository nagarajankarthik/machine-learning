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
	logger->log(INFO, "Created tensor a.");
	shared_ptr<Tensor> c = relu_forward(a);
	logger->log(INFO, "Created tensor c by applying relu to a.");
	vector<int> position{0};
	fill(c->gradients.begin(), c->gradients.end(), 1.);
	logger->log(INFO, "Filled gradients with 1.");
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

TEST_F(TensorOpsTest, DISABLED_SigmoidBackwardTest) {

	a->values = {-1., -1., 1., 1.};
	logger->log(INFO, "Created tensor a.");
	shared_ptr<Tensor> c = sigmoid_forward(a);
	logger->log(INFO, "Created tensor c by applying relu to a.");
	vector<int> position{0};
	fill(c->gradients.begin(), c->gradients.end(), 1.);
	logger->log(INFO, "Filled gradients with 1.");
	c->backward();
	vector<vector<double>> grad = a->get_matrix(position, "gradients");
	for (int j = 0; j < grad[0].size(); j++) {
		ASSERT_FLOAT_EQ(grad[0][j], 0.);
	}
	for (int j = 0; j < grad[0].size(); j++) {
		ASSERT_FLOAT_EQ(grad[1][j], 1.);
	}
}
