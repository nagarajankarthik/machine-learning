#include "tensor.h"
#include "tensor_operations.h"
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

