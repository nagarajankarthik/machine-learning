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

TEST_F(TensorOpsTest, BatchMatmulTest) {
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

