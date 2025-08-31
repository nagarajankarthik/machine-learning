#include "tensor.h"
#include <math.h>
using namespace std;

namespace ml {

inline vector<int> broadcast_shape(vector<int> t1_shape, vector<int> t2_shape,
                                   shared_ptr<Logger> logger,
                                   int start_index = 2) {
  vector<int> new_shape{};
  int m = t1_shape.size();
  int n = t2_shape.size();

  for (int i = start_index; i < max(m, n); i++) {
    int first_index = m - 1 - i;
    int second_index = n - 1 - i;
    if (first_index < 0)
      new_shape.push_back(t2_shape[second_index]);
    else if (second_index < 0)
      new_shape.push_back(t1_shape[first_index]);
    else if (t1_shape[first_index] == t2_shape[second_index])
      new_shape.push_back(t1_shape[first_index]);
    else {
      int min_size = min(t1_shape[first_index], t2_shape[second_index]);
      int max_size = max(t1_shape[first_index], t2_shape[second_index]);
      if (min_size == 1)
        new_shape.push_back(max_size);
      else {
        logger->log(ERROR, "Size mismatch at non-singleton index " +
                               to_string(i) +
                               " from the end of two tensors' shape "
                               "arrays.\n The size of tensor a is " +
                               to_string(t1_shape[first_index]) +
                               " and the size of tensor b is " +
                               to_string(t2_shape[second_index]));
        exit(1);
      }
    }
  }
  reverse(new_shape.begin(), new_shape.end());
  return new_shape;
}

inline vector<vector<double>> add_matrix(vector<vector<double>> m1,
                                         vector<vector<double>> m2,
                                         shared_ptr<Logger> logger) {

  if (m1.size() != m2.size()) {
    logger->log(ERROR, "Error in add_matrix: First input has " +
                           to_string(m1.size()) +
                           " rows but second input has " +
                           to_string(m2.size()) + " rows.");
    exit(1);
  }

  if (m1[0].size() != m2[0].size()) {
    logger->log(ERROR, "Error in add_matrix: First input has " +
                           to_string(m1[0].size()) +
                           " columns but second input has " +
                           to_string(m2[0].size()) + " columns.");
    exit(1);
  }
  vector<vector<double>> matrix_sum(m1.size(), vector<double>(m1[0].size()));
  for (int i = 0; i < m1.size(); i++) {
    for (int j = 0; j < m1[0].size(); j++) {
      matrix_sum[i][j] = m1[i][j] + m2[i][j];
    }
  }
  return matrix_sum;
}

inline void recurse_add_backward(shared_ptr<Tensor> t3, shared_ptr<Tensor> t1,
                                 shared_ptr<Tensor> t2,
                                 vector<int> new_position, int axis = 0) {

  if (axis == t3->shape.size() - 2) {
    vector<vector<double>> g3 = t3->get_matrix(new_position, "gradients");
    t1->set_matrix(new_position, g3, "gradients");
    t2->set_matrix(new_position, g3, "gradients");
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_add_backward(t3, t1, t2, new_position, axis + 1);
  }
  new_position.pop_back();
}

inline void add_batch_backward(shared_ptr<Tensor> t3) {

  shared_ptr<Tensor> t1 = t3->input_first;
  if (!t1)
    return;
  shared_ptr<Tensor> t2 = t3->input_second;

  shared_ptr<Logger> logger = t1->logger;

  vector<int> new_position{};
  recurse_add_backward(t3, t1, t2, new_position);
}

inline void recurse_add_forward(shared_ptr<Tensor> t3,
                                const shared_ptr<Tensor> t1,
                                const shared_ptr<Tensor> t2,
                                vector<int> new_position, int axis = 0) {

  if (axis == t3->shape.size() - 2) {
    vector<vector<double>> m1 = t1->get_matrix(new_position, "values");
    vector<vector<double>> m2 = t2->get_matrix(new_position, "values");
    vector<vector<double>> matrix_sum = add_matrix(m1, m2, t1->logger);
    t3->set_matrix(new_position, matrix_sum);
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_add_forward(t3, t1, t2, new_position, axis + 1);
  }
  new_position.pop_back();
}

inline shared_ptr<Tensor> add_batch_forward(const shared_ptr<Tensor> t1,
                                            const shared_ptr<Tensor> t2) {

  shared_ptr<Logger> logger = t1->logger;
  vector<int> new_shape = broadcast_shape(t1->shape, t2->shape, logger);
  int m = t1->shape.size();
  int n = t2->shape.size();

  if (t1->shape[m - 2] != t2->shape[n - 2]) {
    logger->log(ERROR, " Error in add_batch_forward: At the second last "
                       "dimension, the shapes are  " +
                           to_string(t1->shape[m - 2]) + " and " +
                           to_string(t2->shape[n - 2]) +
                           " for the two input tensors.");
    exit(1);
  }

  if (t1->shape[m - 1] != t2->shape[n - 1]) {
    logger->log(ERROR, " Error in add_batch_forward: At the last "
                       "dimension, the shapes are  " +
                           to_string(t1->shape[m - 1]) + " and " +
                           to_string(t2->shape[n - 1]) +
                           " for the two input tensors.");
    exit(1);
  }

  new_shape.push_back(t1->shape[m - 2]);
  new_shape.push_back(t1->shape[n - 1]);

  int number_of_values = 1;
  for (int i = 0; i < new_shape.size(); i++) {
    number_of_values *= new_shape[i];
  }

  vector<double> sum_values(number_of_values, 0.);

  shared_ptr<Tensor> t3 = make_shared<Tensor>(sum_values, new_shape, logger, t1,
                                              t2, add_batch_backward);
  vector<int> new_position{};
  recurse_add_forward(t3, t1, t2, new_position);
  return t3;
}

// Generic addition operation for two tensors unrelated to matrix addition

inline void recurse_add_tensor_backward(shared_ptr<Tensor> t3,
                                        shared_ptr<Tensor> t1,
                                        shared_ptr<Tensor> t2,
                                        vector<int> new_position,
                                        int axis = 0) {

  if (axis == t3->shape.size()) {
    double g3 = t3->get_element(new_position, "gradients");
    t1->set_element(new_position, g3, "gradients");
    t2->set_element(new_position, g3, "gradients");
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_add_tensor_backward(t3, t1, t2, new_position, axis + 1);
  }
  new_position.pop_back();
}

inline void add_tensor_backward(shared_ptr<Tensor> t3) {

  shared_ptr<Tensor> t1 = t3->input_first;
  if (!t1)
    return;
  shared_ptr<Tensor> t2 = t3->input_second;

  shared_ptr<Logger> logger = t1->logger;

  vector<int> new_position{};
  recurse_add_tensor_backward(t3, t1, t2, new_position);
}

inline void recurse_add_tensor_forward(shared_ptr<Tensor> t3,
                                       const shared_ptr<Tensor> t1,
                                       const shared_ptr<Tensor> t2,
                                       vector<int> new_position, int axis = 0) {

  if (axis == t3->shape.size()) {
    double v1 = t1->get_element(new_position);
    double v2 = t2->get_element(new_position);
    t3->set_element(new_position, v1 + v2);
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_add_tensor_forward(t3, t1, t2, new_position, axis + 1);
  }
  new_position.pop_back();
}

inline shared_ptr<Tensor> add_tensor_forward(const shared_ptr<Tensor> t1,
                                             const shared_ptr<Tensor> t2) {

  shared_ptr<Logger> logger = t1->logger;
  vector<int> new_shape = broadcast_shape(t1->shape, t2->shape, logger, 0);

  int number_of_values = 1;
  for (int i = 0; i < new_shape.size(); i++) {
    number_of_values *= new_shape[i];
  }

  vector<double> sum_values(number_of_values, 0.);

  shared_ptr<Tensor> t3 = make_shared<Tensor>(sum_values, new_shape, logger, t1,
                                              t2, add_tensor_backward);
  vector<int> new_position{};
  recurse_add_tensor_forward(t3, t1, t2, new_position);
  return t3;
}

inline vector<int> get_shape_after_matmul(shared_ptr<Tensor> t1,
                                          shared_ptr<Tensor> t2) {
  int m = t1->shape.size();
  int n = t2->shape.size();
  shared_ptr<Logger> logger = t1->logger;

  vector<int> new_shape = broadcast_shape(t1->shape, t2->shape, logger);

  // Process matrix dimensions
  if (t1->shape[m - 1] != t2->shape[n - 2]) {
    logger->log(ERROR, "Size of tensors unsuitable for batch "
                       "matrix multiplication.");
    logger->log(ERROR, "Size of last dimension of t1 is " +
                           to_string(t1->shape[m - 1]) +
                           " but size of second to last dimension of t2 is " +
                           to_string(t2->shape[n - 2]));
    exit(1);
  }
  new_shape.push_back(t1->shape[m - 2]);
  new_shape.push_back(t2->shape[n - 1]);

  return new_shape;
}

inline vector<vector<double>> matmul(vector<vector<double>> m1,
                                     vector<vector<double>> m2,
                                     shared_ptr<Logger> logger) {

  if (m1[0].size() != m2.size()) {
    logger->log(ERROR, "Invalid matrix dimensions");
    exit(1);
  }

  vector<vector<double>> matrix_product(m1.size(),
                                        vector<double>(m2[0].size()));
  for (int i = 0; i < m1.size(); i++) {
    for (int j = 0; j < m2[0].size(); j++) {
      double sum = 0;
      for (int k = 0; k < m2.size(); k++) {
        sum += m1[i][k] * m2[k][j];
      }
      matrix_product[i][j] = sum;
    }
  }
  return matrix_product;
}

inline vector<vector<double>> transpose_matrix(vector<vector<double>> matrix) {

  vector<vector<double>> transpose(matrix[0].size(),
                                   vector<double>(matrix.size()));
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[0].size(); j++) {
      transpose[j][i] = matrix[i][j];
    }
  }
  return transpose;
}

inline void recurse_matmul_backward(const shared_ptr<Tensor> t3,
                                    shared_ptr<Tensor> t1,
                                    shared_ptr<Tensor> t2,
                                    vector<int> &new_position, int axis = 0) {

  if (axis == t3->shape.size() - 2) {
    vector<vector<double>> g3 = t3->get_matrix(new_position, "gradients");
    vector<vector<double>> m1 = t1->get_matrix(new_position);
    vector<vector<double>> m2 = t2->get_matrix(new_position);
    vector<vector<double>> m1_transpose = transpose_matrix(m1);
    vector<vector<double>> m2_transpose = transpose_matrix(m2);
    vector<vector<double>> g1 = matmul(g3, m2_transpose, t1->logger);
    vector<vector<double>> g2 = matmul(m1_transpose, g3, t1->logger);
    t1->set_matrix(new_position, g1, "gradients");
    t2->set_matrix(new_position, g2, "gradients");
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_matmul_backward(t3, t1, t2, new_position, axis + 1);
  }
  new_position.pop_back();
}

inline void batch_matmul_backward(shared_ptr<Tensor> t3) {

  shared_ptr<Tensor> t1 = t3->input_first;
  if (!t1)
    return;
  shared_ptr<Tensor> t2 = t3->input_second;

  shared_ptr<Logger> logger = t1->logger;

  vector<int> new_position{};
  recurse_matmul_backward(t3, t1, t2, new_position);
}

inline void recurse_matmul_forward(shared_ptr<Tensor> t3,
                                   const shared_ptr<Tensor> t1,
                                   const shared_ptr<Tensor> t2,
                                   vector<int> &new_position, int axis = 0) {

  if (axis == t3->shape.size() - 2) {
    vector<vector<double>> m1 = t1->get_matrix(new_position);
    vector<vector<double>> m2 = t2->get_matrix(new_position);
    vector<vector<double>> matrix_product = matmul(m1, m2, t1->logger);
    t3->set_matrix(new_position, matrix_product);
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_matmul_forward(t3, t1, t2, new_position, axis + 1);
  }
  new_position.pop_back();
}

inline shared_ptr<Tensor> batch_matmul_forward(shared_ptr<Tensor> t1,
                                               shared_ptr<Tensor> t2) {

  shared_ptr<Logger> logger = t1->logger;
  vector<int> new_shape = get_shape_after_matmul(t1, t2);
  int new_size = 1;
  for (int i = 0; i < new_shape.size(); i++) {
    new_size *= new_shape[i];
  }

  function<void(shared_ptr<Tensor>)> batch_matmul_back = batch_matmul_backward;

  shared_ptr<Tensor> t3 =
      make_shared<Tensor>(vector<double>(new_size, 0.), new_shape, logger, t1,
                          t2, batch_matmul_backward);

  vector<int> new_position{};
  recurse_matmul_forward(t3, t1, t2, new_position);
  return t3;
}

inline void recurse_concatenate_backward(const shared_ptr<Tensor> t3,
                                         shared_ptr<Tensor> t1,
                                         shared_ptr<Tensor> t2,
                                         vector<int> &new_position,
                                         int axis = 0, int concat_dim = 0,
                                         bool use_first = true) {

  if (axis == t3->shape.size()) {
    vector<int> new_pos(new_position.begin(), new_position.end());
    if (!use_first)
      new_pos[concat_dim] += t1->shape[concat_dim];
    double g3 = t3->get_element(new_pos, "gradients");
    if (use_first)
      t1->set_element(new_position, g3, "gradients");
    else
      t2->set_element(new_position, g3, "gradients");
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    if (axis == concat_dim && i >= t1->shape[axis]) {
      use_first = false;
      new_position[position_index] = i - t1->shape[axis];
    }
    recurse_concatenate_backward(t3, t1, t2, new_position, axis + 1, concat_dim,
                                 use_first);
  }
  new_position.pop_back();
}

inline void concatenate_backward(shared_ptr<Tensor> t3, int concat_dim) {

  shared_ptr<Tensor> t1 = t3->input_first;
  if (!t1)
    return;
  shared_ptr<Tensor> t2 = t3->input_second;

  shared_ptr<Logger> logger = t1->logger;

  vector<int> new_position{};
  recurse_concatenate_backward(t3, t1, t2, new_position, 0, concat_dim);
}

inline void recurse_concatenate_forward(shared_ptr<Tensor> t3,
                                        shared_ptr<Tensor> t1,
                                        shared_ptr<Tensor> t2,
                                        vector<int> &new_position, int axis = 0,
                                        int concat_dim = 0,
                                        bool use_first = true) {

  if (axis == t3->shape.size()) {
    vector<int> new_pos(new_position.begin(), new_position.end());
    if (!use_first)
      new_pos[concat_dim] += t1->shape[concat_dim];
    double value = use_first ? t1->get_element(new_position)
                             : t2->get_element(new_position);
    double gradient = use_first ? t1->get_element(new_position, "gradients")
                                : t2->get_element(new_position, "gradients");
    t3->set_element(new_pos, value);
    t3->set_element(new_pos, gradient, "gradients");
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    if (axis == concat_dim && i >= t1->shape[axis]) {
      use_first = false;
      new_position[position_index] = i - t1->shape[axis];
    }
    recurse_concatenate_forward(t3, t1, t2, new_position, axis + 1, concat_dim,
                                use_first);
  }
  new_position.pop_back();
}

inline shared_ptr<Tensor> concatenate_forward(shared_ptr<Tensor> t1,
                                              shared_ptr<Tensor> t2,
                                              int concat_dim = 0) {
  vector<int> first_shape = t1->shape;
  vector<int> second_shape = t2->shape;
  shared_ptr<Logger> logger = t1->logger;

  if (first_shape.size() != second_shape.size()) {
    logger->log(ERROR, "Error in concatenate_forward: Input "
                       "Tensors have different shapes.");
    exit(1);
  }

  for (int i = 0; i < first_shape.size(); i++) {
    if (i != concat_dim && first_shape[i] != second_shape[i]) {
      logger->log(ERROR, "Error in concatenate_forward: Input Tensors have "
                         "different shapes along dimension " +
                             to_string(i) + ".");
      logger->log(ERROR,
                  "Concatenation dimension is " + to_string(concat_dim) + ".");
      exit(1);
    }
  }

  vector<int> new_shape = first_shape;
  new_shape[concat_dim] += second_shape[concat_dim];
  int new_size = t1->values.size() + t2->values.size();
  /**
   * See
   * https://stackoverflow.com/questions/30217956/error-variable-cannot-be-implicitly-captured-because-no-default-capture-mode-h
   * https://stackoverflow.com/questions/55124517/stdfunction-and-stdbind-return-value
   * As explained in the second of the two links above, the next 3 lines
   * of code are commented out because the lambda function method is
   * preferred over the std::bind method
   */

  // std::function<void(shared_ptr<Tensor>)> concat_back =
  //     std::bind(concatenate_backward, std::placeholders::_1,
  //     concat_dim);
  auto concat_back = [concat_dim](shared_ptr<Tensor> t3) {
    concatenate_backward(t3, concat_dim);
  };

  shared_ptr<Tensor> t3 = make_shared<Tensor>(
      vector<double>(new_size, 0.), new_shape, logger, t1, t2, concat_back);
  vector<int> new_position{};
  recurse_concatenate_forward(t3, t1, t2, new_position, 0, concat_dim);
  return t3;
}

inline void recurse_concatenate_backward_matrix(
    const shared_ptr<Tensor> t3, shared_ptr<Tensor> t1, shared_ptr<Tensor> t2,
    vector<int> &new_position, int axis = 0, int concat_dim = 0,
    bool use_first = true) {

  if (axis == t3->shape.size() - 2) {
    vector<vector<double>> g3 = t3->get_matrix(new_position, "gradients");
    if (concat_dim < axis) {
      if (use_first)
        t1->set_matrix(new_position, g3, "gradients");
      else
        t2->set_matrix(new_position, g3, "gradients");
    } else if (concat_dim == axis) {
      vector<vector<double>> g1{}, g2{};
      g1.assign(g3.begin(), g3.begin() + t1->shape[axis]);
      g2.assign(g3.begin() + t1->shape[axis], g3.end());
      t1->set_matrix(new_position, g1, "gradients");
      t2->set_matrix(new_position, g2, "gradients");
    } else if (concat_dim == axis + 1) {
      vector<double> tmp{};
      vector<vector<double>> g1(t1->shape[axis], tmp), g2(t1->shape[axis], tmp);
      for (int i = 0; i < t1->shape[axis]; i++) {
        g1[i].assign(g3[i].begin(), g3[i].begin() + t1->shape[axis + 1]);
        g2[i].assign(g3[i].begin() + t1->shape[axis + 1], g3[i].end());
      }
      t1->set_matrix(new_position, g1, "gradients");
      t2->set_matrix(new_position, g2, "gradients");
    }
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    if (axis == concat_dim && i >= t1->shape[axis]) {
      use_first = false;
      new_position[position_index] = i - t1->shape[axis];
    }
    recurse_concatenate_backward_matrix(t3, t1, t2, new_position, axis + 1,
                                        concat_dim, use_first);
  }
  new_position.pop_back();
}

inline void concatenate_backward_matrix(shared_ptr<Tensor> t3, int concat_dim) {

  shared_ptr<Tensor> t1 = t3->input_first;
  if (!t1)
    return;
  shared_ptr<Tensor> t2 = t3->input_second;

  shared_ptr<Logger> logger = t1->logger;

  vector<int> new_position{};
  recurse_concatenate_backward_matrix(t3, t1, t2, new_position, 0, concat_dim);
}

inline void recurse_concatenate_forward_matrix(shared_ptr<Tensor> t3,
                                               shared_ptr<Tensor> t1,
                                               shared_ptr<Tensor> t2,
                                               vector<int> &new_position,
                                               int axis = 0, int concat_dim = 0,
                                               bool use_first = true) {

  if (axis == t3->shape.size() - 2) {
    if (concat_dim < axis) {
      vector<vector<double>> current_matrix{}, current_gradients{};
      if (use_first) {
        current_matrix = t1->get_matrix(new_position, "values");
        current_gradients = t1->get_matrix(new_position, "gradients");
      } else {
        current_matrix = t2->get_matrix(new_position, "values");
        current_gradients = t2->get_matrix(new_position, "gradients");
      }
      vector<int> new_pos(new_position.begin(), new_position.end());
      if (!use_first)
        new_pos[concat_dim] += t1->shape[concat_dim];
      t3->set_matrix(new_pos, current_matrix, "values");
      t3->set_matrix(new_pos, current_gradients, "gradients");
    } else if (concat_dim == axis) {
      vector<vector<double>> first_matrix =
          t1->get_matrix(new_position, "values");
      vector<vector<double>> first_gradients =
          t1->get_matrix(new_position, "gradients");
      vector<vector<double>> second_matrix =
          t2->get_matrix(new_position, "values");
      vector<vector<double>> second_gradients =
          t2->get_matrix(new_position, "gradients");
      first_matrix.insert(first_matrix.end(), second_matrix.begin(),
                          second_matrix.end());
      first_gradients.insert(first_gradients.end(), second_gradients.begin(),
                             second_gradients.end());
      t3->set_matrix(new_position, first_matrix, "values");
      t3->set_matrix(new_position, first_gradients, "gradients");
    } else if (concat_dim == axis + 1) {
      vector<vector<double>> first_matrix =
          t1->get_matrix(new_position, "values");
      vector<vector<double>> first_gradients =
          t1->get_matrix(new_position, "gradients");
      vector<vector<double>> second_matrix =
          t2->get_matrix(new_position, "values");
      vector<vector<double>> second_gradients =
          t2->get_matrix(new_position, "gradients");
      for (int i = 0; i < first_matrix.size(); i++) {
        first_matrix[i].insert(first_matrix[i].end(), second_matrix[i].begin(),
                               second_matrix[i].end());
        first_gradients[i].insert(first_gradients[i].end(),
                                  second_gradients[i].begin(),
                                  second_gradients[i].end());
      }
      t3->set_matrix(new_position, first_matrix, "values");
      t3->set_matrix(new_position, first_gradients, "gradients");
    }

    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    if (axis == concat_dim && i >= t1->shape[axis]) {
      use_first = false;
      new_position[position_index] = i - t1->shape[axis];
    }
    recurse_concatenate_forward_matrix(t3, t1, t2, new_position, axis + 1,
                                       concat_dim, use_first);
  }
  new_position.pop_back();
}

inline shared_ptr<Tensor> concatenate_forward_matrix(shared_ptr<Tensor> t1,
                                                     shared_ptr<Tensor> t2,
                                                     int concat_dim = 0) {
  vector<int> first_shape = t1->shape;
  vector<int> second_shape = t2->shape;
  shared_ptr<Logger> logger = t1->logger;

  if (first_shape.size() != second_shape.size()) {
    logger->log(ERROR, "Error in concatenate_forward: Input "
                       "Tensors have different shapes.");
    exit(1);
  }

  for (int i = 0; i < first_shape.size(); i++) {
    if (i != concat_dim && first_shape[i] != second_shape[i]) {
      logger->log(ERROR, "Error in concatenate_forward: Input Tensors have "
                         "different shapes along dimension " +
                             to_string(i) + ".");
      logger->log(ERROR,
                  "Concatenation dimension is " + to_string(concat_dim) + ".");
      exit(1);
    }
  }

  vector<int> new_shape = first_shape;
  new_shape[concat_dim] += second_shape[concat_dim];
  int new_size = t1->values.size() + t2->values.size();
  /**
   * See
   * https://stackoverflow.com/questions/30217956/error-variable-cannot-be-implicitly-captured-because-no-default-capture-mode-h
   * https://stackoverflow.com/questions/55124517/stdfunction-and-stdbind-return-value
   * As explained in the second of the two links above, the next 3 lines
   * of code are commented out because the lambda function method is
   * preferred over the std::bind method
   */

  // std::function<void(shared_ptr<Tensor>)> concat_back =
  //     std::bind(concatenate_backward, std::placeholders::_1,
  //     concat_dim);
  auto concat_back = [concat_dim](shared_ptr<Tensor> t3) {
    concatenate_backward_matrix(t3, concat_dim);
  };

  shared_ptr<Tensor> t3 = make_shared<Tensor>(
      vector<double>(new_size, 0.), new_shape, logger, t1, t2, concat_back);
  vector<int> new_position{};
  recurse_concatenate_forward_matrix(t3, t1, t2, new_position, 0, concat_dim);
  return t3;
}

// Activation functions
inline void relu_backward(shared_ptr<Tensor> t3) {

  shared_ptr<Tensor> t1 = t3->input_first;
  if (!t1)
    return;
  for (int i = 0; i < t1->values.size(); i++) {
    t1->gradients[i] = t3->values[i] > 0. ? t3->gradients[i] : 0.;
  }
}

inline shared_ptr<Tensor> relu_forward(shared_ptr<Tensor> t1) {
  shared_ptr<Logger> logger = t1->logger;
  shared_ptr<Tensor> t3 =
      make_shared<Tensor>(vector<double>(t1->values.size(), 0.), t1->shape,
                          logger, t1, nullptr, relu_backward);
  for (int i = 0; i < t1->values.size(); i++) {
    t3->values[i] = max(0., t1->values[i]);
  }
  return t3;
}

inline void sigmoid_backward(shared_ptr<Tensor> t3) {

  shared_ptr<Tensor> t1 = t3->input_first;
  if (!t1)
    return;
  for (int i = 0; i < t1->values.size(); i++) {
    double function_result = t3->values[i];
    t1->gradients[i] =
        function_result * (1. - function_result) * t3->gradients[i];
  }
}

inline shared_ptr<Tensor> sigmoid_forward(shared_ptr<Tensor> t1) {
  shared_ptr<Logger> logger = t1->logger;
  shared_ptr<Tensor> t3 =
      make_shared<Tensor>(vector<double>(t1->values.size(), 0.), t1->shape,
                          logger, t1, nullptr, sigmoid_backward);
  for (int i = 0; i < t1->values.size(); i++) {
    t3->values[i] = 1. / (1. + exp(-t1->values[i]));
  }
  return t3;
}

inline void tanh_backward(shared_ptr<Tensor> t3) {

  shared_ptr<Tensor> t1 = t3->input_first;
  if (!t1)
    return;
  for (int i = 0; i < t1->values.size(); i++) {
    double function_result = t3->values[i];
    t1->gradients[i] =
        (1. - function_result * function_result) * t3->gradients[i];
  }
}

inline shared_ptr<Tensor> tanh_forward(shared_ptr<Tensor> t1) {
  shared_ptr<Logger> logger = t1->logger;
  shared_ptr<Tensor> t3 =
      make_shared<Tensor>(vector<double>(t1->values.size(), 0.), t1->shape,
                          logger, t1, nullptr, tanh_backward);
  for (int i = 0; i < t1->values.size(); i++) {
    t3->values[i] = tanh(t1->values[i]);
  }
  return t3;
}

inline vector<vector<double>>
elementwise_multiplication(vector<vector<double>> m1, vector<vector<double>> m2,
                           shared_ptr<Logger> logger) {
  if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
    logger->log(ERROR, "Matrix dimensions do not match in "
                       "elementwise_multiplication.");
    exit(1);
  }
  vector<vector<double>> result(m1.size(), vector<double>(m1[0].size(), 0.));
  for (int i = 0; i < m1.size(); i++) {
    for (int j = 0; j < m1[0].size(); j++) {
      result[i][j] = m1[i][j] * m2[i][j];
    }
  }
  return result;
}

/**
 * Returns the sum of the values in each column of the input matrix.
 */
inline vector<double> matrix_col_sum(vector<vector<double>> matrix) {
  vector<double> result(matrix[0].size(), 0.);
  for (int j = 0; j < matrix[0].size(); j++) {
    double sum = 0.;
    for (int i = 0; i < matrix.size(); i++) {
      sum += matrix[i][j];
    }
    result[j] = sum;
  }
  return result;
}

inline void recurse_softmax_backward(const shared_ptr<Tensor> t3,
                                     shared_ptr<Tensor> t1,
                                     vector<int> &new_position, int axis = 0) {

  if (axis == t3->shape.size() - 2) {
    vector<vector<double>> g3 = t3->get_matrix(new_position, "gradients");
    vector<vector<double>> m3 = t3->get_matrix(new_position);
    vector<vector<double>> gradient_values_product =
        elementwise_multiplication(g3, m3, t1->logger);
    vector<double> gradient_values_product_sum =
        matrix_col_sum(gradient_values_product);

    vector<vector<double>> g1(g3.size(), vector<double>(g3[0].size(), 0.));
    for (int j = 0; j < g1[0].size(); j++) {
      for (int i = 0; i < g1.size(); i++) {
        g1[i][j] = m3[i][j] * (g3[i][j] - gradient_values_product_sum[j]);
      }
    }
    t1->set_matrix(new_position, g1, "gradients");
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_softmax_backward(t3, t1, new_position, axis + 1);
  }
  new_position.pop_back();
}

inline void softmax_backward(shared_ptr<Tensor> t3) {

  shared_ptr<Tensor> t1 = t3->input_first;
  if (!t1)
    return;
  vector<int> new_position{};
  recurse_softmax_backward(t3, t1, new_position);
}

inline vector<vector<double>> evaluate_softmax(const vector<vector<double>> &m1,
                                               shared_ptr<Logger> logger) {
  vector<vector<double>> softmax_result(m1.size(),
                                        vector<double>(m1[0].size(), 0.));
  for (int j = 0; j < m1[0].size(); j++) {
    double sum = 0.;
    for (int i = 0; i < m1.size(); i++) {
      softmax_result[i][j] = exp(m1[i][j]);
      sum += softmax_result[i][j];
    }
    for (int i = 0; i < m1.size(); i++) {
      softmax_result[i][j] /= sum;
    }
  }
  return softmax_result;
}

inline void recurse_softmax_forward(shared_ptr<Tensor> t3,
                                    const shared_ptr<Tensor> t1,
                                    vector<int> &new_position, int axis = 0) {

  if (axis == t3->shape.size() - 2) {
    vector<vector<double>> m1 = t1->get_matrix(new_position);
    vector<vector<double>> softmax_result = evaluate_softmax(m1, t1->logger);
    t3->set_matrix(new_position, softmax_result);
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < t3->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_softmax_forward(t3, t1, new_position, axis + 1);
  }
  new_position.pop_back();
}

inline shared_ptr<Tensor> softmax_forward(shared_ptr<Tensor> t1) {
  shared_ptr<Logger> logger = t1->logger;
  shared_ptr<Tensor> t3 =
      make_shared<Tensor>(vector<double>(t1->values.size(), 0.), t1->shape,
                          logger, t1, nullptr, softmax_backward);
  vector<int> new_position{};
  recurse_softmax_forward(t3, t1, new_position);
  return t3;
}

// Loss functions

inline void
recurse_cross_entropy_backward(shared_ptr<Tensor> predicted,
                               const shared_ptr<Tensor> ground_truth,
                               vector<int> &new_position, int axis = 0) {

  if (axis == predicted->shape.size() - 2) {
    vector<vector<double>> predicted_values =
        predicted->get_matrix(new_position);
    vector<vector<double>> ground_truth_values =
        ground_truth->get_matrix(new_position);

    vector<vector<double>> loss_gradient(
        predicted_values.size(),
        vector<double>(predicted_values[0].size(), 0.));

    for (int j = 0; j < predicted_values[0].size(); j++) {
      for (int i = 0; i < predicted_values.size(); i++) {
        loss_gradient[i][j] =
            -1.0 * ground_truth_values[i][j] / predicted_values[i][j];
      }
    }
    predicted->set_matrix(new_position, loss_gradient, "gradients");
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < predicted->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_cross_entropy_backward(predicted, ground_truth, new_position,
                                   axis + 1);
  }
  new_position.pop_back();
}

inline void cross_entropy_backward(shared_ptr<Tensor> loss) {

  shared_ptr<Tensor> predicted = loss->input_first;
  if (!predicted)
    return;
  shared_ptr<Tensor> ground_truth = loss->input_second;
  vector<int> new_position{};
  recurse_cross_entropy_backward(predicted, ground_truth, new_position);
}

inline vector<double>
evaluate_cross_entropy(vector<vector<double>> predicted,
                       vector<vector<double>> ground_truth,
                       shared_ptr<Logger> logger) {
  if (predicted.size() != ground_truth.size() ||
      predicted[0].size() != ground_truth[0].size()) {
    logger->log(ERROR, "Matrix dimensions do not match in "
                       "evaluate_cross_entropy.");
    exit(1);
  }
  vector<double> loss(predicted[0].size(), 0.);
  for (int j = 0; j < predicted[0].size(); j++) {
    double cross_entropy = 0.;
    for (int i = 0; i < predicted.size(); i++) {
      cross_entropy += ground_truth[i][j] * log(predicted[i][j]);
    }
    loss[j] = -cross_entropy;
  }
  return loss;
}

inline void recurse_cross_entropy_forward(shared_ptr<Tensor> loss,
                                          const shared_ptr<Tensor> predicted,
                                          const shared_ptr<Tensor> ground_truth,
                                          vector<int> &new_position,
                                          int axis = 0) {

  if (axis == loss->shape.size() - 2) {
    vector<vector<double>> m1 = predicted->get_matrix(new_position);
    vector<vector<double>> m2 = ground_truth->get_matrix(new_position);
    vector<double> cross_entropy_loss =
        evaluate_cross_entropy(m1, m2, predicted->logger);
    vector<vector<double>> result{cross_entropy_loss};
    loss->set_matrix(new_position, result, "values");
    return;
  }

  new_position.push_back(0);
  int position_index = new_position.size() - 1;

  for (int i = 0; i < loss->shape[axis]; i++) {
    new_position[position_index] = i;
    recurse_cross_entropy_forward(loss, predicted, ground_truth, new_position,
                                  axis + 1);
  }
  new_position.pop_back();
}

inline shared_ptr<Tensor>
categorical_cross_entropy_forward(shared_ptr<Tensor> predicted,
                                  shared_ptr<Tensor> ground_truth) {
  shared_ptr<Logger> logger = predicted->logger;
  if (predicted->shape != ground_truth->shape) {
    logger->log(ERROR,
                "The shapes of the predicted and ground truth arrays are "
                "mismatched in categorical cross entropy.");
    exit(1);
  }

  vector<int> loss_shape = predicted->shape;
  int last_index = loss_shape.size() - 1;
  loss_shape[last_index - 1] = 1;

  int number_elements = 1;
  for (int i = 0; i < loss_shape.size(); i++) {
    number_elements *= loss_shape[i];
  }

  vector<double> loss_values(number_elements, 0.);
  shared_ptr<Tensor> loss =
      make_shared<Tensor>(loss_values, loss_shape, logger, predicted,
                          ground_truth, cross_entropy_backward);
  vector<int> new_position{};
  recurse_cross_entropy_forward(loss, predicted, ground_truth, new_position);
  return loss;
}

inline void mean_squared_error_backward(shared_ptr<Tensor> loss) {
  shared_ptr<Tensor> predicted = loss->input_first;
  shared_ptr<Tensor> ground_truth = loss->input_second;

  for (int i = 0; i < predicted->values.size(); i++) {
    predicted->gradients[i] +=
        2.0 * (predicted->values[i] - ground_truth->values[i]);
  }
}

inline shared_ptr<Tensor>
mean_squared_error_forward(shared_ptr<Tensor> predicted,
                           shared_ptr<Tensor> ground_truth) {
  shared_ptr<Logger> logger = predicted->logger;
  if (predicted->shape != ground_truth->shape) {
    logger->log(ERROR,
                "The shapes of the predicted and ground truth arrays are "
                "mismatched in mean_squared_error_forward.");
    exit(1);
  }

  vector<int> loss_shape = predicted->shape;

  int number_elements = 1;
  for (int i = 0; i < loss_shape.size(); i++) {
    number_elements *= loss_shape[i];
  }

  vector<double> loss_values(number_elements, 0.);

  for (int i = 0; i < number_elements; i++)
    loss_values[i] = (predicted->values[i] - ground_truth->values[i]) *
                     (predicted->values[i] - ground_truth->values[i]);

  shared_ptr<Tensor> loss =
      make_shared<Tensor>(loss_values, loss_shape, logger, predicted,
                          ground_truth, mean_squared_error_backward);
  return loss;
}

// Convolution operations
/**
 * Function to flip kernel in-place
 * @param kernel: Tensor with shape (num_filters, kernel_height,
 * kernel_width, channels).
 */
void flip_kernel(shared_ptr<Tensor> kernel) {
  vector<int> kernel_shape = kernel->shape;
  int height = kernel_shape[1];
  int width = kernel_shape[2];
  for (int f = 0; f < kernel_shape[0]; f++) {
    for (int c = 0; c < kernel_shape[3]; c++) {
      for (int j = 0; j < kernel_shape[1]; j++) {
        for (int i = 0; i < kernel_shape[2]; i++) {
          if ((width - 1) - i < i)
            kernel->swap_elements(
                vector<int>{f, j, i, c},
                vector<int>{f, (height - 1) - j, (width - 1) - i, c});
        }
      }
    }
  }
}

inline shared_ptr<Tensor> convolution(shared_ptr<Tensor> input,
                                      shared_ptr<Tensor> kernel,
                                      shared_ptr<Tensor> bias, int stride = 1,
                                      int padding = 0, int dilation_input = 1,
                                      int dilation_kernel = 1);
/**
 * Function to perform backward pass of convolution operation.
 * It is assumed that the value of dilation_input used in the forward pass is 1.
 * @param convolution_result: Tensor with shape (batch_size,
 * height_output, width_output, num_filters).
 * @param stride: Stride of the convolution for forward pass.
 * @param padding: Padding used for forward pass.
 * @param dilation_kernel: Dilation used for kernel in forward pass.
 */
inline void convolution_backward(shared_ptr<Tensor> convolution_result,
                                 int stride = 1, int padding = 0,
                                 int dilation_kernel = 1) {

  vector<double> bias_values(convolution_result->shape[3], 0.);
  shared_ptr<Tensor> bias_tensor = make_shared<Tensor>(
      bias_values, vector<int>{1, 1}, convolution_result->logger);

  shared_ptr<Logger> logger = convolution_result->logger;
  shared_ptr<Tensor> gradient_tensor = make_shared<Tensor>(
      convolution_result->gradients, convolution_result->shape,
      convolution_result->logger);
  shared_ptr<Tensor> convolution_input = convolution_result->input_first;
  shared_ptr<Tensor> convolution_kernel = convolution_result->input_second;
  int dilated_kernel_width =
      dilation_kernel * (convolution_kernel->shape[1] - 1) + 1;

  // Assuming that convolution kernel used for forward pass is square i.e.
  // height == width
  // Need to perform separate convolutions for each combination of input and
  // output channels. Each output channel corresponds to a filter in the
  // convolution layer.
  flip_kernel(convolution_kernel);
  for (int c = 0; c < convolution_kernel->shape[3]; c++) {
    vector<double> gradient_channel_values(convolution_result->shape[0] *
                                               convolution_input->shape[1] *
                                               convolution_input->shape[2],
                                           0.);
    shared_ptr<Tensor> gradient_tensor_channel = make_shared<Tensor>(
        gradient_channel_values,
        vector<int>{convolution_result->shape[0], convolution_input->shape[1],
                    convolution_input->shape[2], 1},
        convolution_result->logger);
    for (int f = 0; f < convolution_kernel->shape[0]; f++) {
      // Get the kernel for the current filter and channel
      vector<int> tmp{0, 0};
      vector<vector<int>> kernel_index(4, tmp);
      kernel_index[0][0] = f;
      kernel_index[0][1] = f;
      kernel_index[1][1] = convolution_kernel->shape[1] - 1;
      kernel_index[2][1] = convolution_kernel->shape[2] - 1;
      kernel_index[3][0] = c;
      kernel_index[3][1] = c;
      vector<double> kernel_values{};
      convolution_kernel->get_subtensor(kernel_index, 0, 0, kernel_values);

      shared_ptr<Tensor> kernel_tensor =
          make_shared<Tensor>(kernel_values,
                              vector<int>{1, convolution_kernel->shape[1],
                                          convolution_kernel->shape[2], 1},
                              convolution_kernel->logger);
      vector<vector<int>> gradient_index(4, tmp);
      gradient_index[0][1] = convolution_result->shape[0] - 1;
      gradient_index[1][1] = convolution_result->shape[1] - 1;
      gradient_index[2][1] = convolution_result->shape[2] - 1;
      gradient_index[3][0] = f;
      gradient_index[3][1] = f;
      vector<double> gradient_values{};
      gradient_tensor->get_subtensor(gradient_index, 0, 0, gradient_values);
      shared_ptr<Tensor> gradient_subtensor =
          make_shared<Tensor>(gradient_values,
                              vector<int>{convolution_result->shape[0],
                                          convolution_result->shape[1],
                                          convolution_result->shape[2], 1},
                              convolution_result->logger);
      // Perform convolution with the kernel and the gradient tensor
      // shape of convolution_channel_filter is (batch_size, height, width, 1)
      shared_ptr<Tensor> input_gradients_channel_filter =
          convolution(gradient_subtensor, kernel_tensor, bias_tensor, 1,
                      dilated_kernel_width - 1, stride, dilation_kernel);

      for (int b = 0; b < convolution_input->shape[0]; b++) {
        for (int h = 0; h < convolution_input->shape[1]; h++) {
          for (int w = 0; w < convolution_input->shape[2]; w++) {
            convolution_input->set_element(
                vector<int>{b, h, w, c},
                input_gradients_channel_filter->get_element(
                    vector<int>{b, h + padding, w + padding, 0}),
                "gradients");
          }
        }
      }
    }
  }

  flip_kernel(convolution_kernel);

  for (int f = 0; f < convolution_kernel->shape[0]; f++) {
    for (int c = 0; c < convolution_kernel->shape[3]; c++) {
      for (int b = 0; b < convolution_input->shape[0]; b++) {
        vector<int> tmp{0, 0};
        vector<vector<int>> gradient_index(4, tmp);
        gradient_index[0][0] = b;
        gradient_index[0][1] = b;
        gradient_index[1][1] = gradient_tensor->shape[1] - 1;
        gradient_index[2][1] = gradient_tensor->shape[2] - 1;
        gradient_index[3][0] = f;
        gradient_index[3][1] = f;
        vector<double> gradient_values{};
        gradient_tensor->get_subtensor(gradient_index, 0, 0, gradient_values);
        shared_ptr<Tensor> gradient_subtensor =
            make_shared<Tensor>(gradient_values,
                                vector<int>{1, gradient_tensor->shape[1],
                                            gradient_tensor->shape[2], 1},
                                convolution_result->logger);
        vector<vector<int>> convolution_input_index(4, tmp);
        convolution_input_index[0][0] = b;
        convolution_input_index[0][1] = b;
        convolution_input_index[1][1] = convolution_input->shape[1] - 1;
        convolution_input_index[2][1] = convolution_input->shape[2] - 1;
        convolution_input_index[3][0] = c;
        convolution_input_index[3][1] = c;

        vector<double> convolution_input_values{};
        convolution_input->get_subtensor(convolution_input_index, 0, 0,
                                         convolution_input_values);
        shared_ptr<Tensor> convolution_input_subtensor =
            make_shared<Tensor>(convolution_input_values,
                                vector<int>{1, convolution_input->shape[1],
                                            convolution_input->shape[2], 1},
                                convolution_result->logger);

        // Perform convolution with the kernel and the gradient tensor
        shared_ptr<Tensor> kernel_filter_gradients =
            convolution(convolution_input_subtensor, gradient_subtensor,
                        bias_tensor, 1, padding, 1, stride);

        for (int j = 0; j < kernel_filter_gradients->shape[1];
             j += dilation_kernel) {
          for (int i = 0; i < kernel_filter_gradients->shape[2];
               i += dilation_kernel) {
            convolution_kernel->set_element(
                vector<int>{f, j / dilation_kernel, i / dilation_kernel, c},
                kernel_filter_gradients->get_element(vector<int>{0, j, i, 0}),
                "gradients");
          }
        }
      }
    }
  }
}

/**
 * Function to get values at a specific index in the input tensor
 * @param batch: Batch index
 * @param width_start: Starting width index
 * @param height_start: Starting height index
 * @param input: Input tensor
 * @param dilated_kernel_height: Height of the dilated kernel
 * @param dilated_kernel_width: Width of the dilated kernel
 * @param padding: Padding applied to the input tensor
 * @param dilation_input: Dilation factor for the input tensor
 * @return values: Vector of values from the input tensor. First index is
 * set to batch. Height and width begin at height_start and width_start
 * respectively.
 */
inline vector<double>
get_values_at_index(int batch, int width_start, int height_start,
                    shared_ptr<Tensor> input, int dilated_kernel_height,
                    int dilated_kernel_width, int padding, int dilation_input) {
  vector<double> values{};
  int height_input = input->shape[1];
  int width_input = input->shape[2];
  shared_ptr<Logger> logger = input->logger;

  for (int l = height_start; l < height_start + dilated_kernel_height; l++) {
    for (int k = width_start; k < width_start + dilated_kernel_width; k++) {
      for (int p = 0; p < input->shape[3]; p++) {
        if (k < padding || l < padding ||
            k > padding + (width_input - 1) * dilation_input ||
            l > padding + (height_input - 1) * dilation_input)
          values.push_back(0);

        else if ((l - padding) % dilation_input != 0 ||
                 (k - padding) % dilation_input != 0)
          values.push_back(0);
        else {
          int i = (l - padding) / dilation_input;
          int j = (k - padding) / dilation_input;
          vector<int> required_index = {batch, i, j, p};
          values.push_back(input->get_element(required_index));
        }
      }
    }
  }

  return values;
}

/**
 * Function to perform convolution
 * @param input: Tensor with shape (batch_size, height, width, channels)
 * @param kernel: Tensor with shape (number_filters, kernel_height,
 * kernel_width, channels). Only includes weights but not bias.
 * @param bias: Tensor with shape (1, number_filters). Bias to be added
 * to result of convolution.
 * @param stride: Stride of the convolution.
 * @param padding: Padding of the convolution. All padding grid cells have value
 * 0.
 * @param dilation: Dilation of the convolution.
 * @return conv_result: Tensor with shape (batch_size, (height -
 * kernel_height)/stride + 1, (width - kernel_width)/stride + 1,
 * number_filters)
 */
inline shared_ptr<Tensor> convolution(shared_ptr<Tensor> input,
                                      shared_ptr<Tensor> kernel,
                                      shared_ptr<Tensor> bias, int stride,
                                      int padding, int dilation_input,
                                      int dilation_kernel) {
  shared_ptr<Logger> logger = input->logger;

  // Update input to account for padding and dilation
  int number_filters = kernel->shape[0];
  int kernel_height = kernel->shape[1];
  int kernel_width = kernel->shape[2];
  int channels = input->shape[3];
  int batch_size = input->shape[0];
  int height_input = input->shape[1];
  int width_input = input->shape[2];
  int width_effective = 1 + (width_input - 1) * dilation_input + 2 * padding;
  int height_effective = 1 + (height_input - 1) * dilation_input + 2 * padding;
  int dilated_kernel_height = 1 + dilation_kernel * (kernel_height - 1);
  int dilated_kernel_width = 1 + dilation_kernel * (kernel_width - 1);
  int width_output = 1 + (width_effective - dilated_kernel_width) / stride;
  int height_output = 1 + (height_effective - dilated_kernel_height) / stride;

  vector<double> data_values(batch_size * height_output * width_output *
                                 dilated_kernel_height * dilated_kernel_width *
                                 channels,
                             0.);
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < height_output; i++) {
      for (int j = 0; j < width_output; j++) {
        vector<double> current_values = get_values_at_index(
            b, j * stride, i * stride, input, dilated_kernel_height,
            dilated_kernel_width, padding, dilation_input);
        int offset =
            b * height_output * width_output * dilated_kernel_height *
                dilated_kernel_width * channels +
            i * width_output * dilated_kernel_height * dilated_kernel_width *
                channels +
            j * dilated_kernel_height * dilated_kernel_width * channels;
        for (int k = 0; k < current_values.size(); k++) {
          data_values[offset + k] = current_values[k];
        }
      }
    }
  }

  vector<int> data_shape{batch_size * height_output * width_output,
                         dilated_kernel_height * dilated_kernel_width *
                             channels};
  shared_ptr<Tensor> data =
      make_shared<Tensor>(data_values, data_shape, logger);

  vector<double> weights_values(number_filters * dilated_kernel_height *
                                    dilated_kernel_width * channels,
                                0.);
  vector<int> weights_shape{
      dilated_kernel_height * dilated_kernel_width * channels,
      number_filters,
  };

  for (int j = 0; j < dilated_kernel_height; j++) {
    for (int i = 0; i < dilated_kernel_width; i++) {
      for (int c = 0; c < channels; c++) {
        for (int f = 0; f < number_filters; f++) {
          int offset = j * dilated_kernel_width * channels * number_filters +
                       i * channels * number_filters + c * number_filters + f;
          if (j % dilation_kernel != 0 || i % dilation_kernel != 0)
            weights_values[offset] = 0;
          else {
            int j_ = j / dilation_kernel;
            int i_ = i / dilation_kernel;
            weights_values[offset] = kernel->get_element({f, j_, i_, c});
          }
        }
      }
    }
  }

  shared_ptr<Tensor> weights =
      make_shared<Tensor>(weights_values, weights_shape, logger);

  // Define the following symbols:
  // b: batch size
  // h_r: height of convolution result
  // w_r: width of convolution result
  // nf: number of filters
  // Shape of convolution_result: (b*h_r*w_r, nf)
  shared_ptr<Tensor> convolution_result = batch_matmul_forward(data, weights);

  convolution_result->reshape(
      {batch_size, height_output, width_output, number_filters});
  convolution_result->input_first = input;
  convolution_result->input_second = kernel;
  auto conv_back = [stride, padding, dilation_kernel](shared_ptr<Tensor> t3) {
    convolution_backward(t3, stride, padding, dilation_kernel);
  };
  convolution_result->backward_function = conv_back;
  shared_ptr<Tensor> convolution_result_bias =
      add_tensor_forward(convolution_result, bias);
  return convolution_result_bias;
}
/**
 * Function to perform maximum pooling operation
 * @param input: Tensor with shape (batch_size, height, width, channels)
 * @param kernel_height: Height of the pooling kernel.
 * @param kernel_width: Width of the pooling kernel.
 * @param stride: Controls number of pixels to move between successive pooling
 * operations.
 * @param padding: Padding for the input tensor. All padding grid cells have
 * value negative infinity.
 * @param dilation_kernel: Dilation of the kernel.
 * @return max_pool_result: Tensor with shape (batch_size, (height -
 * kernel_height)/stride + 1, (width - kernel_width)/stride + 1,
 * channels)
 */
inline shared_ptr<Tensor> max_pool(shared_ptr<Tensor> input, int kernel_height,
                                   int kernel_width, int stride, int padding,
                                   int dilation_kernel) {
  shared_ptr<Logger> logger = input->logger;

  // Update input to account for padding and dilation
  int batch_size = input->shape[0];
  int height_input = input->shape[1];
  int width_input = input->shape[2];
  int channels = input->shape[3];
  int width_effective = width_input + 2 * padding;
  int height_effective = height_input + 2 * padding;
  int dilated_kernel_height = 1 + dilation_kernel * (kernel_height - 1);
  int dilated_kernel_width = 1 + dilation_kernel * (kernel_width - 1);
  int width_output = 1 + (width_effective - dilated_kernel_width) / stride;
  int height_output = 1 + (height_effective - dilated_kernel_height) / stride;

  vector<double> result_values(batch_size * height_output * width_output *
                                   dilated_kernel_height *
                                   dilated_kernel_width * channels,
                               0.);
  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < channels; c++) {
      for (int j = 0; j < height_output; j++) {
        for (int i = 0; i < width_output; i++) {
          int row_start = j * stride;
          int col_start = i * stride;
          int offset = b * height_output * width_output * channels +
                       j * width_output * channels + i * channels + c;
          double max_value = -std::numeric_limits<double>::infinity();
          for (int l = row_start; l < row_start + dilated_kernel_height; l++) {
            for (int k = col_start; k < col_start + dilated_kernel_width; k++) {
              max_value = max(max_value, input->get_element({b, l, k, 0}));
            }
          }
          result_values[offset] = max_value;
        }
      }
    }
  }

  vector<int> data_shape{batch_size * height_output * width_output,
                         dilated_kernel_height * dilated_kernel_width *
                             channels};
  return input;
}
// shared_ptr<Tensor> data =
//     make_shared<Tensor>(data_values, data_shape, logger);

// vector<double> weights_values(number_filters * dilated_kernel_height *
//                                   dilated_kernel_width * channels,
//                               0.);
// vector<int> weights_shape{
//     dilated_kernel_height * dilated_kernel_width * channels,
//     number_filters,
// };

// for (int j = 0; j < dilated_kernel_height; j++) {
//   for (int i = 0; i < dilated_kernel_width; i++) {
//     for (int c = 0; c < channels; c++) {
//       for (int f = 0; f < number_filters; f++) {
//         int offset = j * dilated_kernel_width * channels * number_filters +
//                      i * channels * number_filters + c * number_filters +
//                      f;
//         if (j % dilation_kernel != 0 || i % dilation_kernel != 0)
//           weights_values[offset] = 0;
//         else {
//           int j_ = j / dilation_kernel;
//           int i_ = i / dilation_kernel;
//           weights_values[offset] = kernel->get_element({f, j_, i_, c});
//         }
//       }
//     }
//   }
// }

// shared_ptr<Tensor> weights =
//     make_shared<Tensor>(weights_values, weights_shape, logger);

// // Define the following symbols:
// // b: batch size
// // h_r: height of convolution result
// // w_r: width of convolution result
// // nf: number of filters
// // Shape of convolution_result: (b*h_r*w_r, nf)
// shared_ptr<Tensor> convolution_result = batch_matmul_forward(data,
// weights);

// convolution_result->reshape(
//     {batch_size, height_output, width_output, number_filters});
// convolution_result->input_first = input;
// convolution_result->input_second = kernel;
// auto conv_back = [stride](shared_ptr<Tensor> t3) {
//   convolution_backward(t3, stride);
// };
// convolution_result->backward_function = conv_back;
// shared_ptr<Tensor> convolution_result_bias =
//     add_tensor_forward(convolution_result, bias);
// return convolution_result_bias;
// }
} // namespace ml
