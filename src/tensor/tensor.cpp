#include "tensor.h"

namespace ml {
Tensor::Tensor(vector<double> values, vector<int> shape,
               shared_ptr<Logger> logger)
    : values(values), shape(shape), logger(logger) {
  update_strides();
  gradients.resize(values.size());
  fill(gradients.begin(), gradients.end(), 0.);
};

Tensor::Tensor(vector<double> values, vector<int> shape,
               shared_ptr<Logger> logger, shared_ptr<Tensor> input_first,
               shared_ptr<Tensor> input_second)
    : Tensor(values, shape, logger) {

  this->input_first = input_first;
  this->input_second = input_second;
};

Tensor::Tensor(vector<double> values, vector<int> shape,
               shared_ptr<Logger> logger, shared_ptr<Tensor> input_first,
               shared_ptr<Tensor> input_second,
               function<void(shared_ptr<Tensor>)> backward_function)
    : Tensor(values, shape, logger, input_first, input_second) {
  this->backward_function = backward_function;
};

vector<int> Tensor::broadcast_indices(vector<int> position, int offset) const {
  list<int> position_list(position.begin(), position.end());
  while (position_list.size() > shape.size() - offset) {
    position_list.pop_front();
  }
  vector<int> new_position(position_list.begin(), position_list.end());

  if (new_position.size() != shape.size() - offset) {
    logger->log(ERROR, "The specified position has " +
                           to_string(new_position.size()) +
                           " indices, but the tensor has " +
                           to_string(shape.size()) + " dimensions");
    exit(1);
  }

  for (int i = 0; i < new_position.size(); i++) {
    if (new_position[i] >= shape[i]) {
      if (shape[i] == 1) {
        new_position[i] = 0;
      } else {
        logger->log(ERROR, " Invalid index in "
                           "broadcast_indices at dimension " +
                               to_string(i) + ".");
        logger->log(ERROR, " Attempted to access index " +
                               to_string(new_position[i]) +
                               " but the tensor has " + to_string(shape[i]) +
                               " entries.");
        exit(1);
      }
    }
  }

  return new_position;
}

double Tensor::get_element(vector<int> position, string item) {
  int index = 0;
  vector<int> new_position = broadcast_indices(position, 0);
  for (int i = 0; i < new_position.size(); i++) {
    index += strides.at(i) * new_position.at(i);
  }
  return item == "values" ? values.at(index) : gradients.at(index);
}

void Tensor::set_element(vector<int> position, double new_value, string item) {
  int index = 0;
  vector<int> new_position = broadcast_indices(position, 0);
  for (int i = 0; i < new_position.size(); i++) {
    index += strides[i] * new_position[i];
  }
  if (item == "values")
    values[index] = new_value;
  else if (item == "gradients")
    gradients[index] += new_value;
  else {
    logger->log(ERROR, "Invalid item type in set_element: " + item);
    exit(1);
  }
}

void Tensor::swap_elements(vector<int> position_first,
                           vector<int> position_second) {
  int index_first = 0;
  int index_second = 0;
  for (int i = 0; i < shape.size(); i++) {
    index_first += strides[i] * position_first[i];
    index_second += strides[i] * position_second[i];
  }
  swap(values[index_first], values[index_second]);
}

void Tensor::get_subtensor(vector<vector<int>> new_shape, int axis, int offset,
                           vector<double> &new_values) {

  int start = new_shape[axis][0];
  int end = new_shape[axis][1];
  if (axis == new_shape.size() - 1) {
    for (int i = start; i <= end; i++) {
      new_values.push_back(this->values[offset + i]);
    }
  } else {
    for (int i = start; i <= end; i++) {
      get_subtensor(new_shape, axis + 1, offset + i * strides[axis],
                    new_values);
    }
  }
}

vector<vector<double>> Tensor::get_matrix(vector<int> position,
                                          string item) const {
  if (position.size() < shape.size() - 2) {
    logger->log(ERROR, "The specified position has " +
                           to_string(position.size()) +
                           " indices, but the tensor has " +
                           to_string(shape.size()) + " dimensions");
    exit(1);
  }

  // allow for broadcasting
  vector<int> new_position = broadcast_indices(position, 2);

  int index = 0;
  for (int i = 0; i < new_position.size(); i++) {
    index += strides[i] * new_position[i];
  }
  int number_rows = shape[shape.size() - 2];
  int number_cols = shape[shape.size() - 1];
  vector<double> tmp(*shape.rbegin(), 0.);
  vector<vector<double>> matrix(*(++shape.rbegin()), tmp);

  if (item == "values") {
    for (int i = 0; i < number_rows; i++) {
      for (int j = 0; j < number_cols; j++) {
        matrix[i][j] = values[index + i * number_cols + j];
      }
    }
  } else if (item == "gradients") {
    for (int i = 0; i < number_rows; i++) {
      for (int j = 0; j < number_cols; j++) {
        matrix[i][j] = gradients[index + i * number_cols + j];
      }
    }
  }

  return matrix;
}

void Tensor::set_matrix(vector<int> position,
                        const vector<vector<double>> &matrix, string item) {
  if (position.size() < shape.size() - 2) {
    logger->log(ERROR, "The specified position has " +
                           to_string(position.size()) +
                           " indices, but the tensor has " +
                           to_string(shape.size()) + " dimensions");
    exit(1);
  }

  // allow for broadcasting
  vector<int> new_position = broadcast_indices(position, 2);

  int index = 0;
  for (int i = 0; i < new_position.size(); i++) {
    index += strides[i] * new_position[i];
  }
  int number_rows = shape[shape.size() - 2];
  int number_cols = shape[shape.size() - 1];

  if (matrix.size() != number_rows || matrix[0].size() != number_cols) {
    logger->log(ERROR,
                "The matrix specified has size " + to_string(matrix.size()) +
                    " by " + to_string(matrix[0].size()) +
                    " but each matrix comprising the tensor has size " +
                    to_string(number_rows) + " by " + to_string(number_cols));
    exit(1);
  }

  if (item == "values") {
    for (int i = 0; i < number_rows; i++) {
      for (int j = 0; j < number_cols; j++) {
        values[index + i * number_cols + j] = matrix[i][j];
      }
    }
  } else if (item == "gradients") {
    for (int i = 0; i < number_rows; i++) {
      for (int j = 0; j < number_cols; j++) {
        gradients[index + i * number_cols + j] += matrix[i][j];
      }
    }
  }
}

void Tensor::update_strides() {
  strides.clear();
  int product = 1;
  strides.push_back(product);
  for (auto it = shape.rbegin(); it != --shape.rend(); it++) {
    product *= *it;
    strides.push_back(product);
  }
  reverse(strides.begin(), strides.end());
}

void Tensor::reshape(vector<int> new_shape) {
  int new_size = 1;
  for (int i = 0; i < new_shape.size(); i++) {
    new_size *= new_shape[i];
  }
  if (new_size == values.size()) {
    shape = new_shape;
    update_strides();
  } else
    logger->log(ERROR, "Reshape error: new size does not match current size");
}

void Tensor::backward() {
  if (!input_first)
    return;
  backward_function(shared_from_this());
  input_first->backward();
  if (input_second)
    input_second->backward();
}

} // namespace ml
