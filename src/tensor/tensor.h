#include "../utils/logging.h"
#include <functional>
#include <list>
#include <memory>
#include <vector>
#pragma once
using namespace std;

namespace ml {
class Tensor : public enable_shared_from_this<Tensor> {

      public:
	/**
	 * Pointer to logger
	 */
	shared_ptr<Logger> logger = nullptr;

	/**
	 * Values contained in tensor
	 */
	vector<double> values = {};

	/**
	 * Gradients of loss with respect to tensor
	 */
	vector<double> gradients = {};

	/**
	 * Shape of Tensor
	 */
	vector<int> shape = {};

	/**
	 * Predecessor in computational graph
	 */
	shared_ptr<Tensor> input_first = nullptr;

	/**
	 * Predecessor in computational graph
	 */
	shared_ptr<Tensor> input_second = nullptr;

	/**
	 * Pointer to gradient function
	 */
	std::function<void(shared_ptr<Tensor>)> backward_function;

	/**
	 * Constructor
	 */
	Tensor(){};

	/**
	 * Constructor to assign values
	 */
	Tensor(vector<double> values, vector<int> shape,
	       shared_ptr<Logger> logger);
	;

	/**
	 * Constructor to assign values and inputs
	 */
	Tensor(vector<double> values, vector<int> shape,
	       shared_ptr<Logger> logger, shared_ptr<Tensor> input_first,
	       shared_ptr<Tensor> input_second);

	/**
	 * Constructor to assign values, inputs and backward function
	 */
	Tensor(vector<double> values, vector<int> shape,
	       shared_ptr<Logger> logger, shared_ptr<Tensor> input_first,
	       shared_ptr<Tensor> input_second,
	       function<void(shared_ptr<Tensor>)> backward_function);

	/**
	 * Destructor
	 */
	~Tensor(){};

	/**
	 * Function to support element indexing.
	 */
	double get_element(vector<int> position);

	/**
	 * Function to set values of individual elements.
	 */
	void set_element(vector<int> position, double new_value);

	/**
	 * Function to swap values of two elements.
	 */
	void swap_elements(vector<int> position_first,
			   vector<int> position_second);
	/**
	 * Function to retrieve matrix based on specified indices into
	 * batch (non-matrix) dimensions. All dimensions except for the last two
	 * are considered to be batch dimensions.
	 */
	vector<vector<double>> get_matrix(vector<int> position,
					  string item = "values") const;

	/**
	 * Function to set a matrix contained within the Tensor based on
	 * specified indices into batch (non-matrix) dimensions. All dimensions
	 * except for the last two are considered to be batch dimensions.
	 */
	void set_matrix(vector<int> position,
			const vector<vector<double>> &matrix,
			string item = "values");

	/**
	 * Reshape
	 */
	void reshape(vector<int> new_shape);

	/**
	 * Back-propagate gradients to input Tensors
	 */
	void backward();

      private:
	/**
	 * Used to support multi-dimensional indexing
	 */
	vector<int> strides = {};

	/**
	 * Update strides whenever shape of Tensor changes
	 */
	void update_strides();

	/**
	 * Update indices to account for broadcasting
	 */
	vector<int> broadcast_indices(vector<int> position) const;
};
} // namespace ml
