#include "../utils/logging.h"
#include <functional>
#include <vector>
#include <list>
#include <memory>
using namespace std;

namespace ml {
	class Tensor {

		public:
			/**
			 * Pointer to logger
			 */
			shared_ptr<Logger> logger = nullptr;

			/**
			 * Values contained in tensor
			 */
			vector<double> values = {} ;

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
			shared_ptr<Tensor> input_first;

			/**
			 * Predecessor in computational graph
			 */
			shared_ptr<Tensor> input_second;

			/**
			 * Constructor
			 */
			Tensor() {} ;
			
			/**
			 * Constructor to assign values
			 */
			Tensor(vector<double> values, vector<int> shape, shared_ptr<Logger> logger); ;

			/**
			 * Constructor to assign values and inputs
			 */
			Tensor(vector<double> values, vector<int> shape, shared_ptr<Logger> logger, shared_ptr<Tensor> input_first, shared_ptr<Tensor> input_second);

			/**
			 * Constructor to assign values, inputs and backward function
			 */
			Tensor(vector<double> values, vector<int> shape, shared_ptr<Logger> logger, shared_ptr<Tensor> input_first, shared_ptr<Tensor> input_second, function<void(const vector<double>&, shared_ptr<Tensor>, shared_ptr<Tensor>)> backward_function);

			/**
			 * Destructor
			 */
			~Tensor() {} ;

			/**
			 * Function to support element indexing.
			 */
			double at(vector<int> position) ;

			/**
			 * Function to retrieve matrix based on specified indices into 
			 * batch (non-matrix) dimensions. All dimensions except for the last two 
			 * are considered to be batch dimensions.
			 */
			vector<vector<double>> get_matrix_at (vector<int> position) const ;

			/**
			 * Function to set a matrix contained within the Tensor based on specified
			 * indices into 
			 * batch (non-matrix) dimensions. All dimensions except for the last two 
			 * are considered to be batch dimensions.
			 */
			void set_matrix_at(vector<int> position, const vector<vector<double>>& matrix) ; 

			/**
			 * Reshape
			 */
			void reshape(vector<int> new_shape) ;

			/**
			 * Back-propagate gradients to input Tensors
			 */
			void backward(const vector<double> & seed) ;
		private:

			/**
			 * Used to support multi-dimensional indexing
			 */
			vector<int> strides = {};

			/**
			 * Update strides whenever shape of Tensor changes
			 */
			void update_strides() ;

			/**
			 * Pointer to gradient function
			 */
			std::function<void(const vector<double> &, shared_ptr<Tensor>, shared_ptr<Tensor>)> backward_function ;

	};
}// namespace ml


