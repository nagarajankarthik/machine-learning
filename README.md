# Machine Learning Algorithms in C++

# Overview

This repository implements some commonly used machine learning algorithms in C++. Currently, decision trees and random forests have been implemented. 

A neural network module is currently being developed.

The input parameters should be specified in a json file. A representative example is given below:

```
{
	"general": {
		"logfile": "logs.txt"
	},
	"models": [
		{
			"model": "decision_tree_classifier",
			"data": "/mnt/c/Users/65915/example_datasets/iris_processed.csv",
			"search_algorithm": "breadth",
			"impurity_method": "gini",
			"random_seed": 120
		},
		{
			"model": "random_forest_classifier",
			"data": "/mnt/c/Users/65915/example_datasets/iris_processed.csv",
			"search_algorithm": "breadth",
			"impurity_method": "gini",
			"number_trees": 5,
			"random_seed": 120,
			"max_feature_fraction": 1.0
		}
	]
}
```

The parameters that can be specified for each type of model are explained in the [online documentation](https://machine-learning-basic.readthedocs.io/en/latest/index.html).

The next few sections explain the algorithms used to implement each type of machine learning model.

# Decision Trees

The current implementation of decision trees is for classification problems only.

During the training process, the decision tree is grown using either depth-first or breadth-first search. When determining the optimal split for each node, either the gini or entropy methods can be used to calculate the node impurity. More details about these impurity calculation methods can be found [here](https://www.baeldung.com/cs/impurity-entropy-gini-index). The algorithm always selects the split that produces the maximum reduction in impurity.

When performing inference, the tree grown during the training phase is traversed until a leaf node is reached for each instance in the test dataset. The predicted class is the one that occurs most frequently in the leaf node that is reached. 

# Random Forests

The current implementation of random forests is for classification problems only.

During the training process, the random forest grows a number of decision trees using the procedure described above. The data used to grow each tree is obtained from the training data supplied to this module using [bootstrap sampling](https://www.geeksforgeeks.org/bootstrap-method/).

During inference, the prediction of each tree is obtained for each test instance. The class that is predicted by the most number of trees is returned as the predicted class for that test instance.
 
