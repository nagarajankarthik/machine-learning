# Example input files for training machine learning models

This folder contains example input files for training machine learning models. Each file is a JSON file that contains the input parameters required for model training. The details of the file contents are as follows:

- `input_basic_example.json`: This file contains inputs reqired to train the decision tree, random forest and multi-layer perceptron models. To use it, one can download a dataset and specify the appropriate file path in the `data` field. The iris dataset, available from [Kaggle](https://www.kaggle.com/uciml/iris), was used for testing purposes. The following data pre-processing steps should be performed on the data file before using it for model training:

    1. Remove the header row containing the column names, if present.
    2. Remove the id column if present.
    3. Ensure that there are no missing values in the data.
    4. Encode any categorical variables using label encoding. The data file should only contain numeric values.
    5. The first line of the data file should have two integers, separated by a comma, specifying the number of features and target variables.

- `input_lenet_v1.json`: This file contains input parameters required to train the LeNet-5 model. The MNIST dataset, available from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv), was used for testing purposes. The requirements for the train and test data files are identical to the `input_basic_example.json` case. Additionally, the feature values need to be normalized to the range [0, 1]. This can be done by dividing each feature value by 255.
