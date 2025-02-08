# Machine Learning Algorithms

This repository implements some commonly used machine learning algorithms. Currently, random forests and neural networks have been implemented.


## Dependencies

The following dependencies need to be installed before the code can be run. 


- gcc compiler version 11.4.0
- cmake version 3.22.1
- Sphinx 2.2.11-id64-release (95ae9a6)
- doxygen version 1.9.1  
- python 3

The following python packages need to be installed from the [Python Package Index](https://pypi.org/):

- sphinx-rtd-theme
- myst-parser
- breathe 


The code has only been tested on Ubuntu 22.04.4 LTS on Windows Subsystem for Linux 2. 

## Installation Instructions

1. Install the necessary dependencies mentioned in the previous section.

2. Clone the repository from the [github page](https://github.com/nagarajankarthik/machine-learning/tree/main).
3. Create a build directory in the top level `machine-learning` folder

```
mkdir build
```
4. Enter the build directory and run the command ```cmake ..``.

5. Run ```make```.

## Procedure to run the code

1. Set the desired input parameter values in a json file following the [example](https://github.com/nagarajankarthik/machine-learning/blob/main/src/input.json) on the github page.

2. Navigate to the `build` directory described in the previous section.

3. Run a command similar to the following:

```
src/ml [PATH_TO_JSON_INPUT]
```
