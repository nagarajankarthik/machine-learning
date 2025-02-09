# Machine Learning Algorithms

This repository implements some commonly used machine learning algorithms. Currently, decision trees and random forests have been implemented.

A neural network module is currently being developed.


## Dependencies

The following dependencies need to be installed before the code can be run. 


- g++ compiler version 11.4.0
- cmake version 3.22.1
- Sphinx 2.2.11-id64-release (95ae9a6)
- doxygen version 1.9.1
- OpenMP version 4.5 
- python 3

The OpenMP dependency is used for multi-threading and is optional.

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

1. Specify the number of OpenMP threads. This step is only required if multi-threading is being used:

```
export OMP_NUM_THREADS=4
```

2. Set the desired input parameter values in a json file following the [example](https://github.com/nagarajankarthik/machine-learning/blob/main/src/input.json) on the github page.

3. Navigate to the `build` directory described in the previous section.

4. Run a command similar to the following:

```
src/ml [PATH_TO_JSON_INPUT]
```
