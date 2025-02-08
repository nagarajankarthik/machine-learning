# Implementation of Decision Tree 


The decision tree is grown using either depth-first or breadth-first search. When determining the optimal split for each node, either the gini or entropy methods can be used to calculate the node impurity. The algorithm always selects the split that produces the maximum reduction in impurity.

```{eval-rst}
.. doxygenclass:: ml::DecisionTreeClassifier
   :members:

```
