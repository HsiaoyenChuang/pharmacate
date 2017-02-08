# 1. KNN Classification

Currently we're using k-nearest-neighbours (KNN) over support-vector-machine (SVM), which is the way implemented 
in the referenced paper. However, possible problems/mentions are:

 a. KNN is hard to scale on large data, complexity is O(N^2). Sample data has only 390 drugs, it's too tiny to prove 
    lots of results but this is the best I can collect from papers.

 b. SVM usually runs faster but result depends on kernel method (it's better called kernel criterion), which is used 
    to find classification boundary.

 c. Random-Forest and Neural Networks are also mentioned among papers that predicting side effects with drug labels,
    I don't see huge performance difference. But, regarding the restricted size of sample data of the papers, result 
    should finally be evaluated in production.

But these popular methods take only one line of code to be replaced:
```
>> forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=1)
>> multi_target = MultiOutputClassifier(forest, n_jobs=-1)

>> # this can be:
>> knn = sklearn.neighbours.KNeighborsClassifier(...)
>> multi_target = MultiOutputClassifier(knn)

>> # or
>> svm = sklearn.svm.LinearSVC(...)
>> multi_target = MultiOutputClassifier(svm)

# Reference:
#   http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification
```

# 2. Multi-Label Classification

```
>> from sklearn.datasets import make_classification
>> from sklearn.multioutput import MultiOutputClassifier
>> from sklearn.ensemble import RandomForestClassifier
>> from sklearn.utils import shuffle
>> import numpy as np
>> X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
>> y2 = shuffle(y1, random_state=1)
>> y3 = shuffle(y1, random_state=2)
>> Y = np.vstack((y1, y2, y3)).T
>> n_samples, n_features = X.shape # 10,100
>> n_outputs = Y.shape[1] # 3
>> n_classes = 3
>> forest = RandomForestClassifier(n_estimators=100, random_state=1)
>> multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
>> multi_target_forest.fit(X, Y).predict(X)
array([[2, 2, 0],
       [1, 2, 1],
       [2, 1, 0],
       [0, 0, 2],
       [0, 2, 1],
       [0, 0, 2],
       [1, 1, 0],
       [1, 1, 1],
       [0, 0, 2],
       [2, 0, 0]])

# Reference:
#   http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#   http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
```

# 3. Feature Selection

```
# Reference:
#   http://scikit-learn.org/stable/modules/feature_selection.html
```

# 4. MLKNN (method in paper)

MLKNN is simply a multi-output classifier using KNN algorithm. It can be altered by SVM, Random-Forest and
more. It's a one-by-one comaprison among all pairs of sample, with each sample is a distinct drug with same 
drug label features.

# 5. TODO

It was mentioned a new sentiment analyzer is needed to produce stable result.
So there're three todos clear for me (in order):

  a. predict side effects for drug with labels, turn it into drug risk score (working)
  b. predict similar drugs
  c. provide an alternative method to replace old sentiment analyzer
