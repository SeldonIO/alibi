# TrustScores

[\[source\]](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/api/alibi.confidence.html#alibi.confidence.TrustScore)

## Trust Scores

### Overview

It is important to know when a machine learning classifier's predictions can be trusted. Relying on the classifier's (uncalibrated) prediction probabilities is not optimal and can be improved upon. Enter _trust scores_. Trust scores measure the agreement between the classifier and a modified nearest neighbor classifier on the predicted instances. The trust score is the ratio between the distance of the instance to the nearest class different from the predicted class and the distance to the predicted class. A score of 1 would mean that the distance to the predicted class is the same as to the nearest other class. Higher scores correspond to more trustworthy predictions. The original paper on which the algorithm is based is called [To Trust Or Not To Trust A Classifier](https://arxiv.org/abs/1805.11783). Our implementation borrows heavily from and extends the authors' open source [code](https://github.com/google/TrustScore).

The method requires labeled training data to build [k-d trees](https://en.wikipedia.org/wiki/K-d_tree) for each prediction class. When the classifier makes predictions on a test instance, we measure the distance of the instance to each of the trees. The trust score is then calculated by taking the ratio of the smallest distance to any other class than the predicted class and the distance to the predicted class. The distance is measured to the $k$th nearest neighbor in each tree or by using the average distance from the first to the $k$th neighbor.

In order to filter out the impact of outliers in the training data, they can optionally be removed using 2 filtering techniques. The first technique builds a k-d tree for each class and removes a fraction $\alpha$ of the training instances with the largest k nearest neighbor (kNN) distance to the other instances in the class. The second fits a kNN-classifier to the training set, and removes a fraction $\alpha$ of the training instances with the highest prediction class disagreement. Be aware that the first method operates on the prediction class level while the second method runs on the whole training set. It is also important to keep in mind that kNN methods might not be suitable when there are significant scale differences between the input features.

Trust scores can for instance be used as a warning flag for machine learning predictions. If the score drops below a certain value and there is disagreement between the model probabilities and the trust score, the prediction can be explained using techniques like anchors or contrastive explanations.

Trust scores work best for low to medium dimensional feature spaces. When working with high dimensional observations like images, dimensionality reduction methods (e.g. auto-encoders or PCA) could be applied as a pre-processing step before computing the scores. This is demonstrated by the following example [notebook](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/trustscore_mnist.ipynb).

### Usage

#### Initialization and fit

At initialization, the optional filtering method used to remove outliers during the `fit` stage needs to be specified as well:

```python
from alibi.confidence import TrustScore

ts = TrustScore(alpha=.05,
                filter_type='distance_knn',
                k_filter=10,
                leaf_size=40,
                metric='euclidean',
                dist_filter_type='point')
```

All the **hyperparameters** are optional:

* `alpha`: target fraction of instances to filter out.
* `filter_type`: filter method; one of _None_ (no filtering), _distance\_knn_ (first technique discussed in _Overview_) or _probability\_knn_ (second technique).
* `k_filter`: number of neighbors used for the distance or probability based filtering method.
* `leaf_size`: affects the speed and memory usage to build the k-d trees. The memory scales with the ratio between the number of samples and the leaf size.
* `metric`: distance metric used for the k-d trees. _Euclidean_ by default.
* `dist_filter_type`: _point_ uses the distance to the $k$-nearest point while _mean_ uses the average distance from the 1st to the $k$th nearest point during filtering.

In this example, we use the _distance\_knn_ method to filter out 5% of the instances of each class with the largest distance to its 10th nearest neighbor in that class:

```python
ts.fit(X_train, y_train, classes=3)
```

* `classes`: equals the number of prediction classes.

_X\_train_ is the training set and _y\_train_ represents the training labels, either using one-hot encoding (OHE) or simple class labels.

#### Scores

The trust scores are simply calculated through the `score` method. `score` also returns the class labels of the closest not predicted class as a numpy array:

```python
score, closest_class = ts.score(X_test, 
                                y_pred, 
                                k=2,
                                dist_type='point')
```

_y\_pred_ can again be represented using both OHE or via class labels.

* `k`: $k$th nearest neighbor used to compute distance to for each class.
* `dist_type`: similar to the filtering step, we can compute the distance to each class either to the $k$-th nearest point (_point_) or by using the average distance from the 1st to the $k$th nearest point (_mean_).

### Examples

[Trust Scores applied to Iris](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/trustscore_iris.ipynb)

[Trust Scores applied to MNIST](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/trustscore_mnist.ipynb)
