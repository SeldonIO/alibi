# LinearityMeasure

[\[source\]](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/api/alibi.confidence.html#alibi.confidence.LinearityMeasure)

## Measuring the linearity of machine learning models

### Overview

Machine learning models include in general linear and non-linear operations: neural networks may include several layers consisting of linear algebra operations followed by non-linear activation functions, while models based on decision trees are by nature highly non-linear. The linearity measure function and class provide an operational definition for the amount of non-linearity of a map acting on vector spaces. Roughly speaking, the amount of non-linearity of the map is defined based on how much the output of the map applied to a linear superposition of input vectors differs from the linear superposition of the map's outputs for each individual vector. In the context of supervised learning, this definition is immediately applicable to machine learning models, which are fundamentally maps from a input vector space (the feature space) to an output vector space that may represent probabilities (for classification models) or actual values of quantities of interest (for regression models).

Given an input vector space $V$, an output vector space $W$ and a map $M: V \rightarrow W$, the amount of non-linearity of the map $M$ in a region $\beta$ of the input space $V$ and relative to some coefficients $\alpha(v)$ is defined as

$$
L_{\beta, \alpha}^{(M)} = \left\| \int_{\beta} \alpha(v) M(v) dv - M\left(\int_{\beta}\alpha(v)vdv \right) \right\|,
$$

where $v \in V$ and $|\cdot|$ denotes the norm of a vector. If we consider a finite number of vectors $N$, the amount of non-linearity can be defined as

$$
L_{\beta, \alpha}^{(M)} = \left\| \sum_{i} \alpha_{i} M(v_i) - M\left(\sum_i \alpha_i v_i \right) \right\|,
$$

where, with an abuse of notation, $\beta$ is no longer a continuous region in the input space but a collection of input vectors ${v\_i}$ and $\alpha$ is no longer a function but a collection of real coefficients ${\alpha\_i }$ with $i \in {1, ..., N}$. Note that the second expression may be interpreted as an approximation of the integral quantity defined in the first expression, where the vectors ${v\_i}$ are sampled uniformly in the region $\beta$.

### Application to machine learning models

In supervised learning, a model can be considered as a function $M$ mapping vectors from the input space (feature vectors) to vectors in the output space. The output space may represents probabilities in the case of a classification model or values of the target quantities in the case of a regression model. The definition of the linearity measure given above can be applied to the case of a regression model (either a single target regression or a multi target regression) in a straightforward way.

In case of a classifier, let us denote by $z$ the logits vector of the model such that the probabilities of the model $M$ are given by $\text{softmax}(z).$ Since the activation function of the last layer is usually highly non-linear, it is convenient to apply the definition of linearity given above to the logits vector $z.$ In the "white box" scenario, in which we have access to the internal architecture of the model, the vector $z$ is accessible and the amount of non-linearity can be calculated immediately. On the other hand, if the only accessible quantities are the output probabilities (the "black box" scenario), we need to invert the last layer's activation function in order to retrieve $z.$ In other words, that means defining a new map $M^\prime = f^{-1} \circ M(v)$ where $f$ is the activation function at the last layer and considering $L\_{\beta, \alpha}^{(M^\prime)}$ as a measure of the non-linearity of the model. The activation function of the last layer is usually a sigmoid function for binary classification tasks or a softmax function for multi-class classification. The inversion of the sigmoid function does not present any particular challenge, and the map $M^\prime$ can be written as

$$
M^\prime = -\log \circ \left(\frac{1-M(v)}{M(v)}\right).
$$

On the other hand, the softmax probabilities $p$ are defined in terms of the vector $z$ as $p\_j = e^{z\_j}/\sum\_j{e^{z\_j\}},$ where $z\_j$ are the components of $z$. The inverse of the softmax function is thus defined up to a constant $C$ which does not depend on $j$ but might depend on the input vector $v.$ The inverse map $M^\prime = \text{softmax}^{-1} \circ M(v)$ is then given by:

$$
M^\prime = \log \circ M(v) + C(v),
$$

where $C(v)$ is an arbitrary constant depending in general on the input vector $v.$

Since in the black box scenario it is not possible to assess the value of $C$, henceforth we will ignore it and define the amount of non-linearity of a machine learning model whose output is a probability distribution as

$$
L_{\beta, \alpha}^{(\log \circ M)} = \left\| \sum_{i}^N \alpha_{i} \log \circ M(v_i) - \log \circ M\left(\sum_i^N \alpha_i v_i \right)\right\|.
$$

It must be noted that the quantity above may in general be different from the "actual" amount of non-linearity of the model, i.e. the quantity calculated by accessing the activation vectors $z$ directly.

### Implementation

#### Sampling

The module implements two different methods for the sampling of vectors in a neighbourhood of the instance of interest $v.$

* The first sampling method `grid` consists of defining the region $\beta$ as a discrete lattice of a given size around the instance of interest, with the size defined in terms of the L1 distance in the lattice; the vectors are then sampled from the lattice according to a uniform distribution. The density and the size of the lattice are controlled by the resolution parameter `res` and the size parameter `epsilon`. This method is highly efficient and scalable from a computational point of view.
* The second sampling method `knn` consists of sampling from the same probability distribution the instance $v$ was drawn from; this method is implemented by simply selecting the $K$ nearest neighbours to $v$ from a training set, when this is available. The `knn` method imposes the constraint that the neighbourhood of $v$ must include only vectors from the training set, and as a consequence it will exclude out-of-distribution instances from the computation of linearity.

#### Pairwise vs global linearity

The module implements two different methods to associate a value of the linearity measure to $v.$

* The first method consists of measuring the `global` linearity in a region around $v.$ This means that we sample $N$ vectors ${v\_i}$ from a region $\beta$ of the input space around $v$ and apply

$$
L_{\beta, \alpha}^{(M)} = \left\| \sum_{i=1}^N \alpha_{i} M(v_i) - M\left(\sum_{i=1}^N \alpha_i v_i \right) \right\|,
$$

* The second method consists of measuring the `pairwise` linearity between the instance of interest and other vectors close to it, averaging over all such pairs. In other words, we sample $N$ vectors ${v\_i}$ from $\beta$ as in the global method, but in this case we calculate the amount of non-linearity $L\_{(v,v\_i),\alpha}$ for every pair of vectors $(v, v\_i)$ and average over all the pairs. Given two coefficients ${\alpha\_0, \alpha\_1}$ such that $\alpha\_0 + \alpha\_1 = 1,$ we can define the pairwise linearity measure relative to the instance of interest $v$ as

$$
L^{(M)} = \frac{1}{N} \sum_{i=0}^N \left\|\alpha_0 M(v) + \alpha_1 M(v_i) - M(\alpha_0 v + \alpha_1 v_i)\right\|.
$$

The two methods are slightly different from a conceptual point of view: the global linearity measure combines all $N$ vectors sampled in $\beta$ in a single superposition, and can be conceptually regarded as a direct approximation of the integral quantity. Thus, the quantity is strongly linked to the model behavior in the whole region $\beta.$ On the other hand, the pairwise linearity measure is an averaged quantity over pairs of superimposed vectors, with the instance of interest $v$ included in each pair. For that reason, it is conceptually more tied to the instance $v$ itself rather than the region $\beta$ around it.

### Usage

#### LinearityMeasure class

Given a `model` class with a `predict` method that return probabilities distribution in case of a classifier or numeric values in case of a regressor, the linearity measure $L$ around an instance of interest $X$ can be calculated using the class `LinearityMeasure` as follows:

```python
from alibi.confidence import LinearityMeasure

predict_fn = lambda x: model.predict(x)

lm = LinearityMeasure(method='grid', 
                      epsilon=0.04, 
                      nb_samples=10, 
                      res=100,
                      alphas=None, 
                      model_type='classifier', 
                      agg='pairwise',
                      verbose=False)
lm.fit(X_train)
L = lm.score(predict_fn, X)
```

Where `x_train` is the dataset the model was trained on. The `feature_range` is inferred form `x_train` in the `fit` step.

#### linearity\_measure function

Given a `model` class with a `predict` method that return probabilities distribution in case of a classifier or numeric values in case of a regressor, the linearity measure $L$ around an instance of interest $X$ can also be calculated using the `linearity_measure` function as follows:

```python
from alibi.confidence import linearity_measure
from alibi.confidence.model_linearity import infer_feature_range

predict_fn = lambda x: model.predict(x)

feature_range = infer_feature_range(X_train)
L = linearity_measure(predict_fn, 
                      X, 
                      feature_range=feature_range
                      method='grid', 
                      X_train=None, 
                      epsilon=0.04,
                      nb_samples=10, 
                      res=100, 
                      alphas=None, 
                      agg='global',
                      model_type='classifier')
```

Note that in this case the `feature_range` must be explicitly passed to the function and it is inferred beforehand.

### Examples

[Iris dataset](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/linearity_measure_iris.ipynb)

[Fashion MNIST dataset](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/linearity_measure_fashion_mnist.ipynb)
