# CF

[\[source\]](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/api/alibi.explainers.html#alibi.explainers.Counterfactual)

## Counterfactual Instances

Note

To enable support for counterfactual Instances, you may need to run

```bash
pip install alibi[tensorflow]
```

### Overview

A counterfactual explanation of an outcome or a situation $Y$ takes the form "If $X$ had not occured, $Y$ would not have occured" ([Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/counterfactual.html)). In the context of a machine learning classifier $X$ would be an instance of interest and $Y$ would be the label predicted by the model. The task of finding a counterfactual explanation is then to find some $X^\prime$ that is in some way related to the original instance $X$ but leading to a different prediction $Y^\prime$. Reasoning in counterfactual terms is very natural for humans, e.g. asking what should have been done differently to achieve a different result. As a consequence counterfactual instances for machine learning predictions is a promising method for human-interpretable explanations.

The counterfactual method described here is the most basic way of defining the problem of finding such $X^\prime$. Our algorithm loosely follows Wachter et al. (2017): [Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR](https://arxiv.org/abs/1711.00399). For an extension to the basic method which provides ways of finding higher quality counterfactual instances $X^\prime$ in a quicker time, please refer to [Counterfactuals Guided by Prototypes](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFProto.ipynb).

We can reason that the most basic requirements for a counterfactual $X^\prime$ are as follows:

* The predicted class of $X^\prime$ is different from the predicted class of $X$
* The difference between $X$ and $X^\prime$ should be human-interpretable.

While the first condition is straight-forward, the second condition does not immediately lend itself to a condition as we need to first define "interpretability" in a mathematical sense. For this method we restrict ourselves to a particular definition by asserting that $X^\prime$ should be as close as possible to $X$ without violating the first condition. The main issue with this definition of "interpretability" is that the difference between $X^\prime$ and $X$ required to change the model prediciton might be so small as to be un-interpretable to the human eye in which case [we need a more sophisticated approach](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFProto.ipynb).

That being said, we can now cast the search for $X^\prime$ as a simple optimization problem with the following loss:

$$L = L_{\text{pred}} + \lambda L_{\text{dist}},$$

where the first loss term $L\_{\text{pred\}}$ guides the search towards points $X^\prime$ which would change the model prediction and the second term $\lambda L\_{\text{dist\}}$ ensures that $X^\prime$ is close to $X$. This form of loss has a single hyperparameter $\lambda$ weighing the contributions of the two competing terms.

The specific loss in our implementation is as follows:

$$L(X^\prime\vert X) = (f_t(X^\prime) - p_t)^2 + \lambda L_1(X^\prime, X).$$

Here $t$ is the desired target class for $X^\prime$ which can either be specified in advance or left up to the optimization algorithm to find, $p\_t$ is the target probability of this class (typically $p\_t=1$), $f\_t$ is the model prediction on class $t$ and $L\_1$ is the distance between the proposed counterfactual instance $X^\prime$ and the instance to be explained $X$. The use of the $L\_1$ distance should ensure that the $X^\prime$ is a sparse counterfactual - minimizing the number of features to be changed in order to change the prediction.

The optimal value of the hyperparameter $\lambda$ will vary from dataset to dataset and even within a dataset for each instance to be explained and the desired target class. As such it is difficult to set and we learn it as part of the optimization algorithm, i.e. we want to optimize

$$\min_{X^{\prime}}\max_{\lambda}L(X^\prime\vert X)$$

subject to

$$\vert f_t(X^\prime)-p_t\vert\leq\epsilon \text{ (counterfactual constraint)},$$

where $\epsilon$ is a tolerance parameter. In practice this is done in two steps, on the first pass we sweep a broad range of $\lambda$, e.g. $\lambda\in(10^{-1},\dots,10^{-10}$) to find lower and upper bounds $\lambda\_{\text{lb\}}, \lambda\_{\text{ub\}}$ where counterfactuals exist. Then we use bisection to find the maximum $\lambda\in\[\lambda\_{\text{lb\}}, \lambda\_{\text{ub\}}]$ such that the counterfactual constraint still holds. The result is a set of counterfactual instances $X^\prime$ with varying distance from the test instance $X$.

### Usage

#### Initialization

The counterfactual (CF) explainer method works on fully black-box models, meaning they can work with arbitrary functions that take arrays and return arrays. However, if the user has access to a full TensorFlow (TF) or Keras model, this can be passed in as well to take advantage of the automatic differentiation in TF to speed up the search. This section describes the initialization for a TF/Keras model, for fully black-box models refer to [numerical gradients](cf.md#Numerical-Gradients).

First we load the TF/Keras model:

```python
model = load_model('my_model.h5')
```

Then we can initialize the counterfactual object:

```python
shape = (1,) + x_train.shape[1:]
cf = Counterfactual(model, shape, distance_fn='l1', target_proba=1.0,
                    target_class='other', max_iter=1000, early_stop=50, lam_init=1e-1,
                    max_lam_steps=10, tol=0.05, learning_rate_init=0.1,
                    feature_range=(-1e10, 1e10), eps=0.01, init='identity',
                    decay=True, write_dir=None, debug=False)
```

Besides passing the model, we set a number of **hyperparameters** ...

... **general**:

* `shape`: shape of the instance to be explained, starting with batch dimension. Currently only single explanations are supported, so the batch dimension should be equal to 1.
* `feature_range`: global or feature-wise min and max values for the perturbed instance.
* `write_dir`: write directory for Tensorboard logging of the loss terms. It can be helpful when tuning the hyperparameters for your use case. It makes it easy to verify that e.g. not 1 loss term dominates the optimization, that the number of iterations is OK etc. You can access Tensorboard by running `tensorboard --logdir {write_dir}` in the terminal.
* `debug`: flag to enable/disable writing to Tensorboard.

... related to the **optimizer**:

* `max_iterations`: number of loss optimization steps for each value of $\lambda$; the multiplier of the distance loss term.
* `learning_rate_init`: initial learning rate, follows linear decay.
* `decay`: flag to disable learning rate decay if desired
* `early_stop`: early stopping criterion for the search. If no counterfactuals are found for this many steps or if this many counterfactuals are found in a row we change $\lambda$ accordingly and continue the search.
* `init`: how to initialize the search, currently only `"identity"` is supported meaning the search starts from the original instance.

... related to the **objective function**:

* `distance_fn`: distance function between the test instance $X$ and the proposed counterfactual $X^\prime$, currently only `"l1"` is supported.
* `target_proba`: desired target probability for the returned counterfactual instance. Defaults to `1.0`, but it could be useful to reduce it to allow a looser definition of a counterfactual instance.
* `tol`: the tolerance within the `target_proba`, this works in tandem with `target_proba` to specify a range of acceptable predicted probability values for the counterfactual.
* `target_class`: desired target class for the returned counterfactual instance. Can be either an integer denoting the specific class membership or the string `other` which will find a counterfactual instance whose predicted class is anything other than the class of the test instance.
* `lam_init`: initial value of the hyperparameter $\lambda$. This is set to a high value $\lambda=1e^{-1}$ and annealed during the search to find good bounds for $\lambda$ and for most applications should be fine to leave as default.
* `max_lam_steps`: the number of steps (outer loops) to search for with a different value of $\lambda$.

While the default values for the loss term coefficients worked well for the simple examples provided in the notebooks, it is recommended to test their robustness for your own applications.

Warning

Once a `Counterfactual` instance is initialized, the parameters of it are frozen even if creating a new instance. This is due to TensorFlow behaviour which holds on to some global state. In order to change parameters of the explainer in the same session (e.g. for explaining different models), you will need to reset the TensorFlow graph manually:

```python
import tensorflow as tf
tf.keras.backend.clear_session()
```

You may need to reload your model after this. Then you can create a new `Counterfactual` instance with new parameters.

#### Fit

The method is purely unsupervised so no fit method is necessary.

#### Explanation

We can now explain the instance $X$:

```python
explanation = cf.explain(X)
```

The `explain` method returns an `Explanation` object with the following attributes:

* _cf_: dictionary containing the counterfactual instance found with the smallest distance to the test instance, it has the following keys:
  * _X_: the counterfactual instance
  * _distance_: distance to the original instance
  * _lambda_: value of $\lambda$ corresponding to the counterfactual
  * _index_: the step in the search procedure when the counterfactual was found
  * _class_: predicted class of the counterfactual
  * _proba_: predicted class probabilities of the counterfactual
  * _loss_: counterfactual loss
* _orig\_class_: predicted class of original instance
* _orig\_proba_: predicted class probabilites of the original instance
* _all_: dictionary of all instances encountered during the search that satisfy the counterfactual constraint but have higher distance to the original instance than the returned counterfactual. This is organized by levels of $\lambda$, i.e. `explanation['all'][0]` will be a list of dictionaries corresponding to instances satisfying the counterfactual condition found in the first iteration over $\lambda$ during bisection.

#### Numerical Gradients

So far, the whole optimization problem could be defined within the TF graph, making automatic differentiation possible. It is however possible that we do not have access to the model architecture and weights, and are only provided with a `predict` function returning probabilities for each class. The counterfactual can then be initialized in the same way as before, but using a prediction function:

```python
# define model
model = load_model('mnist_cnn.h5')
predict_fn = lambda x: cnn.predict(x)
    
# initialize explainer
shape = (1,) + x_train.shape[1:]
cf = Counterfactual(predict_fn, shape, distance_fn='l1', target_proba=1.0,
                    target_class='other', max_iter=1000, early_stop=50, lam_init=1e-1,
                    max_lam_steps=10, tol=0.05, learning_rate_init=0.1,
                    feature_range=(-1e10, 1e10), eps=0.01, init
```

In this case, we need to evaluate the gradients of the loss function with respect to the input features $X$ numerically:

where $L\_\text{pred}$ is the predict function loss term, $p$ the predict function and $x$ the input features to optimize. There is now an additional hyperparameter to consider:

* `eps`: a float or an array of floats to define the perturbation size used to compute the numerical gradients of $^{\delta p}/\_{\delta X}$. If a single float, the same perturbation size is used for all features, if the array dimension is _(1 x nb of features)_, then a separate perturbation value can be used for each feature. For the Iris dataset, `eps` could look as follows:

```python
eps = np.array([[1e-2, 1e-2, 1e-2, 1e-2]])  # 4 features, also equivalent to eps=1e-2
```

### Examples

[Counterfactual instances on MNIST](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/cf_mnist.ipynb)
