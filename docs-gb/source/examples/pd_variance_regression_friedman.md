# Feature importance and feature interaction based on partial dependece variance

In this notebook example we will explain the global behavior of a regression model trained on a synthetic dataset. We will show how to compute the global feature attribution and the feature interactions for a given model.

We will follow the example from [Greenwell et al. (2018)](https://arxiv.org/pdf/1805.04755.pdf)[\[1\]](pd_variance_regression_friedman.md#References) on the Friedman's regression problem defined in [Friedman et al. (1991)](https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full)[\[2\]](pd_variance_regression_friedman.md#References) and [Breiman et al. (1996)](https://link.springer.com/article/10.1007/BF00058655)[\[3\]](pd_variance_regression_friedman.md#References).

```python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from alibi.explainers.pd_variance import PartialDependenceVariance, plot_pd_variance
```

### Friedman's regression problem

Friedman's regression problem introduced in [Friedman et al.](https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full)[\[2\]](pd_variance_regression_friedman.md#References) and [Breiman et al.](https://link.springer.com/article/10.1007/BF00058655)[\[3\]](pd_variance_regression_friedman.md#References) consists in predicting a target variable based on ten independent features sample from a Uniform(0, 1). Although the feature space consists of ten features, only the first five of them appear in the true model.

The relation between the input features and the response variables, $y$, is given by:

$$
y = 10 \sin(\pi x_1 x_2) + 20 (x_3 - 0.5)^2 + 10 x_4 + 5 x_5 + \epsilon
$$

where $x\_i$, for $i=1, ..., 10$ are the ten input features, and $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

In the following cell, we generate a dataset of 1000 examples which we split into 500 training examples and 500 testing examples. Similar to the paper setup, the simulated observation are generated using a $\sigma=1$.

```python
def generate_target(X: np.ndarray):
    """
    Generates the target/response variable for the Friedman's regression problem.
    
    Parameters
    ----------
    X
        A matrix realisations sample from a Uniform(0, 1). The size of the matrix is `N x 10`,
        where `N` is the number of data instances.
    
    Returns
    -------
    Response variable.
    """
    return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5)**2 \
        + 10 * X[:, 3] + 5 * X[:, 4] + np.random.randn(len(X))

np.random.seed(0)
X = np.random.rand(1000, 10)
y = generate_target(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
```

### Train MLP regressor

Similar with [Greenwell et al.](https://arxiv.org/pdf/1805.04755.pdf)[\[1\]](pd_variance_regression_friedman.md#References), we train a Muti-layer Perceptron (MLP) regressor and report its score on both the train and test split. For the purposes of this examples, we keep the default configuration for the `MLPRegressor`.

```python
# train MLP regressor on data
nn = MLPRegressor(max_iter=10000, random_state=0)
nn = nn.fit(X_train, y_train)

# compute score on train and test dataset
print(f"Train score: {nn.score(X_train, y_train):.3f}")
print(f"Test score: {nn.score(X_test, y_test):.3f}")
```

```
Train score: 0.968
Test score: 0.931
```

### Define explainer

Now that we have the prediction model, we can define the `PartialDependenceVariance` explainer to compute the feature importance and feature interactions.

Note that our explainer can work with any black-box model by providing the prediction function, which in our case will be `nn.predict`. Furthermore, we can specify the feature names and the target names to match our formulation through the parameters `feature_names` and `target_names`.

```python
# define explainer
explainer = PartialDependenceVariance(predictor=nn.predict,
                                      feature_names=[f'x{i}' for i in range(1, 11)],
                                      target_names=['y'],
                                      verbose=True)
```

### Feature importance

With our explainer initialized, we can compute the feature importance for all features through a simple call to the `explain` function. The arguments provided would be a reference dataset `X` which is usually the training dataset (i.e., `X_train` in our example) and setting `method='importance'`. Note that the `explain` function can receive many other arguments through which the user can specify explicitly the features to compute the feature importance for, the grid points (i.e., in our case `grid_resolution=50` to speed up the computation), etc. We refer the reader to our documentation page for further details.

```python
exp_importance = explainer.explain(X=X_train,
                                   method='importance',
                                   grid_resolution=50)
```

```
100%|██████████| 10/10 [00:00<00:00, 28.22it/s]
```

Once our explanation is computed, we can visualize the feature importance in two ways. `Alibi` implements an utility plotting function, `plot_pd_variance`, which helps the user quickly visualize the results.

The simplest way is to visualize the results through a horizontal bar plot. By default, the features are ordered in descending order by their importance from top to bottom. This can be achieved through as simple call to `plot_pd_variance` as follows:

```python
plot_pd_variance(exp=exp_importance,
                 features='all',
                 fig_kw={'figwidth': 7, 'figheight': 5});
```

![png](../../.gitbook/assets/pd_variance_regression_friedman_11_0.png)

We can see straight away that the explainer managed to identify that the first five features are the most salient (i.e., $x\_4, x\_2, x\_1, x\_3, x\_5$ in decreasing order of their importance).

As also recommended in the paper, the feature importance should be analyzed concomitantly with the Partial Dependence Plots (PDP) described in [Friedman et al. (2001)](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boostingmachine/10.1214/aos/1013203451.full)[\[4\]](pd_variance_regression_friedman.md#Reference) and [Molnar (2020)](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/\(https:/christophm.github.io/interpretable-ml-book/pdp.html\))[\[5\]](pd_variance_regression_friedman.md#References) based on which the importance has been calculated. Our utility function allows the user to visualize the PDPs by simply setting the parameter `summarise=False`. As before, the plots are sorted in descending order based on the corresponding feature importance.

```python
plot_pd_variance(exp=exp_importance,
                 features='all',
                 summarise=False,
                 n_cols=3,
                 fig_kw={'figwidth': 15, 'figheight': 20});
```

![png](../../.gitbook/assets/pd_variance_regression_friedman_13_0.png)

From the PDPs, we can observe that the explainer managed to identify correctly the effects of each feature: linear for $x\_4$ and $x\_5$, quadratic for $x\_3$, and sinusoidal for $x\_1$ and $x\_2$. The other variables show a relative flat main effect which according to the method's assumption means a low importance. Also, by inspecting the plots we can see that $x\_4$ main effect spans a range from 8 to somewhere around 19, which is probably one of the reasons why it got the largest importance.

### Feature interaction

As previously mentioned, the `PartialDependenceVariance` explainer is able to compute a measure of feature interaction. The call to the explainer follows the same API as above, just by simply calling the `explain` function with the parameter `method='interaction'`. By default, the explainer will compute a measure of interaction for all possible pairs of features. Note that this is quadratic in the number of features and is based on computing a two-ways partial dependence function for all pairs. Thus, this step might be more computationally demanding. Similar with the computation of the feature importance, the user has the liberty to provide the features pairs for which the feature interaction will be computed and control the computation complexity through the grid parameters.

```python
exp_interaction = explainer.explain(X=X_train,
                                    method='interaction',
                                    grid_resolution=30)
```

```
100%|██████████| 45/45 [00:34<00:00,  1.32it/s]
```

Once the explanation is computed, we can visualize the summary horizontal plot to identify the pairs of features that interact the most. Because the plot can grow very tall due to the quadratic number of feature pairs, we expose the `top_k` parameter to limit the plot to the `top_k` most important features provided through the `features` parameter. In our case we set `top_k=10` and since `features='all'`, the plot will display the 10 feature pairs that interact the most out of all feature pairs.

```python
# plot summary
plot_pd_variance(exp=exp_interaction,
                 features='all',    # considers plotting all features
                 top_k=10,          # plots only the top 10 features from all the `features`
                 fig_kw={'figwidth': 7, 'figheight': 5});
```

![png](../../.gitbook/assets/pd_variance_regression_friedman_18_0.png)

From the plot above we can observe that the explainer attributes non-zero interaction values to many pairs of features, but interaction between $x\_1$ and $x\_2$ is the one that stands out, being an order of magnitude higher than the rest. This is in fact the only pair of features that interact through the function $\sin(\pi x\_1 x\_2)$.

As before, if we would like to visualize more details, we can call again the `plot_pd_variance` with `summarise=False`. For each explained feature pair the function will plot the two-way PDP followed immediately by two conditional importance plots, when conditioning on one feature at a time. Note that the final interaction between the two features is computed as the average of the two conditional interactions (see titles of each subplot). For visualization purposes, we recommend using `n_cols=3`, such that each row will describe only the feature interaction between one pair. Similar as before, the plots are displayed in descending order based on their interaction.

```python
plot_pd_variance(exp=exp_interaction,
                 features='all',  # considers plotting all feature pairs
                 top_k=3,         # plots only the top 3 features pairs from all the `features`
                 summarise=False,
                 n_cols=3,
                 fig_kw={'figwidth': 12, 'figheight': 12});
```

![png](../../.gitbook/assets/pd_variance_regression_friedman_20_0.png)

We can observe that apart from the first plot corresponding to features $x\_1$ and $x\_2$, the other plots present an almost flat trend which can be an indication, but not a guarantee, of the absence of feature interaction. There exist functions for which the `PartialDependenceVariance` explainer does not capture the feature interaction. We refer the reader to the [method description](https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependenceVariance.html) for a more detailed example.

### References

[\[1\]](pd_variance_regression_friedman.md#source_1) Greenwell, Brandon M., Bradley C. Boehmke, and Andrew J. McCarthy. "A simple and effective model-based variable importance measure." arXiv preprint arXiv:1805.04755 (2018).

[\[2\]](pd_variance_regression_friedman.md#source_2) Friedman, Jerome H. "Multivariate adaptive regression splines." The annals of statistics 19.1 (1991): 1-67.

[\[3\]](pd_variance_regression_friedman.md#soruce_3) Breiman, Leo. "Bagging predictors." Machine learning 24.2 (1996): 123-140.

[\[4\]](pd_variance_regression_friedman.md#source_4) Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine." Annals of statistics (2001): 1189-1232.

[\[5\]](pd_variance_regression_friedman.md#source_5) Molnar, Christoph. Interpretable machine learning. Lulu. com, 2020.
