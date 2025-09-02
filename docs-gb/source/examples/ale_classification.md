# Accumulated Local Effects for classifying flowers

In this example we will explain the behaviour of classification models on the Iris dataset. It is recommended to first read the [ALE regression example](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/ale_regression_california.ipynb) to familiarize yourself with how to interpret ALE plots in a simpler setting. Interpreting ALE plots for classification problems become more complex due to a few reasons:

* Instead of one ALE line for each feature we now have one for each class to explain the feature effects towards predicting each class.
* There are two ways to choose the prediction function to explain:
  * Class probability predictions (e.g. `clf.predict_proba` in `sklearn`)
  * Margin or logit predictions (e.g. `clf.decision_function` in `sklearn`)

We will see the implications of explaining each of these prediction functions.

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from alibi.explainers import ALE, plot_ale
```

## Load and prepare the dataset

```python
data = load_iris()
feature_names = data.feature_names
target_names = data.target_names
X = data.data
y = data.target
print(feature_names)
print(target_names)
```

```
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
['setosa' 'versicolor' 'virginica']
```

Shuffle the data and define the train and test set:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

## Fit and evaluate a logistic regression model

```python
lr = LogisticRegression(max_iter=200)
```

```python
lr.fit(X_train, y_train)
```

```
LogisticRegression(max_iter=200)
```

In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.\
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.LogisticRegression

```
LogisticRegression(max_iter=200)
```

```python
accuracy_score(y_test, lr.predict(X_test))
```

```
1.0
```

## Calculate Accumulated Local Effects

There are several options for explaining the classifier predictions using ALE. We define two prediction functions, one in the unnormalized logit space and the other in probability space, and look at how the resulting ALE plot interpretation changes.

```python
logit_fun_lr = lr.decision_function
proba_fun_lr = lr.predict_proba
```

```python
logit_ale_lr = ALE(logit_fun_lr, feature_names=feature_names, target_names=target_names)
proba_ale_lr = ALE(proba_fun_lr, feature_names=feature_names, target_names=target_names)
```

```python
logit_exp_lr = logit_ale_lr.explain(X_train)
proba_exp_lr = proba_ale_lr.explain(X_train)
```

## ALE in logit space

We first look at the ALE plots for explaining the feature effects towards the unnormalized logit scores:

```python
plot_ale(logit_exp_lr, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5}, sharey=None);
```

![png](../../.gitbook/assets/ale_classification_18_0.png)

We see that the feature effects are linear for each class and each feature. This is exactly what we expect because the logistic regression is a linear model in the logit space.

Furthermore, the units of the ALE plots here are in logits, which means that the feature effect at some feature value will be a positive or negative contribution to the logit of each class with respect to the mean feature effect.

Let's look at the interpretation of the feature effects for "petal length" in more detail:

```python
plot_ale(logit_exp_lr, features=[2]);
```

![png](../../.gitbook/assets/ale_classification_21_0.png)

The main insights from an ALE plot are qualitative—we can make several observations:

* The slope of each ALE curve determines the relative effect of the feature `petal length` on the prediction (in logits) for each target class
* In particular, we observe that the feature `petal length` has little relative variation in its effect towards the target class of `versicolor`
* On the other hand, for the target classes of `setosa` and `virginica` the slopes of the curves are significant—the relative feature effect of `petal length` rises/falls for the target class of `virginica`/`setosa` as the `petal length` increases
* The effect of `petal length` on the target classes of `setosa` and `virginica` are inversely related, suggesting that e.g. the effect of longer petal lengths contributes more positively towards predicting `virginica` and negatively towards predicting `setosa`

We can gain even more insight into the ALE plot by looking at the class histograms for the feature `petal length`:

```python
fig, ax = plt.subplots()
for target in range(3):
    ax.hist(X_train[y_train==target][:,2], label=target_names[target]);

ax.set_xlabel(feature_names[2])
ax.legend();
```

![png](../../.gitbook/assets/ale_classification_24_0.png)

Here we see that the three classes are very well separated by this feature. This confirms that the ALE plot is behaving as expected—the feature effects of small value of `petal length` are that of increasing the logit values for the class `setosa` and decreasing for the other two classes. Also note that the range of the ALE values for this feature is particularly high compared to other features which can be interpreted as the model attributing more importance to this feature as it separates the classes well on its own.

## ALE in probability space

We now turn to interpret the ALE plots for explaining the feature effects on the probabilities of each class.

```python
plot_ale(proba_exp_lr, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5});
```

![png](../../.gitbook/assets/ale_classification_28_0.png)

As expected, the ALE plots are no longer linear which reflects the non-linear nature due to the softmax transformation applied to the logits.

Note that, in this case, the ALE are in the units of relative probability mass, i.e. given a feature value how much more (less) probability does the model assign to each class relative to the mean effect of that feature. This also means that any increase in relative probability of one class must result in a decrease in probability of another class. In fact, the ALE curves summed across classes result in 0 as a direct consequence of conservation of probability:

```python
for feature in range(4):
    print(proba_exp_lr.ale_values[feature].sum())
```

```
-5.551115123125783e-17
1.734723475976807e-17
-6.661338147750939e-16
4.440892098500626e-16
```

By transforming the ALE plots into probability space we can gain additional insight into the model behaviour. For example, the ALE curve for the feature `petal width` and class `setosa` is virtually flat. This means that the model does not use this feature to assign higher or lower probability to class `setosa` with respect to the average effect of that feature. This is not readily seen in logit space as the ALE curve has negative slope which would lead us to the opposite conclusion. The interpretation here is that even though the ALE curve in the logit space shows a negative effect with feature value, the effect in the logit space is not significant enough to translate into a tangible effect in the probability space.

Turning to the feature `petal length` we can observe a much more nuanced behaviour of the ALE plots than we saw in the logit space previously. In particular, for the target class `versicolor`, whilst the ALE curve is nearly flat in the logit space, in probability space it reveals a significant uplift over the average effect of `petal length` towards predicting `versicolor` in an interval between `~3-5cm`. This agrees with our observation previously that the histogram of `petal length` by target class reveals that the feature can separate all three classes quite well.

Finally, the feature `sepal width` does not offer significant information to the model to prefer any class over the other (with respect to the mean effect of `sepal_width` that is). If we plot the marginal distribution of `sepal_width` it explains why that is—the overlap in the class conditional histograms of this feature show that it does not increase the model discriminative power:

```python
fig, ax = plt.subplots()
for target in range(3):
    ax.hist(X_train[y_train==target][:,1], label=target_names[target]);

ax.set_xlabel(feature_names[1])
ax.legend();
```

![png](../../.gitbook/assets/ale_classification_32_0.png)

## ALE for gradient boosting

Finally, we look at the resulting ALE plots for a highly non-linear model—a gradient boosted classifier.

```python
from sklearn.ensemble import GradientBoostingClassifier
```

```python
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
```

```
GradientBoostingClassifier()
```

In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.\
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.GradientBoostingClassifier

```
GradientBoostingClassifier()
```

```python
accuracy_score(y_test, gb.predict(X_test))
```

```
1.0
```

As before, we explain the feature contributions in both logit and probability space.

```python
logit_fun_gb = gb.decision_function
proba_fun_gb = gb.predict_proba
```

```python
logit_ale_gb = ALE(logit_fun_gb, feature_names=feature_names, target_names=target_names)
proba_ale_gb = ALE(proba_fun_gb, feature_names=feature_names, target_names=target_names)
```

```python
logit_exp_gb = logit_ale_gb.explain(X_train)
proba_exp_gb = proba_ale_gb.explain(X_train)
```

### ALE in logit space

```python
plot_ale(logit_exp_gb, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5});
```

![png](../../.gitbook/assets/ale_classification_43_0.png)

The ALE curves are no longer linear as the model used is non-linear. Furthermore, we've plotted the ALE curves of different features on the same scale on the $y$-axis which suggests that the features `petal length` and `petal width` are more discriminative for the task. Checking the feature importances of the classifier confirms this:

```python
gb.feature_importances_
```

```
array([0.00221272, 0.01651258, 0.51811252, 0.46316218])
```

### ALE in probability space

```python
plot_ale(proba_exp_gb, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5});
```

![png](../../.gitbook/assets/ale_classification_47_0.png)

Because of the non-linearity of the gradient boosted model the ALE curves in probability space are very similar to the curves in the logit space just on a different scale.

### Comparing ALE between models

We have seen that for both logistic regression and gradient boosting models the features `petal length` and `petal width` have a high feature effect on the classifier predictions. We can explore this in more detail by comparing the ALE curves for both models. In the following we plot the ALE curves of the two features for predicting the class `setosa` in probability space:

```python
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey='row');
plot_ale(proba_exp_lr, features=[2, 3], targets=['setosa'], ax=ax, line_kw={'label': 'LR'});
plot_ale(proba_exp_gb, features=[2, 3], targets=['setosa'], ax=ax, line_kw={'label': 'GB'});
```

![png](../../.gitbook/assets/ale_classification_51_0.png)

From this plot we can draw a couple of conclusions:

* Both models have similar feature effects of `petal length`—a high positive effect for predicting `setosa` for small feature values and a high negative effect for large values (over >3cm).
* While the logistic regression model does not benefit much from the `petal width` feature to discriminate the `setosa` class, the gradient boosted model does exploit this feature to discern between different classes.
