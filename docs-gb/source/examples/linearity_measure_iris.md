# Linearity measure applied to Iris

## General definition

The model linearity module in alibi provides metric to measure how linear an ML model is. Linearity is defined based on how much the linear superposition of the model's outputs differs from the output of the same linear superposition of the inputs.

Given $N$ input vectors $v\_i$, $N$ real coefficients $\alpha\_i$ and a predict function $\text{M}(v\_i)$, the linearity of the predict function is defined as

$$L = \Big|\Big|\sum_i \alpha_i M(v_i) - M\Big(\sum_i \alpha_i v_i\Big) \Big|\Big| \quad \quad \text{If M is a regressor}$$

$$L = \Big|\Big|\sum_i \alpha_i \log \circ M(v_i) - \log \circ M\Big(\sum_i \alpha_i v_i\Big)\Big|\Big| \quad \quad \text{If M is a classifier}$$

Note that a lower value of $L$ means that the model $M$ is more linear.

## Alibi implementation

* Based on the general definition above, alibi calculates the linearity of a model in the neighboorhood of a given instance $v\_0$.

## Iris Data set

* As an example, we will visualize the decision boundaries and the values of the linearity measure for various classifier on the iris dataset. Only 2 features are included for visualization porpuses.

This example will use the [xgboost](https://github.com/dmlc/xgboost) library, which can be installed with:

```python
!pip install xgboost
```

```python
import pandas as pd
import numpy as np
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from itertools import product
from alibi.confidence import linearity_measure, LinearityMeasure
```

## Dataset

```python
ds = load_iris()
X_train, y_train = ds.data[:, :2], ds.target
```

```python
lins_dict = {}
```

## Models

We will experiment with 5 different classifiers:

* A logistic regression model, which is expected to be highly linear.
* A random forest classifier, which is expected to be higly non-linear.
* An xgboost classifier.
* A support vector machine classifier.
* A feed forward neural network

```python
lr = LogisticRegression(fit_intercept=False, multi_class='multinomial', solver='newton-cg')
rf = RandomForestClassifier(n_estimators=100)
xgb = XGBClassifier(n_estimators=100)
svm = SVC(gamma=.1, kernel='rbf', probability=True)
nn = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', max_iter=1000)
```

```python
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
svm.fit(X_train, y_train)
nn.fit(X_train, y_train);
```

## Decision boundaries and linearity

```python
# Creating a grid
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
```

```python
# Flattening points in the grid
X = np.empty((len(xx.flatten()), 2))
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        k = i * xx.shape[1] + j
        X[k] = np.array([xx[i, j], yy[i, j]])
```

### Logistic regression

```python
# Defining predict function for logistic regression
clf = lr
predict_fn = lambda x: clf.predict_proba(x)
```

```python
# Calculating linearity for all points in the grid
lm = LinearityMeasure(agg='pairwise')
lm.fit(X_train)
L = lm.score(predict_fn, X)
L = L.reshape(xx.shape)
lins_dict['LR'] = L.mean()
```

```python
# Visualising decision boundaries and linearity values 
f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(16, 8))
idx = (0,0)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axarr[0].contourf(xx, yy, Z, alpha=0.4)
axarr[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', alpha=1)
axarr[0].set_title('Decision boundaries', fontsize=20)
axarr[0].set_xlabel('sepal length (cm)',fontsize=18)
axarr[0].set_ylabel('sepal width (cm)', fontsize=18)

LPL = axarr[1].contourf(xx, yy, L, alpha=0.8, cmap='Greys')
axarr[1].set_title('Model linearity', fontsize=20)
axarr[1].set_xlabel('sepal length (cm)', fontsize=18)
axarr[1].set_ylabel('sepal width (cm)', fontsize=18)
cbar = f.colorbar(LPL)
#cbar.ax.set_ylabel('Linearity')
plt.show()
print('Decision boundaries (left panel) and linearity measure (right panel) for a logistic regression (LG) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.')
print('Maximum value model linearity: {}'. format(np.round(L.max(), 5))) 
print(f'Minimum value model linearity: {np.round(L.min(),5)}')
```

![png](../../.gitbook/assets/linearity_measure_iris_16_0.png)

```
Decision boundaries (left panel) and linearity measure (right panel) for a logistic regression (LG) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.
Maximum value model linearity: 0.01841
Minimum value model linearity: 0.0
```

### Random forest

```python
# Defining predict function for random forest
clf = rf
predict_fn = lambda x: clf.predict_proba(x)
```

```python
# Calculating linearity for all points in the grid
lm = LinearityMeasure(agg='pairwise')
lm.fit(X_train)
L = lm.score(predict_fn, X)
L = L.reshape(xx.shape)
lins_dict['RF'] = L.mean()
```

```python
# Visualising decision boundaries and linearity values 
f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(16, 8))
idx = (0,0)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axarr[0].contourf(xx, yy, Z, alpha=0.4)
axarr[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', alpha=1)
axarr[0].set_title('Decision boundaries', fontsize=20)
axarr[0].set_xlabel('sepal length (cm)', fontsize=18)
axarr[0].set_ylabel('sepal width (cm)', fontsize=18)

LPL = axarr[1].contourf(xx, yy, L, alpha=0.8, cmap='Greys')
axarr[1].set_title('Model linearity', fontsize=20)
axarr[1].set_xlabel('sepal length (cm)', fontsize=18)
axarr[1].set_ylabel('sepal width (cm)', fontsize=18)

cbar = f.colorbar(LPL)
plt.show()
print('Decision boundaries (left panel) and linearity measure (right panel) for a random forest (RF) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.')
print('Maximum value model linearity: {}'. format(np.round(L.max(), 5))) 
print(f'Minimum value model linearity: {np.round(L.min(),5)}')
```

![png](../../.gitbook/assets/linearity_measure_iris_20_0.png)

```
Decision boundaries (left panel) and linearity measure (right panel) for a random forest (RF) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.
Maximum value model linearity: 12.07288
Minimum value model linearity: 0.0
```

### Xgboost

```python
# Defining predict function for xgboost
clf = xgb
predict_fn = lambda x: clf.predict_proba(x)
```

```python
# Calculating linearity for all points in the grid
lm = LinearityMeasure(agg='pairwise')
lm.fit(X_train)
L = lm.score(predict_fn, X)
L = L.reshape(xx.shape)
lins_dict['XB'] = L.mean()
```

```python
# Visualising decision boundaries and linearity values 
f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(16, 8))
idx = (0,0)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axarr[0].contourf(xx, yy, Z, alpha=0.4)
axarr[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', alpha=1)
axarr[0].set_title('Decision boundaries', fontsize=20)
axarr[0].set_xlabel('sepal length (cm)', fontsize=20)
axarr[0].set_ylabel('sepal width (cm)', fontsize=20)

LPL = axarr[1].contourf(xx, yy, L, alpha=0.8, cmap='Greys')
axarr[1].set_title('L measure', fontsize=20)
axarr[1].set_xlabel('sepal length (cm)', fontsize=20)
axarr[1].set_ylabel('sepal width (cm)', fontsize=20)

cbar = f.colorbar(LPL)
#cbar.ax.set_ylabel('Linearity')
plt.show()
print('Decision boundaries (left panel) and linearity measure (right panel) for a xgboost (XB) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.')
print('Maximum value model linearity: {}'. format(np.round(L.max(), 5))) 
print(f'Minimum value model linearity: {np.round(L.min(),5)}')
```

![png](../../.gitbook/assets/linearity_measure_iris_24_0.png)

```
Decision boundaries (left panel) and linearity measure (right panel) for a xgboost (XB) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.
Maximum value model linearity: 1.42648
Minimum value model linearity: 0.0
```

### SVM

```python
# Defining predict function for svm
clf = svm
predict_fn = lambda x: clf.predict_proba(x)
```

```python
# Calculating linearity for all points in the grid
lm = LinearityMeasure(agg='pairwise')
lm.fit(X_train)
L = lm.score(predict_fn, X)
L = L.reshape(xx.shape)
lins_dict['SM'] = L.mean()
```

```python
# Visualising decision boundaries and linearity values 
f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(16, 8))
idx = (0,0)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axarr[0].contourf(xx, yy, Z, alpha=0.4)
axarr[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', alpha=1)
axarr[0].set_title('Decision boundaries', fontsize=20)
axarr[0].set_xlabel('sepal length (cm)', fontsize=18)
axarr[0].set_ylabel('sepal width (cm)', fontsize=18)

LPL = axarr[1].contourf(xx, yy, L, alpha=0.8, cmap='Greys')
axarr[1].set_title('Model linearity', fontsize=20)
axarr[1].set_xlabel('sepal length (cm)', fontsize=18)
axarr[1].set_ylabel('sepal width (cm)', fontsize=18)

cbar = f.colorbar(LPL)
#cbar.ax.set_ylabel('Linearity')
plt.show()
print('Decision boundaries (left panel) and linearity measure (right panel) for a support vector machine (SM) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.')
print('Maximum value model linearity: {}'. format(np.round(L.max(), 5))) 
print(f'Minimum value model linearity: {np.round(L.min(),5)}')
```

![png](../../.gitbook/assets/linearity_measure_iris_28_0.png)

```
Decision boundaries (left panel) and linearity measure (right panel) for a support vector machine (SM) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.
Maximum value model linearity: 0.45113
Minimum value model linearity: 0.00083
```

### NN

```python
# Defining predict function for svm
clf = nn
predict_fn = lambda x: clf.predict_proba(x)
```

```python
# Calculating linearity for all points in the grid
lm = LinearityMeasure(agg='pairwise')
lm.fit(X_train)
L = lm.score(predict_fn, X)
L = L.reshape(xx.shape)
lins_dict['NN'] = L.mean()
```

```python
# Visualising decision boundaries and linearity values 
f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(16, 8))
idx = (0,0)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axarr[0].contourf(xx, yy, Z, alpha=0.4)
axarr[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', alpha=1)
axarr[0].set_title('Decision boundaries', fontsize=20)
axarr[0].set_xlabel('sepal length (cm)', fontsize=18)
axarr[0].set_ylabel('sepal width (cm)', fontsize=18)

LPL = axarr[1].contourf(xx, yy, L, alpha=0.8, cmap='Greys')
axarr[1].set_title('Model linearity', fontsize=20)
axarr[1].set_xlabel('sepal length (cm)', fontsize=18)
axarr[1].set_ylabel('sepal width (cm)', fontsize=18)

cbar = f.colorbar(LPL)
#cbar.ax.set_ylabel('Linearity')
plt.show()
print('Decision boundaries (left panel) and linearity measure (right panel) for a feed forward neural network classifier (NN) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.')
print('Maximum value model linearity: {}'. format(np.round(L.max(), 5))) 
print(f'Minimum value model linearity: {np.round(L.min(),5)}')
```

![png](../../.gitbook/assets/linearity_measure_iris_32_0.png)

```
Decision boundaries (left panel) and linearity measure (right panel) for a feed forward neural network classifier (NN) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.
Maximum value model linearity: 0.11615
Minimum value model linearity: 3e-05
```

## Average linearity over the whole feature space

```python
ax = pd.Series(data=lins_dict).sort_values().plot(kind='barh', figsize=(10,5), fontsize=20, color='dimgray', 
                                                  width=0.8, logx=True)
ax.set_xlabel('L measure (log scale)', fontsize=20)
print('Comparison of the linearity measure L averaged over the whole feature space for various models trained on the iris dataset: random forest (RF), xgboost (XB), support vector machine (SM), neural network (NN) and logistic regression (LR). Note that the scale of the X axis is logarithmic.')
```

```
Comparison of the linearity measure L averaged over the whole feature space for various models trained on the iris dataset: random forest (RF), xgboost (XB), support vector machine (SM), neural network (NN) and logistic regression (LR). Note that the scale of the X axis is logarithmic.



```

![png](../../.gitbook/assets/linearity_measure_iris_34_1.png)
