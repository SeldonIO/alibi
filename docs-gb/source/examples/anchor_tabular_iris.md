# Anchor explanations on the Iris dataset

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from alibi.explainers import AnchorTabular
```

### Load iris dataset

```python
dataset = load_iris()
feature_names = dataset.feature_names
class_names = list(dataset.target_names)
```

Define training and test set

```python
idx = 145
X_train,Y_train = dataset.data[:idx,:], dataset.target[:idx]
X_test, Y_test = dataset.data[idx+1:,:], dataset.target[idx+1:]
```

### Train Random Forest model

```python
np.random.seed(0)
clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, Y_train)
```

```
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
```

Define predict function

```python
predict_fn = lambda x: clf.predict_proba(x)
```

### Initialize and fit anchor explainer for tabular data

```python
explainer = AnchorTabular(predict_fn, feature_names)
```

Discretize the ordinal features into quartiles

```python
explainer.fit(X_train, disc_perc=(25, 50, 75))
```

```
AnchorTabular(meta={
    'name': 'AnchorTabular',
    'type': ['blackbox'],
    'explanations': ['local'],
    'params': {'seed': None, 'disc_perc': (25, 50, 75)}
})
```

### Getting an anchor

Below, we get an anchor for the prediction of the first observation in the test set. An anchor is a sufficient condition - that is, when the anchor holds, the prediction should be the same as the prediction for this instance.

```python
idx = 0
print('Prediction: ', class_names[explainer.predictor(X_test[idx].reshape(1, -1))[0]])
```

```
Prediction:  virginica
```

We set the precision threshold to 0.95. This means that predictions on observations where the anchor holds will be the same as the prediction on the explained instance at least 95% of the time.

```python
explanation = explainer.explain(X_test[idx], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)
```

```
Anchor: petal width (cm) > 1.80 AND sepal width (cm) <= 2.80
Precision: 0.98
Coverage: 0.32
```
