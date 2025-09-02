# Anchor explanations for income prediction

In this example, we will explain predictions of a Random Forest classifier whether a person will make more or less than $50k based on characteristics like age, marital status, gender or occupation. The features are a mixture of ordinal and categorical data and will be pre-processed accordingly.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from alibi.explainers import AnchorTabular
from alibi.datasets import fetch_adult
```

### Load adult dataset

The `fetch_adult` function returns a `Bunch` object containing the features, the targets, the feature names and a mapping of categorical variables to numbers which are required for formatting the output of the Anchor explainer.

```python
adult = fetch_adult()
adult.keys()
```

```
dict_keys(['data', 'target', 'feature_names', 'target_names', 'category_map'])
```

```python
data = adult.data
target = adult.target
feature_names = adult.feature_names
category_map = adult.category_map
```

Note that for your own datasets you can use our utility function [gen\_category\_map](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/api/alibi.utils.data.rst) to create the category map:

```python
from alibi.utils import gen_category_map
```

Define shuffled training and test set

```python
np.random.seed(0)
data_perm = np.random.permutation(np.c_[data, target])
data = data_perm[:,:-1]
target = data_perm[:,-1]
```

```python
idx = 30000
X_train,Y_train = data[:idx,:], target[:idx]
X_test, Y_test = data[idx+1:,:], target[idx+1:]
```

### Create feature transformation pipeline

Create feature pre-processor. Needs to have 'fit' and 'transform' methods. Different types of pre-processing can be applied to all or part of the features. In the example below we will standardize ordinal features and apply one-hot-encoding to categorical features.

Ordinal features:

```python
ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])
```

Categorical features:

```python
categorical_features = list(category_map.keys())
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])
```

Combine and fit:

```python
preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                               ('cat', categorical_transformer, categorical_features)])
preprocessor.fit(X_train)
```

```
ColumnTransformer(n_jobs=None, remainder='drop', sparse_threshold=0.3,
                  transformer_weights=None,
                  transformers=[('num',
                                 Pipeline(memory=None,
                                          steps=[('imputer',
                                                  SimpleImputer(add_indicator=False,
                                                                copy=True,
                                                                fill_value=None,
                                                                missing_values=nan,
                                                                strategy='median',
                                                                verbose=0)),
                                                 ('scaler',
                                                  StandardScaler(copy=True,
                                                                 with_mean=True,
                                                                 with_std=True))],
                                          verbose=False),
                                 [0, 8, 9, 10]),
                                ('cat',
                                 Pipeline(memory=None,
                                          steps=[('imputer',
                                                  SimpleImputer(add_indicator=False,
                                                                copy=True,
                                                                fill_value=None,
                                                                missing_values=nan,
                                                                strategy='median',
                                                                verbose=0)),
                                                 ('onehot',
                                                  OneHotEncoder(categories='auto',
                                                                drop=None,
                                                                dtype=<class 'numpy.float64'>,
                                                                handle_unknown='ignore',
                                                                sparse=True))],
                                          verbose=False),
                                 [1, 2, 3, 4, 5, 6, 7, 11])],
                  verbose=False)
```

### Train Random Forest model

Fit on pre-processed (imputing, OHE, standardizing) data.

```python
np.random.seed(0)
clf = RandomForestClassifier(n_estimators=50)
clf.fit(preprocessor.transform(X_train), Y_train)
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
predict_fn = lambda x: clf.predict(preprocessor.transform(x))
print('Train accuracy: ', accuracy_score(Y_train, predict_fn(X_train)))
print('Test accuracy: ', accuracy_score(Y_test, predict_fn(X_test)))
```

```
Train accuracy:  0.9655333333333334
Test accuracy:  0.855859375
```

### Initialize and fit anchor explainer for tabular data

```python
explainer = AnchorTabular(predict_fn, feature_names, categorical_names=category_map, seed=1)
```

Discretize the ordinal features into quartiles

```python
explainer.fit(X_train, disc_perc=[25, 50, 75])
```

```
AnchorTabular(meta={
    'name': 'AnchorTabular',
    'type': ['blackbox'],
    'explanations': ['local'],
    'params': {'seed': 1, 'disc_perc': [25, 50, 75]}
})
```

### Getting an anchor

Below, we get an anchor for the prediction of the first observation in the test set. An anchor is a sufficient condition - that is, when the anchor holds, the prediction should be the same as the prediction for this instance.

```python
idx = 0
class_names = adult.target_names
print('Prediction: ', class_names[explainer.predictor(X_test[idx].reshape(1, -1))[0]])
```

```
Prediction:  <=50K
```

We set the precision threshold to 0.95. This means that predictions on observations where the anchor holds will be the same as the prediction on the explained instance at least 95% of the time.

```python
explanation = explainer.explain(X_test[idx], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)
```

```
Anchor: Marital Status = Separated AND Sex = Female
Precision: 0.95
Coverage: 0.18
```

### ...or not?

Let's try getting an anchor for a different observation in the test set - one for the which the prediction is `>50K`.

```python
idx = 6
class_names = adult.target_names
print('Prediction: ', class_names[explainer.predictor(X_test[idx].reshape(1, -1))[0]])

explanation = explainer.explain(X_test[idx], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)
```

```
Prediction:  >50K


Could not find an result satisfying the 0.95 precision constraint. Now returning the best non-eligible result.


Anchor: Capital Loss > 0.00 AND Relationship = Husband AND Marital Status = Married AND Age > 37.00 AND Race = White AND Country = United-States AND Sex = Male
Precision: 0.71
Coverage: 0.05
```

Notice how no anchor is found!

This is due to the imbalanced dataset (roughly 25:75 high:low earner proportion), so during the sampling stage feature ranges corresponding to low-earners will be oversampled. This is a feature because it can point out an imbalanced dataset, but it can also be fixed by producing balanced datasets to enable anchors to be found for either class.
