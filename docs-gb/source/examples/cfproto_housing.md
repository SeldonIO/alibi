# Counterfactuals guided by prototypes on California housing dataset

This notebook goes through an example of [prototypical counterfactuals](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFProto.ipynb) using [k-d trees](https://en.wikipedia.org/wiki/K-d_tree) to build the prototypes. Please check out [this notebook](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/cfproto_mnist.ipynb) for a more in-depth application of the method on MNIST using (auto-)encoders and trust scores.

In this example, we will train a simple neural net to predict whether house prices in California districts are above the median value or not. We can then find a counterfactual to see which variables need to be changed to increase or decrease a house price above or below the median value.

Note

To enable support for CounterfactualProto, you may need to run

```bash
pip install alibi[tensorflow]
```

```python
%matplotlib inline
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from alibi.explainers import CounterfactualProto

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False
```

```
TF version:  2.7.4
Eager execution enabled:  False
```

## Load and prepare California housing dataset

```python
california = fetch_california_housing(as_frame=True)
X = california.data.to_numpy()
target = california.target.to_numpy()
feature_names = california.feature_names
```

```python
california.data.head()
```

|   | MedInc | HouseAge | AveRooms | AveBedrms | Population | AveOccup | Latitude | Longitude |
| - | ------ | -------- | -------- | --------- | ---------- | -------- | -------- | --------- |
| 0 | 8.3252 | 41.0     | 6.984127 | 1.023810  | 322.0      | 2.555556 | 37.88    | -122.23   |
| 1 | 8.3014 | 21.0     | 6.238137 | 0.971880  | 2401.0     | 2.109842 | 37.86    | -122.22   |
| 2 | 7.2574 | 52.0     | 8.288136 | 1.073446  | 496.0      | 2.802260 | 37.85    | -122.24   |
| 3 | 5.6431 | 52.0     | 5.817352 | 1.073059  | 558.0      | 2.547945 | 37.85    | -122.25   |
| 4 | 3.8462 | 52.0     | 6.281853 | 1.081081  | 565.0      | 2.181467 | 37.85    | -122.25   |

Each row represents a whole census group. Explanation of features:

* `MedInc` - median income in block group
* `HouseAge` - median house age in block group
* `AveRooms` - average number of rooms per household
* `AveBedrms` - average number of bedrooms per household
* `Population` - block group population
* `AveOccup` - average number of household members
* `Latitude` - block group latitude
* `Longitude` - block group longitude

For more details on the dataset, refer to the [scikit-learn documentation](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

Transform into classification task: target becomes whether house price is above the overall median or not

```python
y = np.zeros((target.shape[0],))
y[np.where(target > np.median(target))[0]] = 1
```

Standardize data

```python
mu = X.mean(axis=0)
sigma = X.std(axis=0)
X = (X - mu) / sigma
```

Define train and test set

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

## Train model

```python
np.random.seed(42)
tf.random.set_seed(42)
```

```python
def nn_model():
    x_in = Input(shape=(8,))
    x = Dense(40, activation='relu')(x_in)
    x = Dense(40, activation='relu')(x)
    x_out = Dense(2, activation='softmax')(x)
    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return nn
```

```python
nn = nn_model()
nn.summary()
nn.fit(X_train, y_train, batch_size=64, epochs=500, verbose=0)
nn.save('nn_california.h5', save_format='h5')
```

```python
nn = load_model('nn_california.h5')
score = nn.evaluate(X_test, y_test, verbose=0)
print('Test accuracy: ', score[1])
```

```
Test accuracy:  0.87863374
```

## Generate counterfactual guided by the nearest class prototype

Original instance:

```python
X = X_test[1].reshape((1,) + X_test[1].shape)
shape = X.shape
```

Run counterfactual:

```python
# define model
nn = load_model('nn_california.h5')

# initialize and fit the explainer
cf = CounterfactualProto(nn, shape, use_kdtree=True, theta=10., max_iterations=1000,
                         feature_range=(X_train.min(axis=0), X_train.max(axis=0)), 
                         c_init=1., c_steps=10)

cf.fit(X_train)
```

```python
# generate a counterfactual
explanation = cf.explain(X)
```

The prediction flipped from 0 (value below the median) to 1 (above the median):

```python
print(f'Original prediction: {explanation.orig_class}')
print(f'Counterfactual prediction: {explanation.cf["class"]}')
```

```
Original prediction: 0
Counterfactual prediction: 1
```

Let's take a look at the counterfactual. To make the results more interpretable, we will first undo the pre-processing step and then check where the counterfactual differs from the original instance:

```python
orig = X * sigma + mu
counterfactual = explanation.cf['X'] * sigma + mu
delta = counterfactual - orig
for i, f in enumerate(feature_names):
    if np.abs(delta[0][i]) > 1e-4:
        print(f'{f}: {delta[0][i]}')
```

```
AveOccup: -0.9049749915631999
Latitude: -0.31885583625280134
```

So in order for the model to consider the census group as having above median house prices, the average occupancy would have to be lower by almost a whole household member, and the location of the census group would need to shift slightly South.

Comparing the original instance and the counterfactual side-by-side:

```python
pd.DataFrame(orig, columns=feature_names)
```

|   | MedInc | HouseAge | AveRooms | AveBedrms | Population | AveOccup | Latitude | Longitude |
| - | ------ | -------- | -------- | --------- | ---------- | -------- | -------- | --------- |
| 0 | 2.5313 | 30.0     | 5.039384 | 1.193493  | 1565.0     | 2.679795 | 35.14    | -119.46   |

```python
pd.DataFrame(counterfactual, columns=feature_names)
```

|   | MedInc | HouseAge | AveRooms | AveBedrms | Population  | AveOccup | Latitude  | Longitude |
| - | ------ | -------- | -------- | --------- | ----------- | -------- | --------- | --------- |
| 0 | 2.5313 | 30.0     | 5.039384 | 1.193493  | 1565.000004 | 1.77482  | 34.821144 | -119.46   |

Clean up:

```python
os.remove('nn_california.h5')
```
