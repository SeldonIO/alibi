# Counterfactual explanations with ordinally encoded categorical variables

This example notebook illustrates how to obtain [counterfactual explanations](https://docs.seldon.io/projects/alibi/en/stable/methods/CFProto.html) for instances with a mixture of ordinally encoded categorical and numerical variables. A more elaborate notebook highlighting additional functionality can be found [here](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/cfproto_cat_adult_ohe.ipynb). We generate counterfactuals for instances in the _adult_ dataset where we predict whether a person's income is above or below $50k.

Note

To enable support for CounterfactualProto, you may need to run

```bash
pip install alibi[tensorflow]
```

```python
import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs 
from tensorflow.keras.layers import Dense, Input, Embedding, Concatenate, Reshape, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from time import time
from alibi.datasets import fetch_adult
from alibi.explainers import CounterfactualProto

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False
```

```
TF version:  2.2.0
Eager execution enabled:  False
```

## Load adult dataset

The `fetch_adult` function returns a `Bunch` object containing the features, the targets, the feature names and a mapping of the categories in each categorical variable.

```python
adult = fetch_adult()
data = adult.data
target = adult.target
feature_names = adult.feature_names
category_map_tmp = adult.category_map
target_names = adult.target_names
```

Define shuffled training and test set:

```python
def set_seed(s=0):
    np.random.seed(s)
    tf.random.set_seed(s)
```

```python
set_seed()
data_perm = np.random.permutation(np.c_[data, target])
X = data_perm[:,:-1]
y = data_perm[:,-1]
```

```python
idx = 30000
y_train, y_test = y[:idx], y[idx+1:]
```

Reorganize data so categorical features come first:

```python
X = np.c_[X[:, 1:8], X[:, 11], X[:, 0], X[:, 8:11]]
```

Adjust `feature_names` and `category_map` as well:

```python
feature_names = feature_names[1:8] + feature_names[11:12] + feature_names[0:1] + feature_names[8:11]
print(feature_names)
```

```
['Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country', 'Age', 'Capital Gain', 'Capital Loss', 'Hours per week']
```

```python
category_map = {}
for i, (_, v) in enumerate(category_map_tmp.items()):
    category_map[i] = v
```

Create a dictionary with as keys the categorical columns and values the number of categories for each variable in the dataset. This dictionary will later be used in the counterfactual explanation.

```python
cat_vars_ord = {}
n_categories = len(list(category_map.keys()))
for i in range(n_categories):
    cat_vars_ord[i] = len(np.unique(X[:, i]))
print(cat_vars_ord)
```

```
{0: 9, 1: 7, 2: 4, 3: 9, 4: 6, 5: 5, 6: 2, 7: 11}
```

## Preprocess data

Scale numerical features between -1 and 1:

```python
X_num = X[:, -4:].astype(np.float32, copy=False)
xmin, xmax = X_num.min(axis=0), X_num.max(axis=0)
rng = (-1., 1.)
X_num_scaled = (X_num - xmin) / (xmax - xmin) * (rng[1] - rng[0]) + rng[0]
X_num_scaled_train = X_num_scaled[:idx, :]
X_num_scaled_test = X_num_scaled[idx+1:, :]
```

Combine numerical and categorical data:

```python
X = np.c_[X[:, :-4], X_num_scaled].astype(np.float32, copy=False)
X_train, X_test = X[:idx, :], X[idx+1:, :]
print(X_train.shape, X_test.shape)
```

```
(30000, 12) (2560, 12)
```

## Train a neural net

The neural net will use entity embeddings for the categorical variables.

```python
def nn_ord():
    
    x_in = Input(shape=(12,))
    layers_in = []
    
    # embedding layers
    for i, (_, v) in enumerate(cat_vars_ord.items()):
        emb_in = Lambda(lambda x: x[:, i:i+1])(x_in)
        emb_dim = int(max(min(np.ceil(.5 * v), 50), 2))
        emb_layer = Embedding(input_dim=v+1, output_dim=emb_dim, input_length=1)(emb_in)
        emb_layer = Reshape(target_shape=(emb_dim,))(emb_layer)
        layers_in.append(emb_layer)
        
    # numerical layers
    num_in = Lambda(lambda x: x[:, -4:])(x_in)
    num_layer = Dense(16)(num_in)
    layers_in.append(num_layer)
    
    # combine
    x = Concatenate()(layers_in)
    x = Dense(60, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(60, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(60, activation='relu')(x)
    x = Dropout(.2)(x)
    x_out = Dense(2, activation='softmax')(x)
    
    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return nn
```

```python
set_seed()
nn = nn_ord()
nn.summary()
nn.fit(X_train, to_categorical(y_train), batch_size=128, epochs=30, verbose=0)
```

```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 12)]         0                                            
__________________________________________________________________________________________________
lambda (Lambda)                 (None, 1)            0           input_1[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 1)            0           input_1[0][0]                    
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 1)            0           input_1[0][0]                    
__________________________________________________________________________________________________
lambda_3 (Lambda)               (None, 1)            0           input_1[0][0]                    
__________________________________________________________________________________________________
lambda_4 (Lambda)               (None, 1)            0           input_1[0][0]                    
__________________________________________________________________________________________________
lambda_5 (Lambda)               (None, 1)            0           input_1[0][0]                    
__________________________________________________________________________________________________
lambda_6 (Lambda)               (None, 1)            0           input_1[0][0]                    
__________________________________________________________________________________________________
lambda_7 (Lambda)               (None, 1)            0           input_1[0][0]                    
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1, 5)         50          lambda[0][0]                     
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1, 4)         32          lambda_1[0][0]                   
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 1, 2)         10          lambda_2[0][0]                   
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 1, 5)         50          lambda_3[0][0]                   
__________________________________________________________________________________________________
embedding_4 (Embedding)         (None, 1, 3)         21          lambda_4[0][0]                   
__________________________________________________________________________________________________
embedding_5 (Embedding)         (None, 1, 3)         18          lambda_5[0][0]                   
__________________________________________________________________________________________________
embedding_6 (Embedding)         (None, 1, 2)         6           lambda_6[0][0]                   
__________________________________________________________________________________________________
embedding_7 (Embedding)         (None, 1, 6)         72          lambda_7[0][0]                   
__________________________________________________________________________________________________
lambda_8 (Lambda)               (None, 4)            0           input_1[0][0]                    
__________________________________________________________________________________________________
reshape (Reshape)               (None, 5)            0           embedding[0][0]                  
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 4)            0           embedding_1[0][0]                
__________________________________________________________________________________________________
reshape_2 (Reshape)             (None, 2)            0           embedding_2[0][0]                
__________________________________________________________________________________________________
reshape_3 (Reshape)             (None, 5)            0           embedding_3[0][0]                
__________________________________________________________________________________________________
reshape_4 (Reshape)             (None, 3)            0           embedding_4[0][0]                
__________________________________________________________________________________________________
reshape_5 (Reshape)             (None, 3)            0           embedding_5[0][0]                
__________________________________________________________________________________________________
reshape_6 (Reshape)             (None, 2)            0           embedding_6[0][0]                
__________________________________________________________________________________________________
reshape_7 (Reshape)             (None, 6)            0           embedding_7[0][0]                
__________________________________________________________________________________________________
dense (Dense)                   (None, 16)           80          lambda_8[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 46)           0           reshape[0][0]                    
                                                                 reshape_1[0][0]                  
                                                                 reshape_2[0][0]                  
                                                                 reshape_3[0][0]                  
                                                                 reshape_4[0][0]                  
                                                                 reshape_5[0][0]                  
                                                                 reshape_6[0][0]                  
                                                                 reshape_7[0][0]                  
                                                                 dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 60)           2820        concatenate[0][0]                
__________________________________________________________________________________________________
dropout (Dropout)               (None, 60)           0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 60)           3660        dropout[0][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 60)           0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 60)           3660        dropout_1[0][0]                  
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 60)           0           dense_3[0][0]                    
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 2)            122         dropout_2[0][0]                  
==================================================================================================
Total params: 10,601
Trainable params: 10,601
Non-trainable params: 0
__________________________________________________________________________________________________





<tensorflow.python.keras.callbacks.History at 0x7f482905f8d0>
```

## Generate counterfactual

Original instance:

```python
X = X_test[0].reshape((1,) + X_test[0].shape)
```

Initialize counterfactual parameters:

```python
shape = X.shape
beta = .01
c_init = 1.
c_steps = 5
max_iterations = 500
rng = (-1., 1.)  # scale features between -1 and 1
rng_shape = (1,) + data.shape[1:]
feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32), 
                 (np.ones(rng_shape) * rng[1]).astype(np.float32))
```

Initialize explainer. Since the `Embedding` layers in `tf.keras` do not let gradients propagate through, we will only make use of the model's predict function, treat it as a black box and perform numerical gradient calculations.

```python
set_seed()

# define predict function
predict_fn = lambda x: nn.predict(x)

cf = CounterfactualProto(predict_fn,
                         shape,
                         beta=beta,
                         cat_vars=cat_vars_ord,
                         max_iterations=max_iterations,
                         feature_range=feature_range,
                         c_init=c_init,
                         c_steps=c_steps,
                         eps=(.01, .01)  # perturbation size for numerical gradients
                        )
```

Fit explainer. Please check the [documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/CFProto.html) for more info about the optional arguments.

```python
cf.fit(X_train, d_type='abdm', disc_perc=[25, 50, 75]);
```

Explain instance:

```python
set_seed()
explanation = cf.explain(X)
```

Helper function to more clearly describe explanations:

```python
def describe_instance(X, explanation, eps=1e-2):
    print('Original instance: {}  -- proba: {}'.format(target_names[explanation.orig_class],
                                                       explanation.orig_proba[0]))
    print('Counterfactual instance: {}  -- proba: {}'.format(target_names[explanation.cf['class']],
                                                             explanation.cf['proba'][0]))
    print('\nCounterfactual perturbations...')
    print('\nCategorical:')
    X_orig_ord = X
    X_cf_ord = explanation.cf['X']
    delta_cat = {}
    for i, (_, v) in enumerate(category_map.items()):
        cat_orig = v[int(X_orig_ord[0, i])]
        cat_cf = v[int(X_cf_ord[0, i])]
        if cat_orig != cat_cf:
            delta_cat[feature_names[i]] = [cat_orig, cat_cf]
    if delta_cat:
        for k, v in delta_cat.items():
            print('{}: {}  -->   {}'.format(k, v[0], v[1]))
    print('\nNumerical:')
    delta_num = X_cf_ord[0, -4:] - X_orig_ord[0, -4:]
    n_keys = len(list(cat_vars_ord.keys()))
    for i in range(delta_num.shape[0]):
        if np.abs(delta_num[i]) > eps:
            print('{}: {:.2f}  -->   {:.2f}'.format(feature_names[i+n_keys],
                                            X_orig_ord[0,i+n_keys],
                                            X_cf_ord[0,i+n_keys]))
```

```python
describe_instance(X, explanation)
```

```
Original instance: <=50K  -- proba: [0.6976237  0.30237624]
Counterfactual instance: >50K  -- proba: [0.49604183 0.5039582 ]

Counterfactual perturbations...

Categorical:

Numerical:
Capital Gain: -1.00  -->   -0.88
```

The person's incomce is predicted to be above $50k by increasing his or her capital gain.
