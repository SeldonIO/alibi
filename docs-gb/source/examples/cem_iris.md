# Contrastive Explanations Method (CEM) applied to Iris dataset

The Contrastive Explanation Method (CEM) can generate black box model explanations in terms of pertinent positives (PP) and pertinent negatives (PN). For PP, it finds what should be minimally and sufficiently present (e.g. important pixels in an image) to justify its classification. PN on the other hand identify what should be minimally and necessarily absent from the explained instance in order to maintain the original prediction.

The original paper where the algorithm is based on can be found on [arXiv](https://arxiv.org/abs/1802.07623).

This notebook requires the seaborn package for visualization which can be installed via pip:

Note

To enable support for the Contrastive Explanation Method, you may need to run

```bash
pip install alibi[tensorflow]
```

```python
!pip install seaborn
```

```python
import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from alibi.explainers import CEM

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False
```

```
TF version:  2.2.0
Eager execution enabled:  False
```

## Load and prepare Iris dataset

```python
dataset = load_iris()
feature_names = dataset.feature_names
class_names = list(dataset.target_names)
```

Scale data

```python
dataset.data = (dataset.data - dataset.data.mean(axis=0)) / dataset.data.std(axis=0)
```

Define training and test set

```python
idx = 145
x_train,y_train = dataset.data[:idx,:], dataset.target[:idx]
x_test, y_test = dataset.data[idx+1:,:], dataset.target[idx+1:]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

## Define and train logistic regression model

```python
def lr_model():
    x_in = Input(shape=(4,))
    x_out = Dense(3, activation='softmax')(x_in)
    lr = Model(inputs=x_in, outputs=x_out)
    lr.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return lr
```

```python
lr = lr_model()
lr.summary()
lr.fit(x_train, y_train, batch_size=16, epochs=500, verbose=0)
lr.save('iris_lr.h5', save_format='h5')
```

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 4)]               0         
_________________________________________________________________
dense (Dense)                (None, 3)                 15        
=================================================================
Total params: 15
Trainable params: 15
Non-trainable params: 0
_________________________________________________________________
```

## Generate contrastive explanation with pertinent negative

Explained instance:

```python
idx = 0
X = x_test[idx].reshape((1,) + x_test[idx].shape)
print(f'Prediction on instance to be explained: {class_names[np.argmax(lr.predict(X))]}')
print(f'Prediction probabilities for each class on the instance: {lr.predict(X)}')
```

```
Prediction on instance to be explained: virginica
Prediction probabilities for each class on the instance: [[2.2735458e-04 2.4420770e-01 7.5556499e-01]]
```

CEM parameters:

```python
mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
shape = (1,) + x_train.shape[1:]  # instance shape
kappa = .2  # minimum difference needed between the prediction probability for the perturbed instance on the
            # class predicted by the original instance and the max probability on the other classes 
            # in order for the first loss term to be minimized
beta = .1  # weight of the L1 loss term
c_init = 10.  # initial weight c of the loss term encouraging to predict a different class (PN) or 
              # the same class (PP) for the perturbed instance compared to the original instance to be explained
c_steps = 10  # nb of updates for c
max_iterations = 1000  # nb of iterations per value of c
feature_range = (x_train.min(axis=0).reshape(shape)-.1,  # feature range for the perturbed instance
                 x_train.max(axis=0).reshape(shape)+.1)  # can be either a float or array of shape (1xfeatures)
clip = (-1000.,1000.)  # gradient clipping
lr_init = 1e-2  # initial learning rate
```

Generate pertinent negative:

```python
# define model
lr = load_model('iris_lr.h5')

# initialize CEM explainer and explain instance
cem = CEM(lr, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, 
          max_iterations=max_iterations, c_init=c_init, c_steps=c_steps, 
          learning_rate_init=lr_init, clip=clip)
cem.fit(x_train, no_info_type='median')  # we need to define what feature values contain the least
                                         # info wrt predictions
                                         # here we will naively assume that the feature-wise median
                                         # contains no info; domain knowledge helps!
explanation = cem.explain(X, verbose=False)
```

```python
print(f'Original instance: {explanation.X}')
print(f'Predicted class: {class_names[explanation.X_pred]}')
```

```
Original instance: [[ 0.55333328 -1.28296331  0.70592084  0.92230284]]
Predicted class: virginica
```

```python
print(f'Pertinent negative: {explanation.PN}')
print(f'Predicted class: {class_names[explanation.PN_pred]}')
```

```
Pertinent negative: [[ 0.5533333  -1.2829633  -0.5391252   0.92230284]]
Predicted class: versicolor
```

Store explanation to plot later on:

```python
expl = {}
expl['PN'] = explanation.PN
expl['PN_pred'] = explanation.PN_pred
```

## Generate pertinent positive

```python
mode = 'PP'
```

Generate pertinent positive:

```python
# define model
lr = load_model('iris_lr.h5')

# initialize CEM explainer and explain instance
cem = CEM(lr, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, 
          max_iterations=max_iterations, c_init=c_init, c_steps=c_steps, 
          learning_rate_init=lr_init, clip=clip)
cem.fit(x_train, no_info_type='median')
explanation = cem.explain(X, verbose=False)
```

```python
print(f'Pertinent positive: {explanation.PP}')
print(f'Predicted class: {class_names[explanation.PP_pred]}')
```

```
Pertinent positive: [[-7.44469730e-09 -3.47054341e-08  2.68840638e-01  9.17062904e-01]]
Predicted class: virginica
```

```python
expl['PP'] = explanation.PP
expl['PP_pred'] = explanation.PP_pred
```

## Visualize PN and PP

Let's visualize the generated explanations to check if the perturbed instances make sense.

Create dataframe from standardized data:

```python
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['species'] = np.array([dataset.target_names[i] for i in dataset.target])
```

Highlight explained instance and add pertinent negative and positive to the dataset:

```python
pn = pd.DataFrame(expl['PN'], columns=dataset.feature_names)
pn['species'] = 'PN_' + class_names[expl['PN_pred']]
pp = pd.DataFrame(expl['PP'], columns=dataset.feature_names)
pp['species'] = 'PP_' + class_names[expl['PP_pred']]
orig_inst = pd.DataFrame(explanation.X, columns=dataset.feature_names)
orig_inst['species'] = 'orig_' + class_names[explanation.X_pred]
df = pd.concat([df, pn, pp, orig_inst], ignore_index=True)
```

Pair plots between the features show that the pertinent negative is pushed from the original instance (versicolor) into the virginica distribution while the pertinent positive moved away from the virginica distribution.

```python
fig = sns.pairplot(df, hue='species', diag_kind='hist');
```

![png](../../.gitbook/assets/cem_iris_37_0.png)

## Use numerical gradients in CEM

If we do not have access to the Keras or TensorFlow model weights, we can use numerical gradients for the first term in the loss function that needs to be minimized (eq. 1 and 4 in the [paper](https://arxiv.org/pdf/1802.07623.pdf)).

CEM parameters:

```python
mode = 'PN'
```

If numerical gradients are used to compute:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial p} \ast \frac{\partial p}{\partial x}
$$

with L = loss function; p = predict function and x the parameter to optimize, then the tuple _eps_ can be used to define the perturbation used to compute the derivatives. _eps\[0]_ is used to calculate the first partial derivative term and _eps\[1]_ is used for the second term. _eps\[0]_ and _eps\[1]_ can be a combination of float values or numpy arrays. For _eps\[0]_, the array dimension should be _(1 x nb of prediction categories)_ and for _eps\[1]_ it should be _(1 x nb of features)_.

```python
eps0 = np.array([[1e-2, 1e-2, 1e-2]])  # 3 prediction categories, equivalent to 1e-2
eps1 = np.array([[1e-2, 1e-2, 1e-2, 1e-2]])  # 4 features, also equivalent to 1e-2
eps = (eps0, eps1)
```

For complex models with a high number of parameters and a high dimensional feature space (e.g. Inception on ImageNet), evaluating numerical gradients can be expensive as they involve multiple prediction calls for each perturbed instance. The _update\_num\_grad_ parameter allows you to set a batch size on which to evaluate the numerical gradients, drastically reducing the number of prediction calls required.

```python
update_num_grad = 1 
```

Generate pertinent negative:

```python
# define model
lr = load_model('iris_lr.h5')
predict_fn = lambda x: lr.predict(x)  # only pass the predict fn which takes numpy arrays to CEM
                                      # explainer can no longer minimize wrt model weights

# initialize CEM explainer and explain instance
cem = CEM(predict_fn, mode, shape, kappa=kappa, beta=beta, 
          feature_range=feature_range, max_iterations=max_iterations, 
          eps=eps, c_init=c_init, c_steps=c_steps, learning_rate_init=lr_init, 
          clip=clip, update_num_grad=update_num_grad)
cem.fit(x_train, no_info_type='median')
explanation = cem.explain(X, verbose=False)
```

```python
print(f'Original instance: {explanation.X}')
print(f'Predicted class: {class_names[explanation.X_pred]}')
```

```
Original instance: [[ 0.55333328 -1.28296331  0.70592084  0.92230284]]
Predicted class: virginica
```

```python
print(f'Pertinent negative: {explanation.X}')
print(f'Predicted class: {class_names[explanation.X_pred]}')
```

```
Pertinent negative: [[ 0.55333328 -1.28296331  0.70592084  0.92230284]]
Predicted class: virginica
```

Clean up:

```python
os.remove('iris_lr.h5')
```
