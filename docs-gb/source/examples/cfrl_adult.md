# Counterfactual with Reinforcement Learning (CFRL) on Adult Census

This method is described in [Model-agnostic and Scalable Counterfactual Explanations via Reinforcement Learning](https://arxiv.org/abs/2106.02597) and can generate counterfactual instances for any black-box model. The usual optimization procedure is transformed into a learnable process allowing to generate batches of counterfactual instances in a single forward pass even for high dimensional data. The training pipeline is model-agnostic and relies only on prediction feedback by querying the black-box model. Furthermore, the method allows target and feature conditioning.

**We exemplify the use case for the TensorFlow backend. This means that all models: the autoencoder, the actor and the critic are TensorFlow models. Our implementation supports PyTorch backend as well.**

CFRL uses [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) by interleaving a state-action function approximator called critic, with a learning an approximator called actor to predict the optimal action. The method assumes that the critic is differentiable with respect to the action argument, thus allowing to optimize the actor's parameters efficiently through gradient-based methods.

The DDPG algorithm requires two separate networks, an actor $\mu$ and a critic $Q$. Given the encoded representation of the input instance $z = enc(x)$, the model prediction $y\_M$, the target prediction $y\_T$ and the conditioning vector $c$, the actor outputs the counterfactual’s latent representation $z\_{CF} = \mu(z, y\_M, y\_T, c)$. The decoder then projects the embedding $z\_{CF}$ back to the original input space, followed by optional post-processing.

The training step consists of simultaneously optimizing the actor and critic networks. The critic regresses on the reward $R$ determined by the model prediction, while the actor maximizes the critic’s output for the given instance through $L\_{max}$. The actor also minimizes two objectives to encourage the generation of sparse, in-distribution counterfactuals. The sparsity loss $L\_{sparsity}$ operates on the decoded counterfactual $x\_{CF}$ and combines the $L\_1$ loss over the standardized numerical features and the $L\_0$ loss over the categorical ones. The consistency loss $L\_{consist}$ aims to encode the counterfactual $x\_{CF}$ back to the same latent representation where it was decoded from and helps to produce in-distribution counterfactual instances. Formally, the actor's loss can be written as: $L\_{actor} = L\_{max} + \lambda\_{1}L\_{sparsity} + \lambda\_{2}L\_{consistency}$

This example will use the [xgboost](https://github.com/dmlc/xgboost) library, which can be installed with:

Note

To enable support for CounterfactualRLTabular with tensorflow backend, you may need to run

```bash
pip install alibi[tensorflow]
```

```python
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Tuple, Dict, Callable

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from alibi.explainers import CounterfactualRLTabular, CounterfactualRL
from alibi.datasets import fetch_adult
from alibi.models.tensorflow import HeAE
from alibi.models.tensorflow import Actor, Critic
from alibi.models.tensorflow import ADULTEncoder, ADULTDecoder
from alibi.explainers.cfrl_base import Callback
from alibi.explainers.backends.cfrl_tabular import get_he_preprocessor, get_statistics, \
    get_conditional_vector, apply_category_mapping
```

### Load Adult Census Dataset

```python
# Fetch adult dataset
adult = fetch_adult()

# Separate columns in numerical and categorical.
categorical_names = [adult.feature_names[i] for i in adult.category_map.keys()]
categorical_ids = list(adult.category_map.keys())

numerical_names = [name for i, name in enumerate(adult.feature_names) if i not in adult.category_map.keys()]
numerical_ids = [i for i in range(len(adult.feature_names)) if i not in adult.category_map.keys()]

# Split data into train and test
X, Y = adult.data, adult.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13)
```

### Train black-box classifier

```python
# Define numerical standard scaler.
num_transf = StandardScaler()

# Define categorical one-hot encoder.
cat_transf = OneHotEncoder(
    categories=[range(len(x)) for x in adult.category_map.values()],
    handle_unknown="ignore"
)

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transf, categorical_ids),
        ("num", num_transf, numerical_ids),
    ],
    sparse_threshold=0
)
```

```python
# Fit preprocessor.
preprocessor.fit(X_train)

# Preprocess train and test dataset.
X_train_ohe = preprocessor.transform(X_train)
X_test_ohe = preprocessor.transform(X_test)
```

```python
# Select one of the below classifiers.
# clf = XGBClassifier(min_child_weight=0.5, max_depth=3, gamma=0.2)
# clf = LogisticRegression(C=10)
# clf = DecisionTreeClassifier(max_depth=10, min_samples_split=5)
clf = RandomForestClassifier(max_depth=15, min_samples_split=10, n_estimators=50)

# Fit the classifier.
clf.fit(X_train_ohe, Y_train)
```

```
RandomForestClassifier(max_depth=15, min_samples_split=10, n_estimators=50)
```

### Define the predictor (black-box)

Now that we've trained the classifier, we can define the black-box model. Note that the output of the black-box is a distribution which can be either a soft-label distribution (probabilities/logits for each class) or a hard-label distribution (one-hot encoding). Internally, CFRL takes the `argmax`. Moreover the output **DOES NOT HAVE TO BE DIFFERENTIABLE**.

```python
# Define prediction function.
predictor = lambda x: clf.predict_proba(preprocessor.transform(x))
```

```python
# Compute accuracy.
acc = accuracy_score(y_true=Y_test, y_pred=predictor(X_test).argmax(axis=1))
print("Accuracy: %.3f" % acc)
```

```
Accuracy: 0.864
```

### Define and train autoencoder

Instead of directly modelling the perturbation vector in the potentially high-dimensional input space, we first train an autoencoder. The weights of the encoder are frozen and the actor applies the counterfactual perturbations in the latent space of the encoder. The pre-trained decoder maps the counterfactual embedding back to the input feature space.

The autoencoder follows a standard design. The model is composed from two submodules, the encoder and the decoder. The forward pass consists of passing the input to the encoder, obtain the input embedding and pass the embedding through the decoder.

```python
class HeAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x: tf.Tensor, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
```

The heterogeneous variant used in this example uses an additional type checking to ensure that the output of the decoder is a list of tensors.

Heterogeneous dataset require special treatment. In this work we modeled the numerical features by normal distributions with constant standard deviation and categorical features by categorical distributions. Due to the choice of feature modeling, some numerical features can end up having different types than the original numerical features. For example, a feature like `Age` having the type of `int` can become a `float` due to the autoencoder reconstruction (e.g., `Age=26 -> Age=26.3`). This behavior can be undesirable. Thus we performed casting when process the output of the autoencoder (decoder component).

```python
# Define attribute types, required for datatype conversion.
feature_types = {"Age": int, "Capital Gain": int, "Capital Loss": int, "Hours per week": int}

# Define data preprocessor and inverse preprocessor. The invers preprocessor include datatype conversions.
heae_preprocessor, heae_inv_preprocessor = get_he_preprocessor(X=X_train,
                                                               feature_names=adult.feature_names,
                                                               category_map=adult.category_map,
                                                               feature_types=feature_types)

# Define trainset
trainset_input = heae_preprocessor(X_train).astype(np.float32)
trainset_outputs = {
    "output_1": trainset_input[:, :len(numerical_ids)]
}

for i, cat_id in enumerate(categorical_ids):
    trainset_outputs.update({
        f"output_{i+2}": X_train[:, cat_id]
    })
    
trainset = tf.data.Dataset.from_tensor_slices((trainset_input, trainset_outputs))
trainset = trainset.shuffle(1024).batch(128, drop_remainder=True)
```

```python
# Define autoencoder path and create dir if it doesn't exist.
heae_path = os.path.join("tensorflow", "ADULT_autoencoder")
if not os.path.exists(heae_path):
    os.makedirs(heae_path)

# Define constants.
EPOCHS = 50              # epochs to train the autoencoder
HIDDEN_DIM = 128         # hidden dimension of the autoencoder
LATENT_DIM = 15          # define latent dimension

# Define output dimensions.
OUTPUT_DIMS = [len(numerical_ids)]
OUTPUT_DIMS += [len(adult.category_map[cat_id]) for cat_id in categorical_ids]

# Define the heterogeneous auto-encoder.
heae = HeAE(encoder=ADULTEncoder(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM),
            decoder=ADULTDecoder(hidden_dim=HIDDEN_DIM, output_dims=OUTPUT_DIMS))

# Define loss functions.
he_loss = [keras.losses.MeanSquaredError()]
he_loss_weights = [1.]

# Add categorical losses.
for i in range(len(categorical_names)):
    he_loss.append(keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    he_loss_weights.append(1./len(categorical_names))

# Define metrics.
metrics = {}
for i, cat_name in enumerate(categorical_names):
    metrics.update({f"output_{i+2}": keras.metrics.SparseCategoricalAccuracy()})
    
# Compile model.
heae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
             loss=he_loss,
             loss_weights=he_loss_weights,
             metrics=metrics)

if len(os.listdir(heae_path)) == 0:
    # Fit and save autoencoder.
    heae.fit(trainset, epochs=EPOCHS)
    heae.save(heae_path, save_format="tf")
else:
    # Load the model.
    heae = keras.models.load_model(heae_path, compile=False)
```

```
Epoch 1/50
203/203 [==============================] - 3s 6ms/step - loss: 1.1228 - output_1_loss: 0.2364 - output_2_loss: 1.1212 - output_3_loss: 1.2091 - output_4_loss: 0.6083 - output_5_loss: 1.5602 - output_6_loss: 0.9074 - output_7_loss: 0.6149 - output_8_loss: 0.3439 - output_9_loss: 0.7265 - output_2_sparse_categorical_accuracy: 0.6879 - output_3_sparse_categorical_accuracy: 0.5755 - output_4_sparse_categorical_accuracy: 0.7886 - output_5_sparse_categorical_accuracy: 0.4560 - output_6_sparse_categorical_accuracy: 0.7181 - output_7_sparse_categorical_accuracy: 0.8123 - output_8_sparse_categorical_accuracy: 0.8518 - output_9_sparse_categorical_accuracy: 0.8578
Epoch 2/50
203/203 [==============================] - 1s 6ms/step - loss: 0.4056 - output_1_loss: 0.0395 - output_2_loss: 0.6040 - output_3_loss: 0.5136 - output_4_loss: 0.1736 - output_5_loss: 0.5957 - output_6_loss: 0.3023 - output_7_loss: 0.3132 - output_8_loss: 0.0976 - output_9_loss: 0.3288 - output_2_sparse_categorical_accuracy: 0.7967 - output_3_sparse_categorical_accuracy: 0.8346 - output_4_sparse_categorical_accuracy: 0.9540 - output_5_sparse_categorical_accuracy: 0.8339 - output_6_sparse_categorical_accuracy: 0.9210 - output_7_sparse_categorical_accuracy: 0.8900 - output_8_sparse_categorical_accuracy: 0.9735 - output_9_sparse_categorical_accuracy: 0.9092
Epoch 3/50
203/203 [==============================] - 1s 6ms/step - loss: 0.2299 - output_1_loss: 0.0288 - output_2_loss: 0.2825 - output_3_loss: 0.2869 - output_4_loss: 0.1087 - output_5_loss: 0.2454 - output_6_loss: 0.1982 - output_7_loss: 0.1817 - output_8_loss: 0.0510 - output_9_loss: 0.2541 - output_2_sparse_categorical_accuracy: 0.9271 - output_3_sparse_categorical_accuracy: 0.9168 - output_4_sparse_categorical_accuracy: 0.9739 - output_5_sparse_categorical_accuracy: 0.9474 - output_6_sparse_categorical_accuracy: 0.9444 - output_7_sparse_categorical_accuracy: 0.9486 - output_8_sparse_categorical_accuracy: 0.9892 - output_9_sparse_categorical_accuracy: 0.9231
Epoch 4/50
203/203 [==============================] - 1s 6ms/step - loss: 0.1582 - output_1_loss: 0.0220 - output_2_loss: 0.1704 - output_3_loss: 0.1952 - output_4_loss: 0.0748 - output_5_loss: 0.1633 - output_6_loss: 0.1357 - output_7_loss: 0.1176 - output_8_loss: 0.0324 - output_9_loss: 0.2000 - output_2_sparse_categorical_accuracy: 0.9601 - output_3_sparse_categorical_accuracy: 0.9452 - output_4_sparse_categorical_accuracy: 0.9812 - output_5_sparse_categorical_accuracy: 0.9646 - output_6_sparse_categorical_accuracy: 0.9643 - output_7_sparse_categorical_accuracy: 0.9671 - output_8_sparse_categorical_accuracy: 0.9933 - output_9_sparse_categorical_accuracy: 0.9375
Epoch 5/50
203/203 [==============================] - 1s 6ms/step - loss: 0.1218 - output_1_loss: 0.0175 - output_2_loss: 0.1279 - output_3_loss: 0.1429 - output_4_loss: 0.0577 - output_5_loss: 0.1298 - output_6_loss: 0.0969 - output_7_loss: 0.0897 - output_8_loss: 0.0234 - output_9_loss: 0.1656 - output_2_sparse_categorical_accuracy: 0.9697 - output_3_sparse_categorical_accuracy: 0.9632 - output_4_sparse_categorical_accuracy: 0.9850 - output_5_sparse_categorical_accuracy: 0.9686 - output_6_sparse_categorical_accuracy: 0.9757 - output_7_sparse_categorical_accuracy: 0.9726 - output_8_sparse_categorical_accuracy: 0.9950 - output_9_sparse_categorical_accuracy: 0.9493
Epoch 6/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0986 - output_1_loss: 0.0146 - output_2_loss: 0.1042 - output_3_loss: 0.1073 - output_4_loss: 0.0475 - output_5_loss: 0.1068 - output_6_loss: 0.0741 - output_7_loss: 0.0734 - output_8_loss: 0.0181 - output_9_loss: 0.1410 - output_2_sparse_categorical_accuracy: 0.9757 - output_3_sparse_categorical_accuracy: 0.9747 - output_4_sparse_categorical_accuracy: 0.9878 - output_5_sparse_categorical_accuracy: 0.9736 - output_6_sparse_categorical_accuracy: 0.9820 - output_7_sparse_categorical_accuracy: 0.9776 - output_8_sparse_categorical_accuracy: 0.9959 - output_9_sparse_categorical_accuracy: 0.9582
Epoch 7/50
203/203 [==============================] - 1s 7ms/step - loss: 0.0826 - output_1_loss: 0.0126 - output_2_loss: 0.0886 - output_3_loss: 0.0831 - output_4_loss: 0.0408 - output_5_loss: 0.0893 - output_6_loss: 0.0592 - output_7_loss: 0.0626 - output_8_loss: 0.0147 - output_9_loss: 0.1219 - output_2_sparse_categorical_accuracy: 0.9784 - output_3_sparse_categorical_accuracy: 0.9819 - output_4_sparse_categorical_accuracy: 0.9891 - output_5_sparse_categorical_accuracy: 0.9790 - output_6_sparse_categorical_accuracy: 0.9856 - output_7_sparse_categorical_accuracy: 0.9816 - output_8_sparse_categorical_accuracy: 0.9968 - output_9_sparse_categorical_accuracy: 0.9639
Epoch 8/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0713 - output_1_loss: 0.0112 - output_2_loss: 0.0771 - output_3_loss: 0.0680 - output_4_loss: 0.0359 - output_5_loss: 0.0766 - output_6_loss: 0.0498 - output_7_loss: 0.0533 - output_8_loss: 0.0128 - output_9_loss: 0.1069 - output_2_sparse_categorical_accuracy: 0.9808 - output_3_sparse_categorical_accuracy: 0.9844 - output_4_sparse_categorical_accuracy: 0.9905 - output_5_sparse_categorical_accuracy: 0.9819 - output_6_sparse_categorical_accuracy: 0.9880 - output_7_sparse_categorical_accuracy: 0.9846 - output_8_sparse_categorical_accuracy: 0.9971 - output_9_sparse_categorical_accuracy: 0.9689
Epoch 9/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0629 - output_1_loss: 0.0103 - output_2_loss: 0.0687 - output_3_loss: 0.0577 - output_4_loss: 0.0321 - output_5_loss: 0.0671 - output_6_loss: 0.0434 - output_7_loss: 0.0470 - output_8_loss: 0.0112 - output_9_loss: 0.0938 - output_2_sparse_categorical_accuracy: 0.9826 - output_3_sparse_categorical_accuracy: 0.9869 - output_4_sparse_categorical_accuracy: 0.9921 - output_5_sparse_categorical_accuracy: 0.9841 - output_6_sparse_categorical_accuracy: 0.9899 - output_7_sparse_categorical_accuracy: 0.9868 - output_8_sparse_categorical_accuracy: 0.9976 - output_9_sparse_categorical_accuracy: 0.9738
Epoch 10/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0562 - output_1_loss: 0.0092 - output_2_loss: 0.0626 - output_3_loss: 0.0507 - output_4_loss: 0.0292 - output_5_loss: 0.0592 - output_6_loss: 0.0383 - output_7_loss: 0.0414 - output_8_loss: 0.0099 - output_9_loss: 0.0842 - output_2_sparse_categorical_accuracy: 0.9835 - output_3_sparse_categorical_accuracy: 0.9879 - output_4_sparse_categorical_accuracy: 0.9925 - output_5_sparse_categorical_accuracy: 0.9861 - output_6_sparse_categorical_accuracy: 0.9906 - output_7_sparse_categorical_accuracy: 0.9890 - output_8_sparse_categorical_accuracy: 0.9978 - output_9_sparse_categorical_accuracy: 0.9762
Epoch 11/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0511 - output_1_loss: 0.0086 - output_2_loss: 0.0569 - output_3_loss: 0.0454 - output_4_loss: 0.0266 - output_5_loss: 0.0535 - output_6_loss: 0.0346 - output_7_loss: 0.0372 - output_8_loss: 0.0091 - output_9_loss: 0.0760 - output_2_sparse_categorical_accuracy: 0.9857 - output_3_sparse_categorical_accuracy: 0.9893 - output_4_sparse_categorical_accuracy: 0.9935 - output_5_sparse_categorical_accuracy: 0.9883 - output_6_sparse_categorical_accuracy: 0.9916 - output_7_sparse_categorical_accuracy: 0.9900 - output_8_sparse_categorical_accuracy: 0.9980 - output_9_sparse_categorical_accuracy: 0.9783
Epoch 12/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0465 - output_1_loss: 0.0081 - output_2_loss: 0.0522 - output_3_loss: 0.0406 - output_4_loss: 0.0242 - output_5_loss: 0.0476 - output_6_loss: 0.0319 - output_7_loss: 0.0334 - output_8_loss: 0.0082 - output_9_loss: 0.0687 - output_2_sparse_categorical_accuracy: 0.9865 - output_3_sparse_categorical_accuracy: 0.9906 - output_4_sparse_categorical_accuracy: 0.9945 - output_5_sparse_categorical_accuracy: 0.9890 - output_6_sparse_categorical_accuracy: 0.9925 - output_7_sparse_categorical_accuracy: 0.9913 - output_8_sparse_categorical_accuracy: 0.9984 - output_9_sparse_categorical_accuracy: 0.9803
Epoch 13/50


203/203 [==============================] - 1s 6ms/step - loss: 0.0424 - output_1_loss: 0.0075 - output_2_loss: 0.0477 - output_3_loss: 0.0369 - output_4_loss: 0.0221 - output_5_loss: 0.0431 - output_6_loss: 0.0289 - output_7_loss: 0.0305 - output_8_loss: 0.0076 - output_9_loss: 0.0626 - output_2_sparse_categorical_accuracy: 0.9875 - output_3_sparse_categorical_accuracy: 0.9912 - output_4_sparse_categorical_accuracy: 0.9950 - output_5_sparse_categorical_accuracy: 0.9903 - output_6_sparse_categorical_accuracy: 0.9934 - output_7_sparse_categorical_accuracy: 0.9921 - output_8_sparse_categorical_accuracy: 0.9987 - output_9_sparse_categorical_accuracy: 0.9820
Epoch 14/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0390 - output_1_loss: 0.0069 - output_2_loss: 0.0443 - output_3_loss: 0.0341 - output_4_loss: 0.0205 - output_5_loss: 0.0392 - output_6_loss: 0.0267 - output_7_loss: 0.0277 - output_8_loss: 0.0070 - output_9_loss: 0.0571 - output_2_sparse_categorical_accuracy: 0.9880 - output_3_sparse_categorical_accuracy: 0.9923 - output_4_sparse_categorical_accuracy: 0.9953 - output_5_sparse_categorical_accuracy: 0.9908 - output_6_sparse_categorical_accuracy: 0.9936 - output_7_sparse_categorical_accuracy: 0.9930 - output_8_sparse_categorical_accuracy: 0.9986 - output_9_sparse_categorical_accuracy: 0.9840
Epoch 15/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0362 - output_1_loss: 0.0066 - output_2_loss: 0.0412 - output_3_loss: 0.0316 - output_4_loss: 0.0191 - output_5_loss: 0.0352 - output_6_loss: 0.0249 - output_7_loss: 0.0249 - output_8_loss: 0.0064 - output_9_loss: 0.0531 - output_2_sparse_categorical_accuracy: 0.9891 - output_3_sparse_categorical_accuracy: 0.9922 - output_4_sparse_categorical_accuracy: 0.9957 - output_5_sparse_categorical_accuracy: 0.9919 - output_6_sparse_categorical_accuracy: 0.9942 - output_7_sparse_categorical_accuracy: 0.9937 - output_8_sparse_categorical_accuracy: 0.9987 - output_9_sparse_categorical_accuracy: 0.9852
Epoch 16/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0336 - output_1_loss: 0.0064 - output_2_loss: 0.0380 - output_3_loss: 0.0294 - output_4_loss: 0.0178 - output_5_loss: 0.0322 - output_6_loss: 0.0226 - output_7_loss: 0.0224 - output_8_loss: 0.0060 - output_9_loss: 0.0491 - output_2_sparse_categorical_accuracy: 0.9906 - output_3_sparse_categorical_accuracy: 0.9932 - output_4_sparse_categorical_accuracy: 0.9963 - output_5_sparse_categorical_accuracy: 0.9928 - output_6_sparse_categorical_accuracy: 0.9946 - output_7_sparse_categorical_accuracy: 0.9943 - output_8_sparse_categorical_accuracy: 0.9987 - output_9_sparse_categorical_accuracy: 0.9865
Epoch 17/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0313 - output_1_loss: 0.0062 - output_2_loss: 0.0353 - output_3_loss: 0.0271 - output_4_loss: 0.0166 - output_5_loss: 0.0294 - output_6_loss: 0.0214 - output_7_loss: 0.0205 - output_8_loss: 0.0055 - output_9_loss: 0.0456 - output_2_sparse_categorical_accuracy: 0.9910 - output_3_sparse_categorical_accuracy: 0.9936 - output_4_sparse_categorical_accuracy: 0.9965 - output_5_sparse_categorical_accuracy: 0.9935 - output_6_sparse_categorical_accuracy: 0.9949 - output_7_sparse_categorical_accuracy: 0.9947 - output_8_sparse_categorical_accuracy: 0.9990 - output_9_sparse_categorical_accuracy: 0.9879
Epoch 18/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0292 - output_1_loss: 0.0058 - output_2_loss: 0.0327 - output_3_loss: 0.0253 - output_4_loss: 0.0155 - output_5_loss: 0.0272 - output_6_loss: 0.0198 - output_7_loss: 0.0188 - output_8_loss: 0.0052 - output_9_loss: 0.0428 - output_2_sparse_categorical_accuracy: 0.9913 - output_3_sparse_categorical_accuracy: 0.9939 - output_4_sparse_categorical_accuracy: 0.9968 - output_5_sparse_categorical_accuracy: 0.9938 - output_6_sparse_categorical_accuracy: 0.9955 - output_7_sparse_categorical_accuracy: 0.9955 - output_8_sparse_categorical_accuracy: 0.9989 - output_9_sparse_categorical_accuracy: 0.9888
Epoch 19/50
203/203 [==============================] - 1s 7ms/step - loss: 0.0275 - output_1_loss: 0.0057 - output_2_loss: 0.0303 - output_3_loss: 0.0241 - output_4_loss: 0.0146 - output_5_loss: 0.0249 - output_6_loss: 0.0186 - output_7_loss: 0.0173 - output_8_loss: 0.0047 - output_9_loss: 0.0397 - output_2_sparse_categorical_accuracy: 0.9919 - output_3_sparse_categorical_accuracy: 0.9941 - output_4_sparse_categorical_accuracy: 0.9969 - output_5_sparse_categorical_accuracy: 0.9940 - output_6_sparse_categorical_accuracy: 0.9955 - output_7_sparse_categorical_accuracy: 0.9960 - output_8_sparse_categorical_accuracy: 0.9991 - output_9_sparse_categorical_accuracy: 0.9896
Epoch 20/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0254 - output_1_loss: 0.0054 - output_2_loss: 0.0281 - output_3_loss: 0.0218 - output_4_loss: 0.0136 - output_5_loss: 0.0225 - output_6_loss: 0.0170 - output_7_loss: 0.0156 - output_8_loss: 0.0045 - output_9_loss: 0.0375 - output_2_sparse_categorical_accuracy: 0.9927 - output_3_sparse_categorical_accuracy: 0.9949 - output_4_sparse_categorical_accuracy: 0.9972 - output_5_sparse_categorical_accuracy: 0.9949 - output_6_sparse_categorical_accuracy: 0.9962 - output_7_sparse_categorical_accuracy: 0.9965 - output_8_sparse_categorical_accuracy: 0.9992 - output_9_sparse_categorical_accuracy: 0.9905
Epoch 21/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0239 - output_1_loss: 0.0052 - output_2_loss: 0.0259 - output_3_loss: 0.0204 - output_4_loss: 0.0128 - output_5_loss: 0.0206 - output_6_loss: 0.0158 - output_7_loss: 0.0144 - output_8_loss: 0.0041 - output_9_loss: 0.0352 - output_2_sparse_categorical_accuracy: 0.9934 - output_3_sparse_categorical_accuracy: 0.9952 - output_4_sparse_categorical_accuracy: 0.9974 - output_5_sparse_categorical_accuracy: 0.9955 - output_6_sparse_categorical_accuracy: 0.9962 - output_7_sparse_categorical_accuracy: 0.9967 - output_8_sparse_categorical_accuracy: 0.9992 - output_9_sparse_categorical_accuracy: 0.9910
Epoch 22/50
203/203 [==============================] - ETA: 0s - loss: 0.0228 - output_1_loss: 0.0051 - output_2_loss: 0.0247 - output_3_loss: 0.0194 - output_4_loss: 0.0124 - output_5_loss: 0.0195 - output_6_loss: 0.0149 - output_7_loss: 0.0135 - output_8_loss: 0.0040 - output_9_loss: 0.0327 - output_2_sparse_categorical_accuracy: 0.9938 - output_3_sparse_categorical_accuracy: 0.9951 - output_4_sparse_categorical_accuracy: 0.9975 - output_5_sparse_categorical_accuracy: 0.9957 - output_6_sparse_categorical_accuracy: 0.9966 - output_7_sparse_categorical_accuracy: 0.9970 - output_8_sparse_categorical_accuracy: 0.9990 - output_9_sparse_categorical_accuracy: 0.991 - 1s 6ms/step - loss: 0.0227 - output_1_loss: 0.0051 - output_2_loss: 0.0244 - output_3_loss: 0.0193 - output_4_loss: 0.0124 - output_5_loss: 0.0192 - output_6_loss: 0.0149 - output_7_loss: 0.0133 - output_8_loss: 0.0040 - output_9_loss: 0.0332 - output_2_sparse_categorical_accuracy: 0.9939 - output_3_sparse_categorical_accuracy: 0.9952 - output_4_sparse_categorical_accuracy: 0.9975 - output_5_sparse_categorical_accuracy: 0.9957 - output_6_sparse_categorical_accuracy: 0.9966 - output_7_sparse_categorical_accuracy: 0.9971 - output_8_sparse_categorical_accuracy: 0.9991 - output_9_sparse_categorical_accuracy: 0.9914
Epoch 23/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0212 - output_1_loss: 0.0049 - output_2_loss: 0.0228 - output_3_loss: 0.0179 - output_4_loss: 0.0116 - output_5_loss: 0.0174 - output_6_loss: 0.0141 - output_7_loss: 0.0123 - output_8_loss: 0.0037 - output_9_loss: 0.0311 - output_2_sparse_categorical_accuracy: 0.9940 - output_3_sparse_categorical_accuracy: 0.9958 - output_4_sparse_categorical_accuracy: 0.9976 - output_5_sparse_categorical_accuracy: 0.9964 - output_6_sparse_categorical_accuracy: 0.9969 - output_7_sparse_categorical_accuracy: 0.9974 - output_8_sparse_categorical_accuracy: 0.9993 - output_9_sparse_categorical_accuracy: 0.9920
Epoch 24/50


203/203 [==============================] - 1s 7ms/step - loss: 0.0198 - output_1_loss: 0.0046 - output_2_loss: 0.0209 - output_3_loss: 0.0169 - output_4_loss: 0.0109 - output_5_loss: 0.0163 - output_6_loss: 0.0129 - output_7_loss: 0.0116 - output_8_loss: 0.0036 - output_9_loss: 0.0292 - output_2_sparse_categorical_accuracy: 0.9948 - output_3_sparse_categorical_accuracy: 0.9958 - output_4_sparse_categorical_accuracy: 0.9980 - output_5_sparse_categorical_accuracy: 0.9968 - output_6_sparse_categorical_accuracy: 0.9974 - output_7_sparse_categorical_accuracy: 0.9975 - output_8_sparse_categorical_accuracy: 0.9992 - output_9_sparse_categorical_accuracy: 0.9928
Epoch 25/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0190 - output_1_loss: 0.0046 - output_2_loss: 0.0197 - output_3_loss: 0.0157 - output_4_loss: 0.0107 - output_5_loss: 0.0154 - output_6_loss: 0.0123 - output_7_loss: 0.0106 - output_8_loss: 0.0031 - output_9_loss: 0.0278 - output_2_sparse_categorical_accuracy: 0.9950 - output_3_sparse_categorical_accuracy: 0.9960 - output_4_sparse_categorical_accuracy: 0.9979 - output_5_sparse_categorical_accuracy: 0.9968 - output_6_sparse_categorical_accuracy: 0.9973 - output_7_sparse_categorical_accuracy: 0.9978 - output_8_sparse_categorical_accuracy: 0.9994 - output_9_sparse_categorical_accuracy: 0.9930
Epoch 26/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0178 - output_1_loss: 0.0043 - output_2_loss: 0.0186 - output_3_loss: 0.0148 - output_4_loss: 0.0099 - output_5_loss: 0.0139 - output_6_loss: 0.0114 - output_7_loss: 0.0100 - output_8_loss: 0.0032 - output_9_loss: 0.0260 - output_2_sparse_categorical_accuracy: 0.9955 - output_3_sparse_categorical_accuracy: 0.9963 - output_4_sparse_categorical_accuracy: 0.9982 - output_5_sparse_categorical_accuracy: 0.9971 - output_6_sparse_categorical_accuracy: 0.9977 - output_7_sparse_categorical_accuracy: 0.9978 - output_8_sparse_categorical_accuracy: 0.9995 - output_9_sparse_categorical_accuracy: 0.9936
Epoch 27/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0167 - output_1_loss: 0.0041 - output_2_loss: 0.0171 - output_3_loss: 0.0137 - output_4_loss: 0.0092 - output_5_loss: 0.0131 - output_6_loss: 0.0109 - output_7_loss: 0.0096 - output_8_loss: 0.0028 - output_9_loss: 0.0245 - output_2_sparse_categorical_accuracy: 0.9960 - output_3_sparse_categorical_accuracy: 0.9968 - output_4_sparse_categorical_accuracy: 0.9984 - output_5_sparse_categorical_accuracy: 0.9974 - output_6_sparse_categorical_accuracy: 0.9980 - output_7_sparse_categorical_accuracy: 0.9980 - output_8_sparse_categorical_accuracy: 0.9995 - output_9_sparse_categorical_accuracy: 0.9938
Epoch 28/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0160 - output_1_loss: 0.0041 - output_2_loss: 0.0164 - output_3_loss: 0.0130 - output_4_loss: 0.0091 - output_5_loss: 0.0124 - output_6_loss: 0.0098 - output_7_loss: 0.0090 - output_8_loss: 0.0028 - output_9_loss: 0.0230 - output_2_sparse_categorical_accuracy: 0.9964 - output_3_sparse_categorical_accuracy: 0.9972 - output_4_sparse_categorical_accuracy: 0.9983 - output_5_sparse_categorical_accuracy: 0.9980 - output_6_sparse_categorical_accuracy: 0.9982 - output_7_sparse_categorical_accuracy: 0.9981 - output_8_sparse_categorical_accuracy: 0.9994 - output_9_sparse_categorical_accuracy: 0.9944
Epoch 29/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0150 - output_1_loss: 0.0039 - output_2_loss: 0.0153 - output_3_loss: 0.0120 - output_4_loss: 0.0083 - output_5_loss: 0.0114 - output_6_loss: 0.0093 - output_7_loss: 0.0085 - output_8_loss: 0.0023 - output_9_loss: 0.0217 - output_2_sparse_categorical_accuracy: 0.9966 - output_3_sparse_categorical_accuracy: 0.9974 - output_4_sparse_categorical_accuracy: 0.9982 - output_5_sparse_categorical_accuracy: 0.9980 - output_6_sparse_categorical_accuracy: 0.9981 - output_7_sparse_categorical_accuracy: 0.9981 - output_8_sparse_categorical_accuracy: 0.9996 - output_9_sparse_categorical_accuracy: 0.9946
Epoch 30/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0142 - output_1_loss: 0.0038 - output_2_loss: 0.0144 - output_3_loss: 0.0111 - output_4_loss: 0.0080 - output_5_loss: 0.0107 - output_6_loss: 0.0087 - output_7_loss: 0.0079 - output_8_loss: 0.0023 - output_9_loss: 0.0206 - output_2_sparse_categorical_accuracy: 0.9967 - output_3_sparse_categorical_accuracy: 0.9975 - output_4_sparse_categorical_accuracy: 0.9986 - output_5_sparse_categorical_accuracy: 0.9981 - output_6_sparse_categorical_accuracy: 0.9984 - output_7_sparse_categorical_accuracy: 0.9984 - output_8_sparse_categorical_accuracy: 0.9997 - output_9_sparse_categorical_accuracy: 0.9949
Epoch 31/50
203/203 [==============================] - 1s 7ms/step - loss: 0.0139 - output_1_loss: 0.0039 - output_2_loss: 0.0135 - output_3_loss: 0.0109 - output_4_loss: 0.0079 - output_5_loss: 0.0103 - output_6_loss: 0.0084 - output_7_loss: 0.0076 - output_8_loss: 0.0022 - output_9_loss: 0.0194 - output_2_sparse_categorical_accuracy: 0.9972 - output_3_sparse_categorical_accuracy: 0.9979 - output_4_sparse_categorical_accuracy: 0.9985 - output_5_sparse_categorical_accuracy: 0.9980 - output_6_sparse_categorical_accuracy: 0.9984 - output_7_sparse_categorical_accuracy: 0.9984 - output_8_sparse_categorical_accuracy: 0.9996 - output_9_sparse_categorical_accuracy: 0.9957
Epoch 32/50
203/203 [==============================] - 1s 7ms/step - loss: 0.0135 - output_1_loss: 0.0037 - output_2_loss: 0.0132 - output_3_loss: 0.0105 - output_4_loss: 0.0079 - output_5_loss: 0.0100 - output_6_loss: 0.0089 - output_7_loss: 0.0073 - output_8_loss: 0.0022 - output_9_loss: 0.0186 - output_2_sparse_categorical_accuracy: 0.9972 - output_3_sparse_categorical_accuracy: 0.9977 - output_4_sparse_categorical_accuracy: 0.9981 - output_5_sparse_categorical_accuracy: 0.9986 - output_6_sparse_categorical_accuracy: 0.9982 - output_7_sparse_categorical_accuracy: 0.9986 - output_8_sparse_categorical_accuracy: 0.9995 - output_9_sparse_categorical_accuracy: 0.9958
Epoch 33/50
203/203 [==============================] - 1s 7ms/step - loss: 0.0124 - output_1_loss: 0.0035 - output_2_loss: 0.0120 - output_3_loss: 0.0094 - output_4_loss: 0.0069 - output_5_loss: 0.0089 - output_6_loss: 0.0071 - output_7_loss: 0.0071 - output_8_loss: 0.0019 - output_9_loss: 0.0176 - output_2_sparse_categorical_accuracy: 0.9974 - output_3_sparse_categorical_accuracy: 0.9980 - output_4_sparse_categorical_accuracy: 0.9986 - output_5_sparse_categorical_accuracy: 0.9987 - output_6_sparse_categorical_accuracy: 0.9987 - output_7_sparse_categorical_accuracy: 0.9988 - output_8_sparse_categorical_accuracy: 0.9997 - output_9_sparse_categorical_accuracy: 0.9959
Epoch 34/50
203/203 [==============================] - 1s 7ms/step - loss: 0.0128 - output_1_loss: 0.0037 - output_2_loss: 0.0117 - output_3_loss: 0.0095 - output_4_loss: 0.0078 - output_5_loss: 0.0091 - output_6_loss: 0.0094 - output_7_loss: 0.0068 - output_8_loss: 0.0022 - output_9_loss: 0.0166 - output_2_sparse_categorical_accuracy: 0.9976 - output_3_sparse_categorical_accuracy: 0.9981 - output_4_sparse_categorical_accuracy: 0.9985 - output_5_sparse_categorical_accuracy: 0.9985 - output_6_sparse_categorical_accuracy: 0.9985 - output_7_sparse_categorical_accuracy: 0.9987 - output_8_sparse_categorical_accuracy: 0.9995 - output_9_sparse_categorical_accuracy: 0.9962
Epoch 35/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0111 - output_1_loss: 0.0032 - output_2_loss: 0.0108 - output_3_loss: 0.0082 - output_4_loss: 0.0062 - output_5_loss: 0.0079 - output_6_loss: 0.0064 - output_7_loss: 0.0063 - output_8_loss: 0.0017 - output_9_loss: 0.0156 - output_2_sparse_categorical_accuracy: 0.9977 - output_3_sparse_categorical_accuracy: 0.9983 - output_4_sparse_categorical_accuracy: 0.9988 - output_5_sparse_categorical_accuracy: 0.9990 - output_6_sparse_categorical_accuracy: 0.9991 - output_7_sparse_categorical_accuracy: 0.9987 - output_8_sparse_categorical_accuracy: 0.9999 - output_9_sparse_categorical_accuracy: 0.9966
Epoch 36/50


203/203 [==============================] - 1s 6ms/step - loss: 0.0108 - output_1_loss: 0.0033 - output_2_loss: 0.0103 - output_3_loss: 0.0078 - output_4_loss: 0.0059 - output_5_loss: 0.0077 - output_6_loss: 0.0061 - output_7_loss: 0.0059 - output_8_loss: 0.0016 - output_9_loss: 0.0147 - output_2_sparse_categorical_accuracy: 0.9981 - output_3_sparse_categorical_accuracy: 0.9987 - output_4_sparse_categorical_accuracy: 0.9991 - output_5_sparse_categorical_accuracy: 0.9989 - output_6_sparse_categorical_accuracy: 0.9992 - output_7_sparse_categorical_accuracy: 0.9989 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9970
Epoch 37/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0102 - output_1_loss: 0.0031 - output_2_loss: 0.0096 - output_3_loss: 0.0075 - output_4_loss: 0.0058 - output_5_loss: 0.0072 - output_6_loss: 0.0058 - output_7_loss: 0.0056 - output_8_loss: 0.0016 - output_9_loss: 0.0139 - output_2_sparse_categorical_accuracy: 0.9982 - output_3_sparse_categorical_accuracy: 0.9987 - output_4_sparse_categorical_accuracy: 0.9988 - output_5_sparse_categorical_accuracy: 0.9990 - output_6_sparse_categorical_accuracy: 0.9990 - output_7_sparse_categorical_accuracy: 0.9991 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9973
Epoch 38/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0098 - output_1_loss: 0.0030 - output_2_loss: 0.0091 - output_3_loss: 0.0071 - output_4_loss: 0.0055 - output_5_loss: 0.0070 - output_6_loss: 0.0053 - output_7_loss: 0.0054 - output_8_loss: 0.0015 - output_9_loss: 0.0131 - output_2_sparse_categorical_accuracy: 0.9985 - output_3_sparse_categorical_accuracy: 0.9988 - output_4_sparse_categorical_accuracy: 0.9990 - output_5_sparse_categorical_accuracy: 0.9990 - output_6_sparse_categorical_accuracy: 0.9992 - output_7_sparse_categorical_accuracy: 0.9991 - output_8_sparse_categorical_accuracy: 0.9999 - output_9_sparse_categorical_accuracy: 0.9972
Epoch 39/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0095 - output_1_loss: 0.0031 - output_2_loss: 0.0088 - output_3_loss: 0.0066 - output_4_loss: 0.0050 - output_5_loss: 0.0067 - output_6_loss: 0.0051 - output_7_loss: 0.0054 - output_8_loss: 0.0014 - output_9_loss: 0.0126 - output_2_sparse_categorical_accuracy: 0.9984 - output_3_sparse_categorical_accuracy: 0.9990 - output_4_sparse_categorical_accuracy: 0.9993 - output_5_sparse_categorical_accuracy: 0.9990 - output_6_sparse_categorical_accuracy: 0.9993 - output_7_sparse_categorical_accuracy: 0.9990 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9975
Epoch 40/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0091 - output_1_loss: 0.0029 - output_2_loss: 0.0083 - output_3_loss: 0.0066 - output_4_loss: 0.0050 - output_5_loss: 0.0062 - output_6_loss: 0.0049 - output_7_loss: 0.0051 - output_8_loss: 0.0014 - output_9_loss: 0.0121 - output_2_sparse_categorical_accuracy: 0.9985 - output_3_sparse_categorical_accuracy: 0.9990 - output_4_sparse_categorical_accuracy: 0.9992 - output_5_sparse_categorical_accuracy: 0.9992 - output_6_sparse_categorical_accuracy: 0.9993 - output_7_sparse_categorical_accuracy: 0.9990 - output_8_sparse_categorical_accuracy: 0.9997 - output_9_sparse_categorical_accuracy: 0.9975
Epoch 41/50
203/203 [==============================] - 1s 7ms/step - loss: 0.0089 - output_1_loss: 0.0029 - output_2_loss: 0.0079 - output_3_loss: 0.0064 - output_4_loss: 0.0049 - output_5_loss: 0.0063 - output_6_loss: 0.0047 - output_7_loss: 0.0049 - output_8_loss: 0.0013 - output_9_loss: 0.0115 - output_2_sparse_categorical_accuracy: 0.9986 - output_3_sparse_categorical_accuracy: 0.9990 - output_4_sparse_categorical_accuracy: 0.9992 - output_5_sparse_categorical_accuracy: 0.9991 - output_6_sparse_categorical_accuracy: 0.9994 - output_7_sparse_categorical_accuracy: 0.9991 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9978
Epoch 42/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0082 - output_1_loss: 0.0026 - output_2_loss: 0.0075 - output_3_loss: 0.0057 - output_4_loss: 0.0043 - output_5_loss: 0.0057 - output_6_loss: 0.0045 - output_7_loss: 0.0045 - output_8_loss: 0.0013 - output_9_loss: 0.0108 - output_2_sparse_categorical_accuracy: 0.9988 - output_3_sparse_categorical_accuracy: 0.9992 - output_4_sparse_categorical_accuracy: 0.9992 - output_5_sparse_categorical_accuracy: 0.9994 - output_6_sparse_categorical_accuracy: 0.9994 - output_7_sparse_categorical_accuracy: 0.9992 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9981
Epoch 43/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0080 - output_1_loss: 0.0027 - output_2_loss: 0.0070 - output_3_loss: 0.0055 - output_4_loss: 0.0042 - output_5_loss: 0.0054 - output_6_loss: 0.0042 - output_7_loss: 0.0044 - output_8_loss: 0.0011 - output_9_loss: 0.0103 - output_2_sparse_categorical_accuracy: 0.9989 - output_3_sparse_categorical_accuracy: 0.9992 - output_4_sparse_categorical_accuracy: 0.9993 - output_5_sparse_categorical_accuracy: 0.9994 - output_6_sparse_categorical_accuracy: 0.9997 - output_7_sparse_categorical_accuracy: 0.9992 - output_8_sparse_categorical_accuracy: 0.9999 - output_9_sparse_categorical_accuracy: 0.9981
Epoch 44/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0077 - output_1_loss: 0.0027 - output_2_loss: 0.0067 - output_3_loss: 0.0052 - output_4_loss: 0.0039 - output_5_loss: 0.0053 - output_6_loss: 0.0040 - output_7_loss: 0.0043 - output_8_loss: 0.0012 - output_9_loss: 0.0098 - output_2_sparse_categorical_accuracy: 0.9990 - output_3_sparse_categorical_accuracy: 0.9992 - output_4_sparse_categorical_accuracy: 0.9992 - output_5_sparse_categorical_accuracy: 0.9994 - output_6_sparse_categorical_accuracy: 0.9995 - output_7_sparse_categorical_accuracy: 0.9993 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9982
Epoch 45/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0074 - output_1_loss: 0.0026 - output_2_loss: 0.0066 - output_3_loss: 0.0051 - output_4_loss: 0.0038 - output_5_loss: 0.0049 - output_6_loss: 0.0037 - output_7_loss: 0.0042 - output_8_loss: 0.0010 - output_9_loss: 0.0094 - output_2_sparse_categorical_accuracy: 0.9990 - output_3_sparse_categorical_accuracy: 0.9992 - output_4_sparse_categorical_accuracy: 0.9994 - output_5_sparse_categorical_accuracy: 0.9995 - output_6_sparse_categorical_accuracy: 0.9997 - output_7_sparse_categorical_accuracy: 0.9992 - output_8_sparse_categorical_accuracy: 0.9999 - output_9_sparse_categorical_accuracy: 0.9984
Epoch 46/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0075 - output_1_loss: 0.0027 - output_2_loss: 0.0064 - output_3_loss: 0.0050 - output_4_loss: 0.0039 - output_5_loss: 0.0050 - output_6_loss: 0.0041 - output_7_loss: 0.0040 - output_8_loss: 0.0010 - output_9_loss: 0.0091 - output_2_sparse_categorical_accuracy: 0.9989 - output_3_sparse_categorical_accuracy: 0.9993 - output_4_sparse_categorical_accuracy: 0.9993 - output_5_sparse_categorical_accuracy: 0.9994 - output_6_sparse_categorical_accuracy: 0.9994 - output_7_sparse_categorical_accuracy: 0.9992 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9984
Epoch 47/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0072 - output_1_loss: 0.0026 - output_2_loss: 0.0060 - output_3_loss: 0.0048 - output_4_loss: 0.0040 - output_5_loss: 0.0049 - output_6_loss: 0.0038 - output_7_loss: 0.0038 - output_8_loss: 0.0010 - output_9_loss: 0.0088 - output_2_sparse_categorical_accuracy: 0.9991 - output_3_sparse_categorical_accuracy: 0.9993 - output_4_sparse_categorical_accuracy: 0.9992 - output_5_sparse_categorical_accuracy: 0.9994 - output_6_sparse_categorical_accuracy: 0.9996 - output_7_sparse_categorical_accuracy: 0.9994 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9986
Epoch 48/50


203/203 [==============================] - 1s 7ms/step - loss: 0.0069 - output_1_loss: 0.0025 - output_2_loss: 0.0060 - output_3_loss: 0.0047 - output_4_loss: 0.0035 - output_5_loss: 0.0046 - output_6_loss: 0.0036 - output_7_loss: 0.0037 - output_8_loss: 0.0013 - output_9_loss: 0.0086 - output_2_sparse_categorical_accuracy: 0.9991 - output_3_sparse_categorical_accuracy: 0.9993 - output_4_sparse_categorical_accuracy: 0.9993 - output_5_sparse_categorical_accuracy: 0.9995 - output_6_sparse_categorical_accuracy: 0.9994 - output_7_sparse_categorical_accuracy: 0.9994 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9985
Epoch 49/50
203/203 [==============================] - 1s 7ms/step - loss: 0.0065 - output_1_loss: 0.0023 - output_2_loss: 0.0054 - output_3_loss: 0.0042 - output_4_loss: 0.0031 - output_5_loss: 0.0043 - output_6_loss: 0.0034 - output_7_loss: 0.0039 - output_8_loss: 9.7766e-04 - output_9_loss: 0.0082 - output_2_sparse_categorical_accuracy: 0.9992 - output_3_sparse_categorical_accuracy: 0.9994 - output_4_sparse_categorical_accuracy: 0.9996 - output_5_sparse_categorical_accuracy: 0.9995 - output_6_sparse_categorical_accuracy: 0.9997 - output_7_sparse_categorical_accuracy: 0.9993 - output_8_sparse_categorical_accuracy: 0.9998 - output_9_sparse_categorical_accuracy: 0.9985
Epoch 50/50
203/203 [==============================] - 1s 6ms/step - loss: 0.0066 - output_1_loss: 0.0025 - output_2_loss: 0.0053 - output_3_loss: 0.0043 - output_4_loss: 0.0033 - output_5_loss: 0.0044 - output_6_loss: 0.0032 - output_7_loss: 0.0036 - output_8_loss: 9.2933e-04 - output_9_loss: 0.0075 - output_2_sparse_categorical_accuracy: 0.9993 - output_3_sparse_categorical_accuracy: 0.9993 - output_4_sparse_categorical_accuracy: 0.9993 - output_5_sparse_categorical_accuracy: 0.9995 - output_6_sparse_categorical_accuracy: 0.9996 - output_7_sparse_categorical_accuracy: 0.9994 - output_8_sparse_categorical_accuracy: 0.9999 - output_9_sparse_categorical_accuracy: 0.9986
INFO:tensorflow:Assets written to: tensorflow/ADULT_autoencoder/assets
```

### Counterfactual with Reinforcement Learning

```python
# Define constants
COEFF_SPARSITY = 0.5               # sparisty coefficient
COEFF_CONSISTENCY = 0.5            # consisteny coefficient
TRAIN_STEPS = 10000                # number of training steps -> consider increasing the number of steps
BATCH_SIZE = 100                   # batch size
```

#### Define dataset specific attributes and constraints

A desirable property of a method for generating counterfactuals is to allow feature conditioning. Real-world datasets usually include immutable features such as `Sex` or `Race`, which should remain unchanged throughout the counterfactual search procedure. Similarly, a numerical feature such as `Age` should only increase for a counterfactual to be actionable.

```python
# Define immutable features.
immutable_features = ['Marital Status', 'Relationship', 'Race', 'Sex']

# Define ranges. This means that the `Age` feature can not decrease.
ranges = {'Age': [0.0, 1.0]}
```

#### Define and fit the explainer

```python
explainer = CounterfactualRLTabular(predictor=predictor,
                                    encoder=heae.encoder,
                                    decoder=heae.decoder,
                                    latent_dim=LATENT_DIM,
                                    encoder_preprocessor=heae_preprocessor,
                                    decoder_inv_preprocessor=heae_inv_preprocessor,
                                    coeff_sparsity=COEFF_SPARSITY,
                                    coeff_consistency=COEFF_CONSISTENCY,
                                    category_map=adult.category_map,
                                    feature_names=adult.feature_names,
                                    ranges=ranges,
                                    immutable_features=immutable_features,
                                    train_steps=TRAIN_STEPS,
                                    batch_size=BATCH_SIZE,
                                    backend="tensorflow")
```

```python
# Fit the explainer.
explainer = explainer.fit(X=X_train)
```

```
100%|██████████| 10000/10000 [06:37<00:00, 25.17it/s]
```

#### Test explainer

```python
# Select some positive examples.
X_positive = X_test[np.argmax(predictor(X_test), axis=1) == 1]

X = X_positive[:1000]
Y_t = np.array([0])
C = [{"Age": [0, 20], "Workclass": ["State-gov", "?", "Local-gov"]}]
```

```python
# Generate counterfactual instances.
explanation = explainer.explain(X, Y_t, C)
```

```
100%|██████████| 10/10 [00:00<00:00, 34.63it/s]
```

```python
# Concat labels to the original instances.
orig = np.concatenate(
    [explanation.data['orig']['X'], explanation.data['orig']['class']],
    axis=1
)

# Concat labels to the counterfactual instances.
cf = np.concatenate(
    [explanation.data['cf']['X'], explanation.data['cf']['class']],
    axis=1
)

# Define new feature names and category map by including the label.
feature_names = adult.feature_names + ["Label"]
category_map = deepcopy(adult.category_map)
category_map.update({feature_names.index("Label"): adult.target_names})

# Replace label encodings with strings.
orig_pd = pd.DataFrame(
    apply_category_mapping(orig, category_map),
    columns=feature_names
)

cf_pd = pd.DataFrame(
    apply_category_mapping(cf, category_map),
    columns=feature_names
)
```

```python
orig_pd.head(n=10)
```

|   | Age | Workclass    | Education        | Marital Status | Occupation   | Relationship  | Race  | Sex    | Capital Gain | Capital Loss | Hours per week | Country       | Label |
| - | --- | ------------ | ---------------- | -------------- | ------------ | ------------- | ----- | ------ | ------------ | ------------ | -------------- | ------------- | ----- |
| 0 | 60  | Private      | High School grad | Married        | Blue-Collar  | Husband       | White | Male   | 7298         | 0            | 40             | United-States | >50K  |
| 1 | 35  | Private      | High School grad | Married        | White-Collar | Husband       | White | Male   | 7688         | 0            | 50             | United-States | >50K  |
| 2 | 39  | State-gov    | Masters          | Married        | Professional | Wife          | White | Female | 5178         | 0            | 38             | United-States | >50K  |
| 3 | 44  | Self-emp-inc | High School grad | Married        | Sales        | Husband       | White | Male   | 0            | 0            | 50             | United-States | >50K  |
| 4 | 39  | Private      | Bachelors        | Separated      | White-Collar | Not-in-family | White | Female | 13550        | 0            | 50             | United-States | >50K  |
| 5 | 45  | Private      | High School grad | Married        | Blue-Collar  | Husband       | White | Male   | 0            | 1902         | 40             | ?             | >50K  |
| 6 | 50  | Private      | Bachelors        | Married        | Professional | Husband       | White | Male   | 0            | 0            | 50             | United-States | >50K  |
| 7 | 29  | Private      | Bachelors        | Married        | White-Collar | Wife          | White | Female | 0            | 0            | 50             | United-States | >50K  |
| 8 | 47  | Private      | Bachelors        | Married        | Professional | Husband       | White | Male   | 0            | 0            | 50             | United-States | >50K  |
| 9 | 35  | Private      | Bachelors        | Married        | White-Collar | Husband       | White | Male   | 0            | 0            | 70             | United-States | >50K  |

```python
cf_pd.head(n=10)
```

|   | Age | Workclass    | Education        | Marital Status | Occupation   | Relationship  | Race  | Sex    | Capital Gain | Capital Loss | Hours per week | Country       | Label |
| - | --- | ------------ | ---------------- | -------------- | ------------ | ------------- | ----- | ------ | ------------ | ------------ | -------------- | ------------- | ----- |
| 0 | 60  | Private      | High School grad | Married        | Blue-Collar  | Husband       | White | Male   | 320          | 0            | 40             | United-States | <=50K |
| 1 | 35  | Private      | Dropout          | Married        | Blue-Collar  | Husband       | White | Male   | 125          | 0            | 50             | United-States | <=50K |
| 2 | 39  | State-gov    | Dropout          | Married        | Service      | Wife          | White | Female | 538          | 15           | 39             | United-States | <=50K |
| 3 | 44  | Self-emp-inc | High School grad | Married        | Sales        | Husband       | White | Male   | 0            | 0            | 50             | United-States | >50K  |
| 4 | 39  | Private      | Bachelors        | Separated      | White-Collar | Not-in-family | White | Female | 1922         | 0            | 51             | United-States | <=50K |
| 5 | 45  | Private      | High School grad | Married        | Blue-Collar  | Husband       | White | Male   | 0            | 1900         | 41             | Latin-America | >50K  |
| 6 | 50  | Private      | Dropout          | Married        | Service      | Husband       | White | Male   | 0            | 0            | 51             | United-States | <=50K |
| 7 | 29  | Private      | Dropout          | Married        | Sales        | Wife          | White | Female | 0            | 0            | 50             | United-States | <=50K |
| 8 | 47  | Private      | Dropout          | Married        | Service      | Husband       | White | Male   | 0            | 0            | 51             | United-States | <=50K |
| 9 | 35  | Private      | Dropout          | Married        | Sales        | Husband       | White | Male   | 0            | 0            | 71             | United-States | <=50K |

#### Diversity

```python
# Generate counterfactual instances.
X = X_positive[0].reshape(1, -1)
explanation = explainer.explain(X=X, Y_t=Y_t, C=C, diversity=True, num_samples=100, batch_size=10)
```

```
12it [00:00, 26.20it/s]
```

```python
# Concat label column.
orig = np.concatenate(
    [explanation.data['orig']['X'], explanation.data['orig']['class']],
    axis=1
)

cf = np.concatenate(
    [explanation.data['cf']['X'], explanation.data['cf']['class']],
    axis=1
)

# Transfrom label encodings to string.
orig_pd = pd.DataFrame(
    apply_category_mapping(orig, category_map),
    columns=feature_names,
)

cf_pd = pd.DataFrame(
    apply_category_mapping(cf, category_map),
    columns=feature_names,
)
```

```python
orig_pd.head(n=5)
```

|   | Age | Workclass | Education        | Marital Status | Occupation  | Relationship | Race  | Sex  | Capital Gain | Capital Loss | Hours per week | Country       | Label |
| - | --- | --------- | ---------------- | -------------- | ----------- | ------------ | ----- | ---- | ------------ | ------------ | -------------- | ------------- | ----- |
| 0 | 60  | Private   | High School grad | Married        | Blue-Collar | Husband      | White | Male | 7298         | 0            | 40             | United-States | >50K  |

```python
cf_pd.head(n=5)
```

|   | Age | Workclass | Education        | Marital Status | Occupation  | Relationship | Race  | Sex  | Capital Gain | Capital Loss | Hours per week | Country       | Label |
| - | --- | --------- | ---------------- | -------------- | ----------- | ------------ | ----- | ---- | ------------ | ------------ | -------------- | ------------- | ----- |
| 0 | 60  | Private   | Dropout          | Married        | Blue-Collar | Husband      | White | Male | 143          | 0            | 40             | United-States | <=50K |
| 1 | 60  | Private   | High School grad | Married        | Blue-Collar | Husband      | White | Male | 49           | 0            | 40             | United-States | <=50K |
| 2 | 60  | Private   | High School grad | Married        | Blue-Collar | Husband      | White | Male | 84           | 0            | 40             | United-States | <=50K |
| 3 | 60  | Private   | High School grad | Married        | Blue-Collar | Husband      | White | Male | 87           | 0            | 41             | United-States | <=50K |
| 4 | 60  | Private   | High School grad | Married        | Blue-Collar | Husband      | White | Male | 97           | 0            | 40             | United-States | <=50K |

### Logging

Logging is clearly important when dealing with deep learning models. Thus, we provide an interface to write custom callbacks for logging purposes after each training step which we defined [here](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/api/alibi.explainers.cfrl_base.rst#alibi.explainers.cfrl_base.Callback). In the following cells we provide some example to log in **Weights and Biases**.

#### Logging reward callback

```python
class RewardCallback(Callback):
    def __call__(self,
                 step: int, 
                 update: int, 
                 model: CounterfactualRL,
                 sample: Dict[str, np.ndarray],
                 losses: Dict[str, float]):
        
        if (step + update) % 100 != 0:
            return
        
        # get the counterfactual and target
        Y_t = sample["Y_t"]
        X_cf = model.params["decoder_inv_preprocessor"](sample["X_cf"])
        
        # get prediction label
        Y_m_cf = predictor(X_cf)
        
        # compute reward
        reward = np.mean(model.params["reward_func"](Y_m_cf, Y_t))
        wandb.log({"reward": reward})
```

#### Logging losses callback

```python
class LossCallback(Callback):
    def __call__(self,
                 step: int, 
                 update: int, 
                 model: CounterfactualRL,
                 sample: Dict[str, np.ndarray],
                 losses: Dict[str, float]):
        # Log training losses.
        if (step + update) % 100 == 0:
            wandb.log(losses)
```

#### Logging tables callback

```python
class TablesCallback(Callback):
    def __call__(self,
                 step: int, 
                 update: int, 
                 model: CounterfactualRL,
                 sample: Dict[str, np.ndarray],
                 losses: Dict[str, float]):
        # Log every 1000 steps
        if step % 1000 != 0:
            return
        
        # Define number of samples to be displayed.
        NUM_SAMPLES = 5
        
        X = heae_inv_preprocessor(sample["X"][:NUM_SAMPLES])        # input instance
        X_cf = heae_inv_preprocessor(sample["X_cf"][:NUM_SAMPLES])  # counterfactual
        
        Y_m = np.argmax(sample["Y_m"][:NUM_SAMPLES], axis=1).astype(int).reshape(-1, 1) # input labels
        Y_t = np.argmax(sample["Y_t"][:NUM_SAMPLES], axis=1).astype(int).reshape(-1, 1) # target labels
        Y_m_cf = np.argmax(predictor(X_cf), axis=1).astype(int).reshape(-1, 1)          # counterfactual labels
        
        # Define feature names and category map for input.
        feature_names = adult.feature_names + ["Label"]
        category_map = deepcopy(adult.category_map)
        category_map.update({feature_names.index("Label"): adult.target_names})
        
        # Construct input array.
        inputs = np.concatenate([X, Y_m], axis=1)
        inputs = pd.DataFrame(apply_category_mapping(inputs, category_map),
                              columns=feature_names)
        
        # Define feature names and category map for counterfactual output.
        feature_names += ["Target"]
        category_map.update({feature_names.index("Target"): adult.target_names})
        
        # Construct output array.
        outputs = np.concatenate([X_cf, Y_m_cf, Y_t], axis=1)
        outputs = pd.DataFrame(apply_category_mapping(outputs, category_map),
                               columns=feature_names)
        
        # Log table.
        wandb.log({
            "Input": wandb.Table(dataframe=inputs),
            "Output": wandb.Table(dataframe=outputs)
        })
```

Having defined the callbacks, we can define a new explainer that will include logging.

```python
import wandb

# Initialize wandb.
wandb_project = "Adult Census Counterfactual with Reinforcement Learning"
wandb.init(project=wandb_project)

# Define explainer as before and include callbacks.
explainer = CounterfactualRLTabular(...,
                                    callbacks=[LossCallback(), RewardCallback(), TablesCallback()])

# Fit the explainers.
explainer = explainer.fit(X=X_train)

# Close wandb.
wandb.finish()
```
