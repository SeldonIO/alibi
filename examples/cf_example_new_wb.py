# flake8 noqa
import tensorflow as tf
import logging

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np
from alibi.explainers.experimental.counterfactuals import WachterCounterfactual
from typing_extensions import Final

from timeit import default_timer as timer
from tensorflow.keras.optimizers import Adam

# Load and prepare data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)

xmin, xmax = -.5, .5
x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin

# Load model

cnn = load_model('mnist_cnn.h5')
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])

X = x_test[0].reshape((1,) + x_test[0].shape)

# Initialise explainer
logger = logging.getLogger(__name__)
shape = (1,) + x_train.shape[1:]
target_proba = 1.0
target_class = 'other'  # type: Final  # any class other than 7 will do
max_iter = 1000
lam_init = 1e-1
max_lam_steps = 10
learning_rate_init = 0.1
feature_range = (x_train.min(), x_train.max())

# method_opts = {'tol': 0.35}  # want counterfactuals with p(class)>0.99

# TESTS TO DO: PASS OPTIMIZER IN VARIOUS WAYS, VARIOUS METHOD OPTS
# OVERRIDE VARIOUS ATTRIBUTES
optimizer = Adam
optimizer_opts = {'learning_rate': 0.1}
# method_opts = {'lam_opts': {'max_lam_steps': 2}}

cf = WachterCounterfactual(cnn, predictor_type='whitebox')  # , method_opts=method_opts)
logging_opts = {'log_traces': True, 'trace_dir': 'logs/wb_watcher_public_class_final'}
t_start = timer()
cf.explain(X, target_class, optimizer=optimizer, optimizer_opts=optimizer_opts, logging_opts=logging_opts)
t_elapsed = timer() - t_start
print(t_elapsed)
