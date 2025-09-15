# alibi/tests/test_counterfactual_proto.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow.python.keras.backend as K

from alibi.api.interfaces import Explanation
from alibi.explainers.cfproto import CounterfactualProto


def test_cfproto_uses_k_session_blackbox():
    tf.reset_default_graph()
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)

    with sess.as_default():
        # Simple TF1 graph: softmax(Wx+b)
        x_ph = tf.placeholder(tf.float32, shape=(None, 4), name="x")
        W = tf.get_variable("W", shape=(4, 2),
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", shape=(2,),
                            initializer=tf.zeros_initializer())
        logits = tf.matmul(x_ph, W) + b
        probs = tf.nn.softmax(logits)

        sess.run(tf.global_variables_initializer())

        def predict_fn(x: np.ndarray) -> np.ndarray:
            return sess.run(probs, feed_dict={x_ph: x})

        explainer = CounterfactualProto(
            predict=predict_fn,
            shape=(1, 4),
            max_iterations=5,
            c_steps=1,
            c_init=0.0,
            kappa=0.0,
            beta=0.1,
            gamma=0.0,
            use_kdtree=False,
        )

        assert explainer.sess is K.get_session()

        x0 = np.zeros((1, 4), dtype="float32")
        explanation = explainer.explain(x0)

        assert isinstance(explanation, Explanation)
        assert "orig_class" in explanation.data
        if explanation.data.get("cf") and "X" in explanation.data["cf"]:
            assert explanation.data["cf"]["X"].shape == x0.shape
