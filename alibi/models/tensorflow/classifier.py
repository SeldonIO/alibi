import tensorflow as tf
import tensorflow.keras as keras


class Classifier(keras.Model):
    def __init__(self, classifier_net: keras.Model, name: str = "classifier"):
        super().__init__(name=name)
        self.classifier_net = classifier_net

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.classifier_net(x)
