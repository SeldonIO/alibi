# flake8: noqa E731

import keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
from alibi.explainers import AnchorImage


def test_anchor_image():
    # load and prepare fashion MNIST data
    (x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_train = np.reshape(x_train, x_train.shape + (1,))
    y_train = to_categorical(y_train)

    # define and train model
    def model():
        x_in = Input(shape=(28, 28, 1))
        x = Conv2D(filters=8, kernel_size=2, padding="same", activation="relu")(x_in)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        x_out = Dense(10, activation="softmax")(x)
        cnn = Model(inputs=x_in, outputs=x_out)
        cnn.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return cnn

    cnn = model()
    cnn.fit(x_train, y_train, batch_size=256, epochs=1)

    # test explainer initialization
    predict_fn = lambda x: cnn.predict(x)
    segmentation_fn = "slic"
    segmentation_kwargs = {"n_segments": 10, "compactness": 10, "sigma": 0.5}
    image_shape = (28, 28, 1)
    explainer = AnchorImage(
        predict_fn,
        image_shape,
        segmentation_fn=segmentation_fn,
        segmentation_kwargs=segmentation_kwargs,
    )
    assert explainer.predict_fn(np.zeros((1,) + image_shape)).shape == (1,)

    # test sampling and segmentation functions
    image = x_train[0]
    segments, sample_fn = explainer.get_sample_fn(image, p_sample=0.5)
    raw_data, data, labels = sample_fn([], 10)
    assert raw_data.shape == data.shape
    assert data.shape[0] == labels.shape[0]
    assert data.shape[1] == len(np.unique(segments))

    # test explanation
    threshold = 0.95
    explanation = explainer.explain(image, threshold=threshold)
    assert explanation["anchor"].shape == image_shape
    assert explanation["precision"] >= threshold
    assert len(np.unique(explanation["segments"])) == len(np.unique(segments))
