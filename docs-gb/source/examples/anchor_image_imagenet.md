# Anchor explanations for ImageNet

```python
import tensorflow as tf
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from alibi.datasets import load_cats
from alibi.explainers import AnchorImage
```

### Load InceptionV3 model pre-trained on ImageNet

```python
model = InceptionV3(weights='imagenet')
```

### Load and pre-process sample images

The _load\_cats_ function loads a small sample of images of various cat breeds.

```python
image_shape = (299, 299, 3)
data, labels = load_cats(target_size=image_shape[:2], return_X_y=True)
print(f'Images shape: {data.shape}')
```

```
Images shape: (4, 299, 299, 3)
```

Apply image preprocessing, make predictions and map predictions back to categories. The output label is a tuple which consists of the class name, description and the prediction probability.

```python
images = preprocess_input(data)
preds = model.predict(images)
label = decode_predictions(preds, top=3)
print(label[0])
```

```
1/1 [==============================] - 4s 4s/step
[('n02123045', 'tabby', 0.82086897), ('n02123159', 'tiger_cat', 0.14372891), ('n02124075', 'Egyptian_cat', 0.01642174)]
```

### Define prediction function

```python
predict_fn = lambda x: model.predict(x)
```

### Initialize anchor image explainer

The segmentation function will be used to generate superpixels. It is important to have meaningful superpixels in order to generate a useful explanation. Please check scikit-image's [segmentation methods](http://scikit-image.org/docs/dev/api/skimage.segmentation.html) (_felzenszwalb_, _slic_ and _quickshift_ built in the explainer) for more information.

In the example, the pixels not in the proposed anchor will take the average value of their superpixel. Another option is to superimpose the pixel values from other images which can be passed as a numpy array to the _images\_background_ argument.

```python
segmentation_fn = 'slic'
kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5, 'start_label': 0}
explainer = AnchorImage(predict_fn, image_shape, segmentation_fn=segmentation_fn, 
                        segmentation_kwargs=kwargs, images_background=None)
```

```
1/1 [==============================] - 2s 2s/step
```

### Explain a prediction

The explanation of the below image returns a mask with the superpixels that constitute the anchor.

```python
i = 0
plt.imshow(data[i]);
```

![png](../../.gitbook/assets/anchor_image_imagenet_14_0.png)

The _threshold_, _p\_sample_ and _tau_ parameters are also key to generate a sensible explanation and ensure fast enough convergence. The _threshold_ defines the minimum fraction of samples for a candidate anchor that need to lead to the same prediction as the original instance. While a higher threshold gives more confidence in the anchor, it also leads to longer computation time. _p\_sample_ determines the fraction of superpixels that are changed to either the average value of the superpixel or the pixel value for the superimposed image. The pixels in the proposed anchors are of course unchanged. The parameter _tau_ determines when we assume convergence. A bigger value for _tau_ means faster convergence but also looser anchor restrictions.

```python
image = images[i]
np.random.seed(0)
explanation = explainer.explain(image, threshold=.95, p_sample=.5, tau=0.25)
```

```
1/1 [==============================] - 0s 25ms/step
4/4 [==============================] - 5s 304ms/step
4/4 [==============================] - 1s 437ms/step
4/4 [==============================] - 1s 448ms/step
4/4 [==============================] - 1s 436ms/step
```

Superpixels in the anchor:

```python
plt.imshow(explanation.anchor);
```

![png](../../.gitbook/assets/anchor_image_imagenet_18_0.png)

A visualization of all the superpixels:

```python
plt.imshow(explanation.segments);
```

![png](../../.gitbook/assets/anchor_image_imagenet_20_0.png)
