from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation
import logging
import numpy as np
from typing import Any, Callable, Tuple
from skimage.segmentation import felzenszwalb, slic, quickshift
import copy

logger = logging.getLogger(__name__)


class AnchorImage(object):

    def __init__(self, predict_fn: Callable, image_shape: tuple, segmentation_fn: Any = 'slic',
                 segmentation_kwargs: dict = {'n_segments': 10, 'compactness': 10, 'sigma': .5},
                 images_background: np.ndarray = None) -> None:
        """
        Initialize anchor image explainer.

        Parameters
        ----------
        predict_fn
            Model prediction function.
        image_shape
            Shape of the image to be explained.
        segmentation_fn
            Any of the built in segmentation function strings: 'felzenszwalb', 'slic' or 'quickshift' or a custom
            segmentation function (callable) which returns an image mask with labels for each superpixel.
            See http://scikit-image.org/docs/dev/api/skimage.segmentation.html for more info.
        segmentation_kwargs
            Keyword arguments for the built in segmentation functions.
        images_background
            Images to overlay superpixels on.
        """
        if callable(segmentation_fn) and segmentation_kwargs is not None:
            logger.warning('Specified both a segmentation function to create superpixels and keyword '
                           'arguments for built segmentation functions. By default '
                           'the specified segmentation function will be used.')

        # check if predict_fn returns predicted class or prediction probabilities for each class
        # if needed adjust predict_fn so it returns the predicted class
        if np.argmax(predict_fn(np.zeros((1,) + image_shape)).shape) == 0:
            self.predict_fn = predict_fn
        else:
            self.predict_fn = lambda x: np.argmax(predict_fn(x), axis=1)

        # segmentation function is either a user-defined function or one of the values in
        fn_options = {'felzenszwalb': felzenszwalb, 'slic': slic, 'quickshift': quickshift}
        if callable(segmentation_fn):
            self.custom_segmentation = True
            self.segmentation_fn = segmentation_fn
        else:
            self.custom_segmentation = False
            self.segmentation_fn = lambda x: fn_options[segmentation_fn](x, **segmentation_kwargs)

        self.images_background = images_background

    def get_sample_fn(self, image: np.ndarray, p_sample: float = 0.5) -> Tuple[np.ndarray, Callable]:
        """
        Create sampling function and superpixel mask.

        Parameters
        ----------
        image
            Image to be explained
        p_sample
            Probability for a pixel to be represented by the average value of its superpixel or
            the pixel value of a superimposed image

        Returns
        -------
        segments
            Superpixels generated from image
        sample_fn
            Function returning the sampled images with label
        """
        # check if grayscale images need to be converted to RGB for superpixel generation
        if not self.custom_segmentation and image.shape[-1] == 1:
            image_segm = np.repeat(image, 3, axis=2)
        else:
            image_segm = image.copy()

        segments = self.segmentation_fn(image_segm)  # generate superpixels

        # each superpixel is a feature
        features = list(np.unique(segments))
        n_features = len(features)

        # true label is prediction on original image
        true_label = self.predict_fn(np.expand_dims(image, axis=0))[0]

        def sample_fn_image(present: list, num_samples: int,
                            compute_labels: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Create sampling function by masking certain superpixels from the original image and replacing them
            with the pixel values from superimposed images.

            Parameters
            ----------
            present
                List with features (= superpixels) present in the proposed anchor
            num_samples
                Number of samples used
            compute_labels
                Boolean whether to use labels coming from model predictions as 'true' labels

            Returns
            -------
            raw_data
                "data" output concatenated with the indices of the chosen background images for each sample
            data
                Nb of samples times nb of features matrix indicating whether a feature (= a superpixel) is
                present in the sample or masked
            labels
                Create labels using model predictions if compute_labels equals True
            """
            if not compute_labels:
                # for each sample, randomly sample whether a superpixel is represented by its average value or not
                data = np.random.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
                data[:, present] = 1  # superpixels in candidate anchor need to be present
                return np.array([]), data, np.array([])

            # for each sample, randomly sample whether a superpixel is represented by its
            # average value or not according to p_sample
            data = np.random.choice([0, 1], num_samples * n_features,
                                    p=[p_sample, 1 - p_sample]).reshape((num_samples, n_features))
            data[:, present] = 1  # superpixels in candidate anchor need to be present

            # for each sample, need to sample one of the background images
            chosen = np.random.choice(range(len(self.images_background)), data.shape[0], replace=True)

            # create masked images
            imgs = []
            for d, r in zip(data, chosen):
                temp = copy.deepcopy(image)
                zeros = np.where(d == 0)[0]  # unused superpixels for the sample
                # create mask for each superpixel not present in the sample
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                # for mask: replace values with those of background image
                temp[mask] = self.images_background[r][mask]
                imgs.append(temp)
            imgs = np.array(imgs)

            preds = self.predict_fn(imgs)  # make prediction on masked images

            # check if label for the masked images are the same as the true label
            labels = np.array((preds == true_label).astype(int))

            # concat data and indices of chosen background images for each sample
            raw_data = np.hstack((data, chosen.reshape(-1, 1)))  # nb of samples * (nb of superpixels + 1)
            return raw_data, data, labels

        if type(self.images_background) == np.ndarray:
            return segments, sample_fn_image

        # create fudged image where the pixel value in each superpixel is set to the average over the
        # superpixel for each channel
        fudged_image = image.copy()
        for x in np.unique(segments):
            fudged_image[segments == x] = [np.mean(image[segments == x][:, i]) for i in range(image.shape[-1])]

        def sample_fn_fudged(present: list, num_samples: int,
                             compute_labels: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Create sampling function by masking certain superpixels from the original image and replacing them
            with that superpixel's average value.

            Parameters
            ----------
            present
                List with features (= superpixels) present in the proposed anchor
            num_samples
                Number of samples used
            compute_labels
                Boolean whether to use labels coming from model predictions as 'true' labels

            Returns
            -------
            raw_data
                Same as data
            data
                Nb of samples times nb of features matrix indicating whether a feature (= a superpixel) is
                present in the sample or masked
            labels
                Create labels using model predictions if compute_labels equals True
            """
            if not compute_labels:
                # for each sample, randomly sample whether a superpixel is represented by its average value or not
                data = np.random.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
                data[:, present] = 1  # superpixels in candidate anchor need to be present
                return np.array([]), data, np.array([])

            # for each sample, randomly sample whether a superpixel is represented by its
            # average value or not according to p_sample
            data = np.random.choice([0, 1], num_samples * n_features,
                                    p=[p_sample, 1 - p_sample]).reshape((num_samples, n_features))
            data[:, present] = 1  # superpixels in candidate anchor need to be present

            # create perturbed (fudged) image for each sample using image masks
            imgs = []
            for row in data:
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]  # superpixels to be averaged for the sample
                # create mask for each pixel in the superpixels that are averaged
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
            imgs = np.array(imgs)

            preds = self.predict_fn(imgs)  # make prediction on masked images

            # check if labels for the masked images are the same as the true label
            labels = (preds == true_label).astype(int)

            raw_data = data
            return raw_data, data, labels

        return segments, sample_fn_fudged

    def explain(self, image: np.ndarray, threshold: float = 0.95, delta: float = 0.1,
                tau: float = 0.15, batch_size: int = 100, p_sample: float = 0.5, **kwargs: Any):
        """
        Explain instance and return anchor with metadata.

        Parameters
        ----------
        image
            Image to be explained
        threshold
            Minimum precision threshold
        delta
            Used to compute beta
        tau
            Margin between lower confidence bound and minimum precision of upper bound
        batch_size
            Batch size used for sampling
        p_sample
            Probability for a pixel to be represented by the average value of its superpixel

        Returns
        -------
        explanation
            Dictionary containing the anchor explaining the instance with additional metadata
        """
        # build sampling function and segments
        segments, sample_fn = self.get_sample_fn(image, p_sample=p_sample)

        # get anchors and add metadata
        exp = AnchorBaseBeam.anchor_beam(sample_fn, delta=delta,
                                         epsilon=tau, batch_size=batch_size,
                                         desired_confidence=threshold, **kwargs)  # type: Any
        exp['instance'] = image
        exp['prediction'] = self.predict_fn(np.expand_dims(image, axis=0))[0]

        # overlay image with anchor mask and do same for the examples
        anchor = AnchorImage.overlay_mask(image, segments, exp['feature'])
        cover_options = ['covered', 'covered_true', 'covered_false', 'uncovered_true', 'uncovered_false']
        for ex in exp['examples']:
            for opt in cover_options:
                tmp = [AnchorImage.overlay_mask(image, segments, list(np.where(ex[opt][i] == 1)[0]))
                       for i in range(ex[opt].shape[0])]
                ex[opt] = tmp

        exp = AnchorExplanation('image', exp)

        # output explanation dictionary
        explanation = {}
        explanation['anchor'] = anchor
        explanation['segments'] = segments
        explanation['precision'] = exp.precision()
        explanation['coverage'] = exp.coverage()
        explanation['raw'] = exp.exp_map
        return explanation

    @staticmethod
    def overlay_mask(image: np.ndarray, segments: np.ndarray, mask_features: list,
                     scale: tuple = (0, 255)) -> np.ndarray:
        """
        Overlay image with mask described by the mask features.

        Parameters
        ----------
        image
            Image to be explained
        segments
            Superpixels
        mask_features
            List with superpixels present in mask
        scale
            Pixel scale for masked image

        Returns
        -------
        Image overlaid with mask.
        """
        mask = np.zeros(segments.shape)
        for f in mask_features:
            mask[segments == f] = 1
        img_max, img_min = image.max(), image.min()
        image = ((image - img_min) / (img_max - img_min)) * (scale[1] - scale[0]) + scale[0]
        masked_image = (image * np.expand_dims(mask, 2)).astype(int)
        return masked_image
