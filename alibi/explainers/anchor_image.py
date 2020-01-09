import copy
import logging

import numpy as np

from functools import partial
from typing import Any, Callable, Tuple

from alibi.utils.data import ArgmaxTransformer
from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation
from skimage.segmentation import felzenszwalb, slic, quickshift

logger = logging.getLogger(__name__)

DEFAULT_SEGMENTATION_KWARGS = {
    'felzenszwalb': {},
    'quickshift': {},
    'slic': {'n_segments': 10, 'compactness': 10, 'sigma': .5}
}


class AnchorImage(object):

    def __init__(self, predictor: Callable, image_shape: tuple, segmentation_fn: Any = 'slic',
                 segmentation_kwargs: dict = None, images_background: np.ndarray = None, seed: int = None) -> None:
        """
        Initialize result image explainer.

        Parameters
        ----------
        predictor
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
        seed
            If set, ensures different runs with the same input will yield same explanation.
        """

        np.random.seed(seed)

        if isinstance(segmentation_fn, str) and not segmentation_kwargs:
            try:
                segmentation_kwargs = DEFAULT_SEGMENTATION_KWARGS[segmentation_fn]
            except KeyError:
                logger.warning('DEFAULT_SEGMENTATION_KWARGS did not contain any entry'
                               'for segmentation method {}. No kwargs will be passed to'
                               'the segmentation function!'.format(segmentation_fn))
                segmentation_kwargs = {}

        elif callable(segmentation_fn) and segmentation_kwargs:
            logger.warning('Specified both a segmentation function to create superpixels and keyword '
                           'arguments for built segmentation functions. By default '
                           'the specified segmentation function will be used.')

        # check if predictor returns predicted class or prediction probabilities for each class
        # if needed adjust predictor so it returns the predicted class
        if np.argmax(predictor(np.zeros((1,) + image_shape)).shape) == 0:
            self.predictor = predictor
        else:
            self.predictor = ArgmaxTransformer(predictor)

        # segmentation function is either a user-defined function or one of the values in
        fn_options = {'felzenszwalb': felzenszwalb, 'slic': slic, 'quickshift': quickshift}
        if callable(segmentation_fn):
            self.custom_segmentation = True
            self.segmentation_fn = segmentation_fn
        else:
            self.custom_segmentation = False
            self.segmentation_fn = partial(fn_options[segmentation_fn], **segmentation_kwargs)

        self.images_background = images_background
        self.image_shape = image_shape
        self.segments: np.ndarray = None  # [H, W] int array; each int is a superpixel labels
        self.segment_labels: list = None  # superpixel labels
        self.image: np.ndarray = None     # instance to be explained
        self.p_sample: float = 0.5        # a superpixel is perturbed with prob 1 - p_sample

    def generate_superpixels(self, image: np.ndarray) -> np.ndarray:
        """
        Generates superpixels from (i.e., segments) an image.

        Parameters
        ----------
        image
            A grayscale or RGB image.

        Returns
        -------
            A [H, W] array of integers. Each integer is a segment (superpixel) label.
        """

        # Grayscale images are repeated across channels
        if not self.custom_segmentation and image.shape[-1] == 1:
            image_segm = np.repeat(image, 3, axis=2)
        else:
            image_segm = image.copy()

        return self.segmentation_fn(image_segm)

    def _choose_superpixels(self, num_samples: int, p_sample: float = 0.5) -> np.ndarray:
        """
        Generates a binary mask of dimension [num_samples, M] where M is the number of
        image superpixels (segments).

        Parameters
        ----------
        num_samples
            Number of perturbed images to be generated
        p_sample:
            The probability that a superpixel is perturbed

        Returns
        -------
        data
            Binary 2D mask, where each non-zero entry in a row indicates that
            the values of the particular image segment will not be perturbed.
        """

        n_features = len(self.segment_labels)
        data = np.random.choice([0, 1], num_samples * n_features, p=[p_sample, 1-p_sample])
        data = data.reshape((num_samples, n_features))

        return data

    def sampler(self, anchor: tuple, num_samples: int, c_labels: bool = True) -> list:
        """
        Sample images from a perturbation distribution by masking randomly chosen superpixels
        from the original image and replacing them with pixel values from superimposed images
        if background images are provided to the explainer. Otherwise, the superpixels from the
        original image are replaced with their average values.

        Parameters
        ----------
        anchor
            List with segment_labels (= superpixels) present in the proposed result
        num_samples
            Number of samples used
        c_labels
            Whether to use labels coming from model predictions as 'true' labels

        Returns
        -------
            If c_labels=True, a list containing the following is returned:
                - covered_true (np.ndarray): an array containing perturbed examples where the result
                    applies and the model prediction on perturbed is the same as the instance prediction
                - covered_false (np.ndarray): an array containing examples where the result applies and
                    the model prediction on pertrurbed sample is NOT the same as the instance prediction
                - labels (np.ndarray): an np.array with num_samples ints indicating whether
                    the prediction on the perturbed sample matches (1) the label of the instance to be
                    explained or not (0)
                - data (np.ndarray): Matrix with 1s and 0s indicating whether the values in a superpixel will
                    remain unchanged (1) or will be perturbed (0), for each sample
                - 1.0: indicates exact coverage is not computed for this algorithm
                - result[0]: position of result in the batch request
            Otherwise, a list containing the data matrix only is returned.
        """

        if c_labels:
            raw_data, data = self.perturbation(anchor[1], num_samples)
            labels = self.compare_labels(raw_data)
            covered_true = raw_data[labels][:self.n_covered_ex]
            covered_true = [self._scale(img) for img in covered_true]
            covered_false = raw_data[np.logical_not(labels)][:self.n_covered_ex]
            covered_false = [self._scale(img) for img in covered_false]
            # coverage set to -1.0 as we can't compute 'true'coverage for this model

            return [covered_true, covered_false, labels.astype(int), data, -1.0, anchor[0]]

        else:
            data = self._choose_superpixels(num_samples)
            data[:, anchor[1]] = 1  # superpixels in candidate result are not perturbed

            return [data]

    def perturbation(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perturbs an image by altering the values of selected superpixels. If a dataset of image
        backgrounds is provided to the explainer, then the superpixels are replaced with the
        equivalent superpixels from the background image. Otherwise, the superpixels are replaced
        by their average value.

        Parameters
        ----------
        anchor:
            Contains the superpixels whose values are not going to be perturbed.
        num_samples:
            Number of perturbed samples to be returned.

        Returns
        -------
        imgs
            A [num_samples, H, W, C] array of perturbed images.
        segments_mask
            A [num_samples, M] binary mask, where M is the number of image superpixels
            segments. 1 indicates the values in that particular superpixels are not
            perturbed.
        """

        image = self.image
        segments = self.segments

        # choose superpixels to be perturbed
        segments_mask = self._choose_superpixels(num_samples, p_sample=self.p_sample)
        segments_mask[:, anchor] = 1

        # for each sample, need to sample one of the background images if provided
        if self.images_background:
            backgrounds = np.random.choice(
                range(len(self.images_background)), 
                segments_mask.shape[0], 
                replace=True,
            )
            segments_mask = np.hstack((segments_mask, backgrounds.reshape(-1, 1)))  
        else:
            backgrounds = [None] * segments_mask.shape[0]
            # create fudged image where the pixel value in each superpixel is set to the average over the
            # superpixel for each channel
            fudged_image = image.copy()
            n_channels = image.shape[-1]
            for x in np.unique(segments):
                fudged_image[segments == x] = [np.mean(image[segments == x][:, i]) for i in range(n_channels)]

        pert_imgs = []
        for mask, background_idx in zip(segments_mask, backgrounds):
            temp = copy.deepcopy(image)
            to_perturb = np.where(mask == 0)[0]
            # create mask for each superpixel not present in the sample
            mask = np.zeros(segments.shape).astype(bool)
            for superpixel in to_perturb:
                mask[segments == superpixel] = True
            if background_idx:
                # replace values with those of background image
                temp[mask] = self.images_background[background_idx][mask]
            else:
                # ... or with the averaged superpixel value
                temp[mask] = fudged_image[mask]
            pert_imgs.append(temp)

        return np.array(pert_imgs), segments_mask

    def compare_labels(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute the agreement between a classifier prediction on an instance to be explained
        and the prediction on a set of samples which have a subset of perturbed superpixels.

        Parameters
        ----------
        samples
            Samples whose labels are to be compared with the instance label.

        Returns
        -------
            A boolean array indicating whether the prediction was the same as the instance label.
        """

        return self.predictor(samples) == self.instance_label

    def explain(self, image: np.ndarray, threshold: float = 0.95, delta: float = 0.1,
                tau: float = 0.15, batch_size: int = 100, p_sample: float = 0.5,
                n_covered_ex: int = 10, **kwargs: Any) -> dict:

        """
        Explain instance and return result with metadata.

        Parameters
        ----------
        image
            Image to be explained.
        threshold
            Minimum precision threshold.
        delta
            Used to compute beta.
        tau
            Margin between lower confidence bound and minimum precision of upper bound.
        batch_size
            Batch size used for sampling.
        p_sample
            Probability for a pixel to be represented by the average value of its superpixel.
        n_covered_ex
            How many examples where anchors apply to store for each result sampled during search
            (both examples where prediction on samples agrees/disagrees with desired_label are stored).

        Returns
        -------
        explanation
            Dictionary containing the result explaining the instance with additional metadata.
        """

        # TODO: Decide if additional arguments should be specified here (e.g., beam, max_anchor_size, top_n etc)

        self.image = image
        self.n_covered_ex = n_covered_ex
        self.p_sample = p_sample
        self.segments = self.generate_superpixels(image)
        self.segment_labels = list(np.unique(self.segments))
        self.instance_label = self.predictor(image[np.newaxis, ...])[0]

        # get anchors and add metadata
        self.mab = AnchorBaseBeam(samplers=[self.sampler], **kwargs)
        result: Any = self.mab.anchor_beam(
            delta=delta,
            epsilon=tau,
            batch_size=batch_size,
            desired_confidence=threshold,
            **kwargs,
        )

        return self.build_explanation(image, result, self.instance_label)

    def build_explanation(self, image: np.ndarray, exp: dict, true_label: int) -> dict:
        """
        Uses the metadata returned by the result search algorithm together with
        the instance to be explained to build an explanation object.

        Parameters
        ----------
        image
            Instance to be explained.
        exp
            Dictionary containing the search result and metadata.
        true_label
            Label of the instance to be explained.
        """

        exp['instance'] = image
        exp['prediction'] = true_label

        # overlay image with result mask
        anchor = self.overlay_mask(image, self.segments, exp['feature'])
        exp = AnchorExplanation('image', exp)

        return {
            'result': anchor,
            'segments': self.segments,
            'precision': exp.precision(),
            'coverage': exp.coverage(),
            'raw': exp.exp_map,
            'meta': {'name': self.__class__.__name__}
        }

    @staticmethod
    def _scale(image: np.ndarray, scale: tuple = (0, 255)) -> np.ndarray:
        """
        Scales an image in a specified range.

        Parameters
        ----------
        image
            Image to be scale.
        scale
            The scaling interval.

        Returns
        -------
        img_scaled
            Scaled image.
        """

        img_max, img_min = image.max(), image.min()
        img_std = (image - img_min) / (img_max - img_min)
        img_scaled = img_std * (scale[1] - scale[0]) + scale[0]

        return img_scaled

    def overlay_mask(self, image: np.ndarray, segments: np.ndarray, mask_features: list,
                     scale: tuple = (0, 255)) -> np.ndarray:
        """
        Overlay image with mask described by the mask segment_labels.

        Parameters
        ----------
        image
            Image to be explained.
        segments
            Superpixels
        mask_features
            List with superpixels present in mask.
        scale
            Pixel scale for masked image.

        Returns
        -------
        masked_image
            Image overlaid with mask.
        """

        mask = np.zeros(segments.shape)
        for f in mask_features:
            mask[segments == f] = 1
        image = self._scale(image, scale=scale)
        masked_image = (image * np.expand_dims(mask, 2)).astype(int)

        return masked_image
