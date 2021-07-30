import copy
import logging

import numpy as np

from functools import partial
from typing import Any, Callable

from alibi.utils.wrappers import ArgmaxTransformer
from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR_IMG
from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation
from .anchor_image_utils import scale_image
from .anchor_image_sampler import AnchorImageSampler
from skimage.segmentation import felzenszwalb, slic, quickshift

logger = logging.getLogger(__name__)

DEFAULT_SEGMENTATION_KWARGS = {
    "felzenszwalb": {},
    "quickshift": {},
    "slic": {"n_segments": 10, "compactness": 10, "sigma": 0.5},
}


class AnchorImage(Explainer):
    def __init__(
        self,
        predictor: Callable,
        image_shape: tuple,
        segmentation_fn: Any = "slic",
        segmentation_kwargs: dict = None,
        images_background: np.ndarray = None,
        seed: int = None,
    ) -> None:
        """
        Initialize anchor image explainer.

        Parameters
        ----------
        predictor
            A callable that takes a tensor of N data points as inputs and returns N outputs.
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
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ANCHOR))
        np.random.seed(seed)

        if isinstance(segmentation_fn, str) and not segmentation_kwargs:
            try:
                segmentation_kwargs = DEFAULT_SEGMENTATION_KWARGS[segmentation_fn]  # type: ignore
            except KeyError:
                logger.warning(
                    "DEFAULT_SEGMENTATION_KWARGS did not contain any entry"
                    "for segmentation method {}. No kwargs will be passed to"
                    "the segmentation function!".format(segmentation_fn)
                )
                segmentation_kwargs = {}
        elif callable(segmentation_fn) and segmentation_kwargs:
            logger.warning(
                "Specified both a segmentation function to create superpixels and "
                "keyword arguments for built segmentation functions. By default "
                "the specified segmentation function will be used."
            )

        # set the predictor
        self.image_shape = image_shape
        self.predictor = self._transform_predictor(predictor)

        # segmentation function is either a user-defined function or one of the values in
        fn_options = {
            "felzenszwalb": felzenszwalb,
            "slic": slic,
            "quickshift": quickshift,
        }
        if callable(segmentation_fn):
            self.custom_segmentation = True
            self.segmentation_fn = segmentation_fn
        else:
            self.custom_segmentation = False
            self.segmentation_fn = partial(
                fn_options[segmentation_fn], **segmentation_kwargs
            )

        self.images_background = images_background
        # a superpixel is perturbed with prob 1 - p_sample
        self.p_sample = 0.5  # type: float

        # update metadata
        self.meta["params"].update(
            custom_segmentation=self.custom_segmentation,
            segmentation_kwargs=segmentation_kwargs,
            p_sample=self.p_sample,
            seed=seed,
            image_shape=self.image_shape,
            images_background=self.images_background,
        )
        if not self.custom_segmentation:
            self.meta["params"].update(segmentation_fn=segmentation_fn)
        else:
            self.meta["params"].update(segmentation_fn="custom")

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

        image_preproc = self._preprocess_img(image)

        return self.segmentation_fn(image_preproc)

    def _preprocess_img(self, image: np.ndarray) -> np.ndarray:
        """
        Applies necessary transformations to the image prior to segmentation.

        Parameters
        ----------
        image
            A grayscale or RGB image.

        Returns
        -------
            A preprocessed image.
        """

        # Grayscale images are repeated across channels
        if not self.custom_segmentation and image.shape[-1] == 1:
            image_preproc = np.repeat(image, 3, axis=2)
        else:
            image_preproc = image.copy()

        return image_preproc

    def explain(
        self,  # type: ignore
        image: np.ndarray,
        p_sample: float = 0.5,
        threshold: float = 0.95,
        delta: float = 0.1,
        tau: float = 0.15,
        batch_size: int = 100,
        coverage_samples: int = 10000,
        beam_size: int = 1,
        stop_on_first: bool = False,
        max_anchor_size: int = None,
        min_samples_start: int = 100,
        n_covered_ex: int = 10,
        binary_cache_size: int = 10000,
        cache_margin: int = 1000,
        verbose: bool = False,
        verbose_every: int = 1,
        **kwargs: Any
    ) -> Explanation:

        """
        Explain instance and return anchor with metadata.

        Parameters
        ----------
        image
            Image to be explained.
        p_sample
            Probability for a pixel to be represented by the average value of its superpixel.
        threshold
            Minimum precision threshold.
        delta
            Used to compute beta.
        tau
            Margin between lower confidence bound and minimum precision of upper bound.
        batch_size
            Batch size used for sampling.
        coverage_samples
            Number of samples used to estimate coverage from during result search.
        beam_size
            The number of anchors extended at each step of new anchors construction.
        stop_on_first
            If True, the beam search algorithm will return the first anchor that has satisfies the
            probability constraint.
        max_anchor_size
            Maximum number of features in result.
        min_samples_start
            Min number of initial samples.
        n_covered_ex
            How many examples where anchors apply to store for each anchor sampled during search
            (both examples where prediction on samples agrees/disagrees with desired_label are stored).
        binary_cache_size
            The result search pre-allocates binary_cache_size batches for storing the binary arrays
            returned during sampling.
        cache_margin
            When only max(cache_margin, batch_size) positions in the binary cache remain empty, a new cache
            of the same size is pre-allocated to continue buffering samples.
        verbose
            Display updates during the anchor search iterations.
        verbose_every
            Frequency of displayed iterations during anchor search process.

        Returns
        -------
        explanation
            `Explanation` object containing the anchor explaining the instance with additional metadata as attributes.
        """
        # get params for storage in meta
        params = locals()
        remove = ["image", "self"]
        for key in remove:
            params.pop(key)

        sampler = AnchorImageSampler(
            predictor=self.predictor,
            segmentation_fn=self.segmentation_fn,
            custom_segmentation=self.custom_segmentation,
            image=image,
            images_background=self.images_background,
            p_sample=p_sample,
            n_covered_ex=n_covered_ex,
        )

        # get anchors and add metadata
        mab = AnchorBaseBeam(
            samplers=[sampler.sample],
            sample_cache_size=binary_cache_size,
            cache_margin=cache_margin,
            **kwargs,
        )
        result = mab.anchor_beam(
            desired_confidence=threshold,
            delta=delta,
            epsilon=tau,
            batch_size=batch_size,
            coverage_samples=coverage_samples,
            beam_size=beam_size,
            stop_on_first=stop_on_first,
            max_anchor_size=max_anchor_size,
            min_samples_start=min_samples_start,
            verbose=verbose,
            verbose_every=verbose_every,
            **kwargs,
        )  # type: Any

        return self.build_explanation(
            image, result, sampler.instance_label, params, sampler
        )

    def build_explanation(
        self,
        image: np.ndarray,
        result: dict,
        predicted_label: int,
        params: dict,
        sampler: AnchorImageSampler,
    ) -> Explanation:
        """
        Uses the metadata returned by the anchor search algorithm together with
        the instance to be explained to build an explanation object.

        Parameters
        ----------
        image
            Instance to be explained.
        result
            Dictionary containing the search anchor and metadata.
        predicted_label
            Label of the instance to be explained.
        params
            Parameters passed to `explain`
        """

        result["instance"] = image
        result["instances"] = np.expand_dims(image, 0)
        result["prediction"] = np.array([predicted_label])

        # overlay image with anchor mask
        anchor = self.overlay_mask(image, sampler.segments, result["feature"])
        exp = AnchorExplanation("image", result)

        # output explanation dictionary
        data = copy.deepcopy(DEFAULT_DATA_ANCHOR_IMG)
        data.update(
            anchor=anchor,
            segments=sampler.segments,
            precision=exp.precision(),
            coverage=exp.coverage(),
            raw=exp.exp_map,
        )

        # create explanation object
        explanation = Explanation(meta=copy.deepcopy(self.meta), data=data)

        # params passed to explain
        explanation.meta["params"].update(params)
        return explanation

    def overlay_mask(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        mask_features: list,
        scale: tuple = (0, 255),
    ) -> np.ndarray:
        """
        Overlay image with mask described by the mask features.

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
        image = scale_image(image, scale=scale)
        masked_image = (image * np.expand_dims(mask, 2)).astype(int)

        return masked_image

    def _transform_predictor(self, predictor: Callable) -> Callable:
        # check if predictor returns predicted class or prediction probabilities for each class
        # if needed adjust predictor so it returns the predicted class
        if np.argmax(predictor(np.zeros((1,) + self.image_shape)).shape) == 0:
            return predictor
        else:
            transformer = ArgmaxTransformer(predictor)
            return transformer

    def reset_predictor(self, predictor: Callable) -> None:
        self.predictor = self._transform_predictor(predictor)
