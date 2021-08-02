import numpy as np
import copy

from typing import Tuple, Callable, List, Union

from .anchor_image_utils import scale_image


class AnchorImageSampler:
    def __init__(
        self,
        # TODO: Should we call `predictor`, `prediction_fn` instead?
        predictor: Callable,
        segmentation_fn: Callable,
        custom_segmentation: bool,
        image: np.ndarray,
        images_background: np.ndarray = None,
        p_sample: float = 0.5,
        n_covered_ex: int = 10,
    ):
        """
        Initialize anchor image sampler.

        Parameters
        ----------
        predictor
            A callable that takes a tensor of N data points as inputs and returns N outputs.
        segmentation_fn
            Function used to segment the images.
        image
            Image to be explained.
        images_background
            Images to overlay superpixels on.
        p_sample
            Probability for a pixel to be represented by the average value of its superpixel.
        n_covered_ex
            How many examples where anchors apply to store for each anchor sampled during search
            (both examples where prediction on samples agrees/disagrees with desired_label are stored).
        """
        self.predictor = predictor
        self.segmentation_fn = segmentation_fn
        self.custom_segmentation = custom_segmentation
        self.image = image
        self.images_background = images_background
        self.n_covered_ex = n_covered_ex
        self.p_sample = p_sample
        self.segments = self.generate_superpixels(image)
        self.segment_labels = list(np.unique(self.segments))
        self.instance_label = self.predictor(image[np.newaxis, ...])[0]

    def __call__(
        self, anchor: Tuple[int, tuple], num_samples: int, compute_labels: bool = True
    ) -> List[Union[np.ndarray, float, int]]:
        """
        Sample images from a perturbation distribution by masking randomly chosen superpixels
        from the original image and replacing them with pixel values from superimposed images
        if background images are provided to the explainer. Otherwise, the superpixels from the
        original image are replaced with their average values.

        Parameters
        ----------
        anchor
            int: order of anchor in the batch
            tuple: features (= superpixels) present in the proposed anchor
        num_samples
            Number of samples used
        compute_labels
            If True, an array of comparisons between predictions on perturbed samples and
            instance to be explained is returned.

        Returns
        -------
            If compute_labels=True, a list containing the following is returned:
                - covered_true: perturbed examples where the anchor applies and the model prediction
                    on perturbed is the same as the instance prediction
                - covered_false: perturbed examples where the anchor applies and the model prediction
                    on pertrurbed sample is NOT the same as the instance prediction
                - labels: num_samples ints indicating whether the prediction on the perturbed sample
                    matches (1) the label of the instance to be explained or not (0)
                - data: Matrix with 1s and 0s indicating whether the values in a superpixel will
                    remain unchanged (1) or will be perturbed (0), for each sample
                - 1.0: indicates exact coverage is not computed for this algorithm
                - anchor[0]: position of anchor in the batch request
            Otherwise, a list containing the data matrix only is returned.
        """

        if compute_labels:
            raw_data, data = self.perturbation(anchor[1], num_samples)
            labels = self.compare_labels(raw_data)
            covered_true = raw_data[labels][: self.n_covered_ex]
            covered_true = [scale_image(img) for img in covered_true]
            covered_false = raw_data[np.logical_not(labels)][: self.n_covered_ex]
            covered_false = [scale_image(img) for img in covered_false]
            # coverage set to -1.0 as we can't compute 'true'coverage for this model

            return [covered_true, covered_false, labels.astype(int), data, -1.0, anchor[0]]  # type: ignore

        else:
            data = self._choose_superpixels(num_samples)
            data[:, anchor[1]] = 1  # superpixels in candidate anchor are not perturbed

            return [data]

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

    def _choose_superpixels(
        self, num_samples: int, p_sample: float = 0.5
    ) -> np.ndarray:
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
        data = np.random.choice(
            [0, 1], num_samples * n_features, p=[p_sample, 1 - p_sample]
        )
        data = data.reshape((num_samples, n_features))

        return data

    def perturbation(
        self, anchor: tuple, num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            # create fudged image where the pixel value in each superpixel is set to the
            # average over the superpixel for each channel
            fudged_image = image.copy()
            n_channels = image.shape[-1]
            for x in np.unique(segments):
                fudged_image[segments == x] = [
                    np.mean(image[segments == x][:, i]) for i in range(n_channels)
                ]

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
                # TODO: Could images_background be None herre?
                temp[mask] = self.images_background[background_idx][mask]
            else:
                # ... or with the averaged superpixel value
                # TODO: Where is fudged_image defined?
                temp[mask] = fudged_image[mask]
            pert_imgs.append(temp)

        return np.array(pert_imgs), segments_mask

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
