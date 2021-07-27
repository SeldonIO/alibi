import tensorflow as tf
from typing import List, Dict

from alibi.explainers.backends.tensorflow.cfrl_base import TfCounterfactualRLBaseBackend
from alibi.explainers.backends.cfrl_tabular import CounterfactualRLTabularBackend


class TfCounterfactualRLTabularBackend(CounterfactualRLTabularBackend, TfCounterfactualRLBaseBackend):
    @staticmethod
    def tensorflow_sample_differentiable(x_ohe_hat_split: List[tf.Tensor],
                                         category_map: Dict[int, List[str]]) -> List[tf.Tensor]:
        """
        Samples differentiable reconstruction.

        Parameters
        ----------
        x_ohe_hat_split
            List of one-hot encoded reconstructed columns form the auto-encoder.
        category_map
            Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
            values for an attribute.

        Returns
        -------
        Differentiable reconstruction.
        """
        num_attr = len(x_ohe_hat_split) - len(category_map)
        cat_attr = len(category_map)
        x_out = []

        # pass numerical attributes as they are
        if num_attr > 0:
            x_out.append(x_ohe_hat_split[0])

        # sample categorical attributes
        if cat_attr > 0:
            for head in x_ohe_hat_split[-cat_attr:]:
                out = tf.argmax(head, axis=1)

                # transform to one-hot encoding
                out = tf.one_hot(out, depth=head.shape[1])
                proba = tf.nn.softmax(head, axis=1)
                out = out - tf.stop_gradient(proba) + proba
                x_out.append(out)

        return x_out

    @staticmethod
    def l0_ohe(input: tf.Tensor,
               target: tf.Tensor,
               reduction: str = 'none') -> tf.Tensor:
        """
        Computes the L0 loss for a one-hot encoding representation.

        Parameters
        ----------
        input
            Input tensor.
        target
            Target tensor
        reduction
            Specifies the reduction to apply to the output: `none` | `mean` | `sum`.

        Returns
        -------
        L0 loss.
        """
        loss = tf.maximum(target - input, tf.zeros_like(input))

        if reduction == 'none':
            return loss

        if reduction == 'mean':
            tf.reduce_mean(loss)

        if reduction == 'sum':
            return tf.reduce_sum(loss)

        raise ValueError(f"Reduction {reduction} not implemented.")

    @staticmethod
    def l1_loss(input: tf.Tensor, target=tf.Tensor, reduction: str = 'none') -> tf.Tensor:
        """
        Computes the L1 loss.

        Parameters
        ----------
        input
           Input tensor.
        target
           Target tensor
        reduction
           Specifies the reduction to apply to the output: `none` | `mean` | `sum`.

        Returns
        -------
        L1 loss.
        """
        loss = tf.abs(input - target)

        if reduction == 'none':
            return loss

        if reduction == 'mean':
            return tf.reduce_mean(loss)

        if reduction == 'sum':
            return tf.reduce_sum(loss)

        raise ValueError(f"Reduction {reduction} not implemented.")

    @staticmethod
    def tensorflow_he_sparsity_loss(x_ohe_hat_split: List[tf.Tensor],
                                    x_ohe: tf.Tensor,
                                    category_map: Dict[int, List[str]],
                                    weight_num: float = 1.0,
                                    weight_cat: float = 1.0):
        """
        Computes heterogeneous sparsity loss.

        Parameters
        ----------
        x_ohe_hat_split
            List of one-hot encoded reconstructed columns form the auto-encoder.
        x_ohe
            One-hot encoded representation of the input.
        category_map
            Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
            values for an attribute.
        weight_num
            Numerical loss weight.
        weight_cat
            Categorical loss weight.

        Returns
        -------
        Heterogeneous sparsity loss.
        """
        # split the input into a list of tensor, where each element corresponds to a network head
        x_ohe_num_split, x_ohe_cat_split = TfCounterfactualRLTabularBackend.split_ohe(x_ohe=x_ohe,
                                                                                      category_map=category_map)

        # sample differentiable output
        x_ohe_hat_split = TfCounterfactualRLTabularBackend.sample_differentiable(x_ohe_hat_split=x_ohe_hat_split,
                                                                                 category_map=category_map)

        # define numerical and categorical loss
        num_loss, cat_loss = 0, 0
        offset = 0

        # compute numerical loss
        if len(x_ohe_num_split) > 0:
            offset = 1
            num_loss = tf.reduce_mean(TfCounterfactualRLTabularBackend.l1_loss(input=x_ohe_hat_split[0],
                                                                               target=x_ohe_num_split[0],
                                                                               reduction='none'))

        # compute categorical loss
        if len(x_ohe_cat_split) > 0:
            for i in range(len(x_ohe_cat_split)):
                batch_size = x_ohe_hat_split[i].shape[0]
                cat_loss += tf.reduce_sum(TfCounterfactualRLTabularBackend.l0_ohe(input=x_ohe_hat_split[i + offset],
                                                                                  target=x_ohe_cat_split[i],
                                                                                  reduction='none')) / batch_size

            cat_loss /= len(x_ohe_cat_split)

        return {"sparsity_num_loss": weight_num * num_loss, "sparsity_cat_loss": weight_cat * cat_loss}

    @staticmethod
    def tensorflow_he_consistency_loss(z_cf_pred: tf.Tensor, z_cf_tgt: Union[np.ndarray, tf.Tensor], **kwargs):
        """
        Computes heterogeneous consistency loss.

        Parameters
        ----------
        z_cf_pred
                Counterfactual embedding prediction.
        z_cf_tgt
            Counterfactual embedding target.


        Returns
        -------
        Heterogeneous consistency loss.
        """
        # compute consistency loss
        loss = tf.reduce_mean(tf.square(z_cf_pred - z_cf_tgt))
        return {"consistency_loss": loss}