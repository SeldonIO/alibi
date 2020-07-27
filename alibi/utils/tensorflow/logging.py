import functools
import os
from functools import partial
from typing import Dict, Union, Any

import numpy as np
import tensorflow as tf

from alibi.utils.logging import tensorboard_logger, TensorboardWriterBase


@tensorboard_logger
class TFTensorboardWriter(TensorboardWriterBase):

    framework = 'tensorflow'

    def __init__(self):

        super().__init__()

    def _init_summaries(self):
        """
        See superclass documentation.
        """

        # see Tensorboard docs for meaning of kwargs
        self._summary_functions_kwargs = {
            'audio':  {'sample_rate': 44100, 'encoding': None, 'max_outputs': 3},
            'histogram': {'buckets': None},
            'image': {'max_outputs': 3},

        }

        self._summary_functions = {
            'audio': partial(tf.summary.audio, **self._summary_functions_kwargs['audio']),
            'histogram': partial(tf.summary.histogram, **self._summary_functions_kwargs['histogram']),
            'image': partial(tf.summary.image, **self._summary_functions_kwargs['image']),
            'scalar': tf.summary.scalar,
            'text': tf.summary.text,
        }

    def _create_writer(self):
        """
        See superclass documentation.
        """

        trace_dir = os.path.join(self.logging_opts['trace_dir'], f"run_{self.run}")
        self.writer = tf.summary.create_file_writer(trace_dir)
        self.run += 1

    def record(self, step, data_store: Dict[str, Union[float, np.ndarray]], prefix: str = '') -> None:
        """
        TensoFlow record implementation. See superclass for further infromation.

        Parameters
        ----------
        See superclass description.
        """

        with tf.summary.record_if(tf.equal(step % self.summary_freq, 0)):
            with self.writer.as_default():
                for variable in self.tracked_variables:
                    if variable not in data_store:
                        continue
                    tag, var_type, description = self._vars_mapping[variable]
                    if var_type == 'image' and step % self.image_summary_freq > 0:
                        continue
                    prefixed_tag = os.path.join(prefix, tag)
                    self._summary_functions[var_type](name=prefixed_tag, data=data_store[variable], step=step)
        self.writer.flush()

    def record_step(self, step: int, tag: str, value: Union[float, np.ndarray], data_type: str, **kwargs) -> None:
        """
        Used to record a single step from a quantity. Potential use cases:

            - Log the final solution of a search procedure (e.g., a counterfactual for an image)
            - Log quantities that are not defined in the 'tracked_variables` section of the default configuration
            - Log quantities for which the summary operations require kwargs different than the ones set for the
            operation in the recorder.

        Parameters
        ----------
        step
            Step value for which the quantity is computed.
        tag
            Tag for the quantity to be logged.
        value
            Contains only 1 element representing the quantity to be logged
        data_type: {'audio', 'image', 'histogram', 'text'}
            Data type of the quantity to be logged.
        kwargs
            Any other kwargs accepted by the operation that writes `data_type` to the TensorBoard.
        """

        if isinstance(self._summary_functions[data_type], functools.partial):
            summary_op = self._summary_functions[data_type].func
        else:
            summary_op = self._summary_functions[data_type]

        with self.writer.as_default():
            summary_op(name=tag, data=value, step=step, **kwargs)

        self.writer.flush()

    def update_data_type_kwargs(self, data_type: str, kwargs: Dict[str, Any]) -> None:
        """
        Updates the keyword arguments the writer uses for writing the operations. Used to override the default
        kwargs for the operations. Positional arguments (e.g., sampling_frequency for audio signals) can also be
        overriden if specified by name.

        data_type: {'audio', 'image', 'histogram', 'text'}
            Data type for which the kwargs of the writer function have to be overridden
        kwargs
            A dictionary of the form::

                {
                'buckets': 5,
                }
            where the key is the kwarg name and the value is the value of the kwarg.
        """

        self._summary_functions_kwargs[data_type] = kwargs
        self._summary_functions[data_type] = partial(self._summary_functions[data_type].func, **kwargs)

    def record_graph(self):
        raise NotImplementedError("Graph logging is not supported!")
