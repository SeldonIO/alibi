import functools
import os
from collections import namedtuple
from functools import partial
from typing import Dict, Any, List, Union

import numpy as np
import tensorflow as tf


tensorboard_loggers = []


# TODO: FUNCTOOLS.WRAPS?
def tensorboard_logger(obj):
    tensorboard_loggers.append(obj)
    return obj


def _get_var_name(tag: str) -> str:
    """
    Returns the name of the variable to be logged given a tag for that variable.
    """
    if '/' in tag:
        return tag.split("/")[-1]
    return tag


class TensorboardWriter:

    def __init__(self):

        self.run = 0
        self.tracked_variables = []
        self._vars_mapping = {}
        self._variable_metadata = namedtuple('variable_metadata', 'tag var_type description')
        self.summary_freq = 1
        self.image_summary_freq = 1
        self.trace_dir = ''

        self._init_summaries()

    def _init_summaries(self):
        """
        Create a mapping from the input data types to the summary operations that can write the data types to
        TensorBoard.
        """
        raise NotImplementedError("Subclass must define summary operations!")

    def setup(self, logging_opts: Dict[str, Any]):

        # TODO: DOCSTRING

        self.logging_opts = logging_opts
        if self.logging_opts['log_traces']:
            self.trace_dir = logging_opts['trace_dir']
            self.summary_freq = logging_opts['summary_freq']
            self.image_summary_freq = logging_opts['image_summary_freq']
            self._create_writer()
            # NB: Variables mappings and tracking variables lists are calculated only once.
            # That means that overriding the tracked variables list is not supported at explain time
            if not self._vars_mapping:
                self._vars_mapping = self._init_variables_mapping(logging_opts['tracked_variables'])
                self.tracked_variables = self._vars_mapping.keys()

        return self

    def _create_writer(self):
        """
        Creates a SummaryWriter object that writes data to event files.
        """
        raise NotImplementedError("Subclass must implement writer creation!")

    def _init_variables_mapping(self, tracked_variables: Dict[str, List[str]]):
        """
        Creates a mapping from the name of the variable to be logged as known by the recorder calling context to the
        a record containing the tag, type and the descriptions of the variable.
        """

        vars_mapping = {}
        tags = tracked_variables['tags']
        types = tracked_variables['data_types']
        descriptions = tracked_variables['descriptions']
        if len(tags) != len(types) or len(tags) != len(descriptions):
            raise ValueError(
                "Incorrect definition for logged variables. The 'tags', 'data_types' and 'descriptions' fileds of the"
                "'tracked_variables' fields must be of the same length"
            )
        for tag, data_type, description in zip(tags, types, descriptions):
            var_name = _get_var_name(tag)
            vars_mapping[var_name] = self._variable_metadata(tag=tag, var_type=data_type, description=description)

        return vars_mapping

    def record(self, step: int, data_store: Dict[str, Any], prefix: str = ''):
        """
        Writes the quantities in the `data_store` to the TensorBoard. A quantity will be written only
        if it is registered in the ``tracked_variables`` part of the logging configuration and the variable
        name (i.e., what follows the last / in the tag) is a key in `data_store`.

        Parameters
        ----------
        step
            Step value for which the quantities are logged.
        data_store
            A dictionary with the structure::

                {
                    'var_name': [value],
                }

            It should contain all the variables to be logged. For a variable to be logged, it has to be defined in
            the ``'tracked_variables'`` attribute of the default logging configuration that initialises this object.
            If you don't want to update the logging configuration, use `record_step` instead.
        prefix
            Allows prefixing all tags in the store. Useful for logging the same quantities under different contexts
            (e.g., train/valid).
        """
        raise NotImplementedError("Subclass must implement record method to write the data to TensorBoard!")


@tensorboard_logger
class TFTensorboardWriter(TensorboardWriter):

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
