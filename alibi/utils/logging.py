from collections import namedtuple
from typing import Dict, Any, List

tensorboard_loggers = {'pytorch': None, 'tensorflow': None}
"""
dict: A registry that contains TensorBoard writers for PyTorch and TensorFlow. This registry can be imported and used to
return a TensorboardWriter object for the desired framework.
"""


def tensorboard_logger(obj):
    """
    A decorator that adds a TensorBoard writer to a registry. Implementations requiring the writer should import the
    registry, choose the writer for the framework the algorithm needs and initialise it.
    """
    tensorboard_loggers[obj.framework] = obj
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
