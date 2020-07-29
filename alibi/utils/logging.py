from collections import namedtuple
from typing import Dict, Any, List

DEFAULT_LOGGING_OPTS = {
    'verbose': False,
    'log_traces': True,
    'trace_dir': 'tb_logs/',
    'summary_freq': 1,
    'image_summary_freq': 10,
    'tracked_variables':  {'tags': [], 'data_types': [], 'descriptions': []},
}
"""
dict: The default values for logging options.


    - 'verbose': if `True` the logger (a `logging.Loggger` instance) will be set to ``DEBUG`` level. 

    - 'log_traces': if `True`, data about the optimisation process will be logged to TensorBoard with a frequency \
    specified  by `summary_freq` input. The algorithm will also log images if `X` is a 4-dimensional tensor \
    (corresponding to a leading dimension of 1 and `(H, W, C)` dimensions) with a frequency specified by 
    `image_summary_freq`. For each `explain` run, a subdirectory of `trace_dir` with `run_{}` is created and  {} is \
    replaced by the run number. To see the logs run the command in the `trace_dir` \
    directory:: 

        ``tensorboard --logdir tb_logs``
     replacing ``trace_dir`` with your own path. Then run ``localhost:6006`` in your browser to see the traces. The \
     traces can be visualised as the optimisation proceeds and can provide useful information on how to adjust the \
     optimisation in cases of non-convergence of fitting or explainers.

    - 'trace_dir': the directory where the optimisation infromation is logged. 

    - 'summary_freq': logging frequency for optimisation information.

    - 'image_summary_freq': logging frequency for intermediate for image data
    
    - 'tracked_variables': This should be used to specify what will be logged:
    
            * 'tags': a list of tags with variable names (e.g., ``['training/loss', 'training'/accuracy]``). To log \
            these quantities to TensorBoard, one should do the following::
                    
                    self.data_store['loss'] = loss_val
                    self.data_store['accuracy'] = accuracy
                    self.tensorboard.record(self.step, self.data_store)
                    
            Note that the quantities will not be logged unless the string after the last / in the tag name matches the \
            key of `self.data_store`. 
 
            * 'data_types': a list of data types for logged quantities. For the above example, the list would be \
            ``['scalar', 'scalar']``. Other supported data types are 'audio', 'image', 'text' and 'histogram'.
            
            * 'descriptions': A list of optional descriptions. To skip a descriptions, for the above example the list \
            would be ``['', '']`` and to add a description for the first variable it would be ``['my first var', '']``. 

See the documentation for `alibi.utils.logging.TensorboardWriterBase` and 
`alibi.utils.tensorflow.logging.TFTensorboardWriter` for more details about the logs. 

Any subset of these options can be overridden by passing a dictionary with the corresponding subset of keys when calling
`explain` method. 


Examples
--------
To specify a the name logging directory for TensorFlow event files and a new frequency for logging images call `explain`
with the following key-word arguments::

    logging_opts = {'trace_dir': 'experiments/cf', 'image_summary_freq': 5}
"""

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


class TensorboardWriterBase:

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
        """
        Sets up the writer by:

            - Creating a `SummaryFileWriter` object that writes data to event file
            - Creating a mapping from the variable tag and type to the callable to the summary function that can write \
            it to the event file
        """

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
