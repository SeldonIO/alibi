import argparse
import cProfile
import importlib
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from operator import methodcaller
from timeit import default_timer as timer
from typing import Any, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import alibi
import spacy

import benchmark.utils.models as models
import ruamel.yaml as yaml

from alibi.explainers import AnchorImage, AnchorTabular, AnchorText, DistributedAnchorTabular
from alibi.utils.distributed import check_ray
from alibi.utils.download import spacy_model
from alibi.utils.wrappers import Predictor
from benchmark.utils.data import FashionMnistProcessor, ImageNetPreprocessor
from nested_lookup import nested_lookup

SUPPORTED_EXPLAINERS = ['tabular', 'text', 'image']
SUPPORTED_DATASETS = ['adult', 'imagenet', 'movie_sentiment', 'fashion_mnist', 'imagenet']
SUPPORTED_CLASSIFIERS = ['rf', 'lr', 'fashion_mnist_cnn', 'InceptionV3']
BUILTIN_SEGMENTATIONS = ['felzenszwalb', 'slic', 'quickshift']

# TODO: Typing and documentation
# TODO: opts not used currently in preprocess_adult, pipelines hardcoded
# TODO: in the future one should be able to pass their own classifier object or object name and loading
#  function as opp to using fit_*
# TODO: make it more flexible for users to configure the name of their experiments from configuration file
# TODO: For classifier model, it would be nice if one specified the module and we would use getattr/method caller
#  to work with the model inside the correct function?
# TODO: Specifying a custom preprocesor operating on the input data (or subset) should be the first thing that's
#   checked in a generic pre-process fcn. If that's not specified, then it should be deferred to a specific one
#  (i.e. implement this in the context manager)
# TODO: The preprocessors for all data should work by specifying the module, preprocessor and options for all
#  data (so we should not have a preprocess_* or custom arguments, just define a function that takes the data as an arg, its args and
#  and kwargs and that's it)
# TODO: In the future the classifier/model should be specified via module + class name or fcn name. A separate
#  fit option with args and kwargs to pass to the fit_* function. Raise value error if fit_* doesn't exist in utils.models


class Timer:
    def __init__(self):
        pass

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        self.t_elapsed = timer() - self.start


def load_dataset(*, dataset='adult', load_opts=None):

    args, kwargs = (), {}
    method = 'fetch_{}'.format(dataset)
    if load_opts:
        if load_opts['args']:
            args = tuple(load_opts['args'])
        if load_opts['kwargs']:
            kwargs = load_opts['kwargs']
    caller = methodcaller(method, *args, **kwargs)
    dataset = caller(alibi.datasets)

    return dataset


def split_data(dataset, opts):

    seed = opts['seed']
    method = opts['method']
    print("Splitting using {} method ...".format(method))
    if method == 'shuffle':
        n_train_records = opts['n_train_records']
        return _shuffle(dataset, n_train_records, seed)
    elif method == 'train_test_val':
        split_fractions = {
            'test': opts['test_size'],
            'val': opts['val_size'],
        }
        return _train_test_val(dataset, seed, split_fractions)
    else:
        return _auto(dataset)


def _shuffle(dataset, n_train_records, seed):

    np.random.seed(seed)
    fmt = "{} records included in train split..."
    print(fmt.format(n_train_records))
    data, target = dataset.data, dataset.target
    data_perm = np.random.permutation(np.c_[data, target])
    data = data_perm[:, :-1]
    target = data_perm[:, -1]
    X_train, Y_train = data[:n_train_records, :], target[:n_train_records]
    X_test, Y_test = data[n_train_records + 1:, :], target[n_train_records + 1:]

    return {'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test
            }


def _train_test_val(dataset, seed, split_fractions, split_val=True):

    data, labels = dataset.data, dataset.target
    np.random.seed(seed)
    test_size = split_fractions['test']
    print("... test size is: {}.".format(test_size))
    train, test, train_labels, test_labels = train_test_split(data,
                                                              labels,
                                                              test_size=split_fractions['test'],
                                                              random_state=seed,
                                                              )
    splits = {
        'X_train': train,
        'Y_train': np.array(train_labels),
        'X_test': test,
        'Y_test': np.array(test_labels),
    }

    if split_val:
        val_size = split_fractions['val']
        print("... val size is: {}.".format(val_size))
        train, val, train_labels, val_labels = train_test_split(train,
                                                                train_labels,
                                                                test_size=val_size,
                                                                random_state=seed,
                                                                )
        val_labels = np.array(val_labels)
        splits.update(
            [('X_train', train),
             ('Y_train', train_labels),
             ('X_val', val),
             ('Y_val', val_labels)
             ]
        )

    return splits


def _auto(dataset):

    data = dataset.data
    (X_train, y_train), (X_test, y_test) = data.load_data()

    return {'X_train': X_train,
            'Y_train': y_train,
            'X_test': X_test,
            'Y_test': y_test
            }


def get_preprocessor(module, preprocessor, dataset=None):

    module = importlib.import_module(module)
    preprocessor_callable = getattr(module, preprocessor)
    if dataset:
        raise NotImplementedError("Fitting preprocessor to training data is not implemented!")

    return preprocessor_callable

    # TODO: dataset is passed as a kwarg in case the preprocessor
    #  needs to be fit to the data (to be developed in the future)


def preprocess_adult(dataset, splits, opts=None):

    feature_names = dataset.feature_names
    category_map = dataset.category_map
    X_train = splits['X_train']

    ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
    ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])

    categorical_features = list(category_map.keys())
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                                   ('cat', categorical_transformer, categorical_features)])
    preprocessor.fit(X_train)

    return preprocessor


def preprocess_movie_sentiment(dataset, splits, opts=None):

    X_train = splits['X_train']
    preprocessor = CountVectorizer(min_df=opts['min_df'])
    preprocessor.fit(X_train)

    return preprocessor


def preprocess_fashion_mnist(dataset, splits, opts=None):

    # TODO: Can we do better in terms of splitting the pre-processing and the model?
    #   Other explainers use the pre-processor, this one does not ...
    preprocessor = FashionMnistProcessor(**opts)

    return preprocessor


def preprocess_imagenet(dataset, splits, opts=None):

    if 'custom' in opts:
        return getattr(eval(opts['custom']['module']), opts['custom']['function'])
    else:
        preprocessor = ImageNetPreprocessor()

    return preprocessor


def display_performance(splits, predictor):
    print('Train accuracy: ', accuracy_score(splits['Y_train'], predictor(splits['X_train'])))
    print('Test accuracy: ', accuracy_score(splits['Y_test'], predictor(splits['X_test'])))


def predict_fcn(clf, preprocessor=None):
    if preprocessor:
        return lambda x: clf.predict(preprocessor.transform(x))

    return lambda x: clf.predict(x)


def fit_rf(config, splits=None, preprocessor=None):

    print("Fitting classifier ...")

    np.random.seed(config['seed'])
    clf = RandomForestClassifier(n_estimators=config['n_estimators'])
    if preprocessor:
        clf.fit(preprocessor.transform(splits['X_train']), splits['Y_train'])
    else:
        clf.fit(splits['X_train'], splits['Y_train'])

    display_performance(splits, Predictor(clf, preprocessor=preprocessor))

    return Predictor(clf, preprocessor=preprocessor)


def fit_lr(config, splits=None, preprocessor=None):

    print("Fitting classifier ...")
    np.random.seed(config['seed'])
    clf = LogisticRegression(solver=config['solver'])
    if preprocessor:
        clf.fit(preprocessor.transform(splits['X_train']), splits['Y_train'])
    else:
        clf.fit(splits['X_train'], splits['Y_train'])

    display_performance(splits, Predictor(clf, preprocessor=preprocessor))

    return Predictor(clf, preprocessor=preprocessor)


def fit_fashion_mnist_cnn(config, splits=None, preprocessor=None):

    clf = models.fashion_mnist_cnn()
    if preprocessor:
        clf.fit(
            preprocessor.transform_x(splits['X_train']),
            preprocessor.transform_y(splits['Y_train']),
            batch_size=config['batch_size'],
            epochs=config['epochs'],
        )
        score = clf.evaluate(
            preprocessor.transform_x(splits['X_test']),
            preprocessor.transform_y(splits['Y_test']),
        )
    else:
        clf.fit(
            splits['X_train'],
            splits['Y_train'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
        )
        score = clf.evaluate(splits['X_test'], splits['Y_test'])

    print('Test accuracy: ', score[1])

    return Predictor(clf)


def fit_InceptionV3(config, splits=None, preprocessor=None):

    # TODO: This should be specified by a loading function, fit is
    #  temporary solution ...
    module = importlib.import_module(config['module'])
    model = getattr(module, config['name'])
    clf = model(weights=config['weights'])

    return Predictor(clf)


def get_tabular_explainer(predictor, config, dataset=None, split=None):

    if not dataset or not split:
        raise ValueError("Anchor Tabular requires both a dataset object and"
                         " a dictionary with datasets split for fitting but"
                         " at least one was not passed to get_tabular_function!")

    feature_names = dataset.feature_names
    category_map = dataset.category_map
    X_train = split['X_train']

    if check_ray():
        if config['parallel']:
            explainer = DistributedAnchorTabular(predictor,
                                                 feature_names,
                                                 categorical_names=category_map,
                                                 seed=config['seed'],
                                                 )
        else:
            explainer = AnchorTabular(predictor,
                                      feature_names,
                                      categorical_names=category_map,
                                      seed=config['seed'],
                                      )
    else:
        explainer = AnchorTabular(predictor,
                                  feature_names,
                                  categorical_names=category_map,
                                  seed=config['seed'],
                                  )

    explainer.fit(X_train,
                  **config
                  )

    return explainer


def get_text_explainer(predictor, config, dataset=None, split=None):

    model_name = config['pert_model']
    spacy_model(model=model_name)
    perturbation_model = spacy.load(model_name)

    return AnchorText(perturbation_model, predictor, seed=config['seed'])


def get_image_explainer(predictor, config, dataset=None, split=None):

    if config['segmentation_fn'] in BUILTIN_SEGMENTATIONS:
        segmentation_fn = config['segmentation_fn']
    else:
        try:
            segmentation_fn = getattr(models, config['segmentation_fn'])
        except AttributeError:
            print("The segmentation function {} was not found in benchmark.utils.models."
                  "Please ensure it is implemented there!".format(config['segmentation_fn']))
            print("Defaulting to 'slic' segmentation with default kwargs")
            segmentation_fn = 'slic'

    if split:
        shape_instance = split['X_train'][config['shape_idx']]
    else:
        shape_instance = dataset.data[config['shape_idx']]

    # add singleton channel dimension
    if shape_instance.ndim == 2:
        shape_instance = shape_instance[..., np.newaxis]

    image_shape = shape_instance.shape

    if 'segmentation_kwargs' in config:
        segmentation_kwargs = config['segmentation_kwargs']
    else:
        segmentation_kwargs = None

    explainer = AnchorImage(
        predictor,
        image_shape,
        segmentation_fn=segmentation_fn,
        segmentation_kwargs=segmentation_kwargs,
        seed=config['seed'],
    )

    return explainer


def get_explanation(explainer, instance, expln_config):
    # TODO: not nice to pass random crap as kwargs to the explain, separate that out in yaml
    return explainer.explain(instance,
                             **expln_config,
                             )


def _display_prediction(predict_fn, instance_id, splits, dataset):
    class_names = dataset.classnames
    X_test = splits['X_test']
    print('Prediction: ', class_names[predict_fn(X_test[instance_id].reshape(1, -1))[0]])


def _adult_prediction(predictor, instance, dataset):
    class_names = dataset.target_names
    pred = class_names[predictor(instance.reshape(1, -1))[0]]
    alternative = class_names[1 - predictor(instance.reshape(1, -1))[0]]
    print('Prediction: ', pred)
    print('Alternative', alternative)
    return pred, alternative


def _movie_sentiment_prediction(predictor, instance, dataset):
    class_names = dataset.target_names
    pred = class_names[predictor([instance])[0]]
    alternative = class_names[1 - predictor([instance])[0]]
    print('Prediction: ', pred)
    print('Alternative:', alternative)
    return pred, alternative


def _fashion_mnist_prediction(predictor, instance, dataset):

    instance = instance[np.newaxis, ...]
    class_names = dataset.target_names
    pred = class_names[predictor(instance).argmax()]
    print('Prediction:', pred)

    return pred, None


def _imagenet_prediction(predictor, instance, dataset):

    from tensorflow.keras.applications.inception_v3 import decode_predictions
    instance = instance[np.newaxis, ...]
    pred = decode_predictions(predictor(instance), top=3)[0]
    print('Prediction:', pred)

    return pred, None


def display_explanation(pred, alternative, explanation, explainer_type, show_covered=False):

    # TODO: This should be done via a registered function for each type of classifier
    if explainer_type != 'image':
        print('Anchor: %s' % (' AND '.join(explanation['names'])))
    else:
        plt.imshow(explanation['anchor'][:, :, 0])
    print('Precision: %.2f' % explanation['precision'])
    print('Coverage: %.2f' % explanation['coverage'])

    if show_covered:
        print('\nExamples where anchor applies and model predicts {}'.format(pred))
        print('\n'.join([x for x in explanation['raw']['examples'][-1]['covered_true']]))
        print('\nExamples where anchor applies and model predicts {}'.format(alternative))
        print('\n'.join([x for x in explanation['raw']['examples'][-1]['covered_false']]))


class ExplainerExperiment(object):

    def __init__(self, *args, **kwargs):

        self.dataset_name = kwargs['data']['dataset']
        self.data_config = kwargs['data']
        self.explainer_config = kwargs['explainer']
        self.experiment_config = kwargs['experiment']
        self.clf_config = kwargs['classifier']

        self._all_configs = [kwargs[key] for key in kwargs.keys()]
        self._this_module = sys.modules[__name__]
        self._default_data_store = {
            'commit_hash': self.experiment_config['commit_hash'],
            't_elapsed': [],
            'clf_config': {},
            'exp_config': {},
            'expln_config': {},
            'data_config': {},
        }
        self._data_fields = self.experiment_config['save']['fields']
        self._data_mapping = self.experiment_config['save']['mapping']
        # self._data_store contains the fields specified in experiment settings
        # in addition to the _default_data_store fields
        self._create_data_store()

        self.explainer = None
        self.explainer_type = kwargs['explainer']['type']
        self.instance = None
        self.splits = None

    def __enter__(self):

        # load and optionally split dataset
        load_opts = None
        if self.data_config['load_opts']:
            load_opts = {
                'args': self.data_config['load_opts']['args'],
                'kwargs': self.data_config['load_opts']['kwargs'],
            }
        dataset = load_dataset(dataset=self.dataset_name, load_opts=load_opts)

        if self.data_config['split']:
            splits = split_data(dataset, self.data_config['split_opts'])
            self.splits = splits
        self.dataset = dataset

        # optionally preprocess the dataset
        # TODO: messy, tidy up by specifying the preprocess function in yaml directly
        if self.data_config['preprocess']:
            preproc_opts = self.data_config['preprocess_opts']
            custom = 'custom' in preproc_opts if isinstance(preproc_opts, dict) else False
            if custom:
                    obj = get_preprocessor(
                        preproc_opts['module'],
                        preproc_opts['preprocessor']['name'],
                    )
                    # TODO: There has to be a better way to do this
                    if not preproc_opts['preprocessor']['args']:
                        args = ()
                    else:
                        args = preproc_opts['preprocessor']['args']
                    if not preproc_opts['preprocessor']['kwargs']:
                        kwargs = {}
                    else:
                        kwargs = preproc_opts['preprocessor']['kwargs']

                    if preproc_opts['preprocessor']['type'] == 'obj':
                        preproc = obj(*args, **kwargs)  # initialise preprocessor
                    else:
                        preproc = obj
            else:
                preprocess_fcn = 'preprocess_{}'.format(self.dataset_name)
                preproc = getattr(self._this_module, preprocess_fcn)(
                    dataset,
                    self.splits,
                    opts=self.data_config['preprocess_opts'],
                )
        else:
            preproc = None

        # fit classifier
        clf_fcn = 'fit_{}'.format(self.clf_config['name'])
        self.predictor = getattr(self._this_module, clf_fcn)(
            self.clf_config,
            splits=self.splits,
            preprocessor=preproc,
        )

        # retrieve instance to be explained
        instance_idx = self.experiment_config['instance_idx']
        instance_split = self.experiment_config['instance_split']
        if instance_split:
            self.instance = splits['X_{}'.format(instance_split)][instance_idx]
        else:
            self.instance = dataset.data[instance_idx]
        if self.experiment_config['transform_instance']:
            self.instance = preproc(self.instance)

        # create and fit explainer instance
        explainer_fcn = 'get_{}_explainer'.format(self.explainer_config['type'])
        explainer = getattr(self._this_module, explainer_fcn)(
            self.predictor,
            self.explainer_config,
            dataset,
            self.splits,
        )
        self.explainer = explainer

        return self

    def __exit__(self, *args):

        if self.experiment_config['test_only']:
            print("WARNING: test_only set to true in experiment config, "
                  "experiment output .pkl will not be saved!")
        else:
            self._save_exp_metadata()

            if not os.path.exists(self.experiment_config['ckpt_dir']):
                os.makedirs(self.experiment_config['ckpt_dir'])
            else:
                print("WARNING: Checkpoint directory already exists, "  # TODO: Setup logging 
                      "files may be overwritten!")

            exp_config = self.experiment_config
            ckpt_name = self._get_ckpt_name(
                exp_config['ckpt_prefix'],
                exp_config['name'],
                self._all_configs
            )
            fullpath = os.path.join(exp_config['ckpt_dir'], ckpt_name)
            fullpath = fullpath if fullpath.split(".")[-1] == 'pkl' else fullpath + '.pkl'

            with open(fullpath, 'wb') as f:
                pickle.dump(self._data_store, f)

    def _read_recursive(self, data: dict, fields: Sequence) -> Any:
        if len(fields) == 1:
            return data[fields[0]]
        return self._read_recursive(data[fields[0]], fields[1:])

    def _create_data_store(self):
        self._data_store = {field: [] for field in self._data_fields}
        self._data_store.update(self._default_data_store)

    @staticmethod
    def _get_ckpt_name(prefix, name_mapping, all_configs):

        placeholder = '{}'
        ckpt_fmt = ''
        values = [prefix]

        for name, abbrev in name_mapping.items():
            values.append(nested_lookup(name, all_configs)[0])
            if abbrev:
                name = abbrev
            ckpt_fmt += name + '_' + placeholder + '_'

        ckpt_fmt = ckpt_fmt.rstrip("_")
        ckpt_fmt = placeholder + '_' + ckpt_fmt

        return ckpt_fmt.format(*values)

    def _save_exp_metadata(self):
        self._data_store['data_config'] = self.data_config
        self._data_store['clf_config'] = self.clf_config
        self._data_store['expln_config'] = self.explainer_config
        self._data_store['exp_config'] = self.experiment_config

    def update(self, explanation, t_elapsed):
        for field in self._data_fields:
            if field in self._data_mapping:
                data = self._read_recursive(
                    explanation,
                    self._data_mapping[field],
                )
                self._data_store[field].append(data)
            else:
                self._data_store[field].append(explanation[field])
        self._data_store['t_elapsed'].append(t_elapsed)


def display(config, explanation, exp):

    if config['experiment']['verbose']:
        pred_fcn = '_{}_prediction'.format(config['data']['dataset'])
        # display prediction on instance to be explained
        pred, alternative = getattr(exp._this_module, pred_fcn)(
            exp.predictor,
            exp.instance,
            exp.dataset,
        )
        display_explanation(pred,
                            alternative,
                            explanation,
                            exp.explainer_type,
                            show_covered=config['experiment']['show_covered'])
    return


def check(config):

    if config['explainer']['type'] not in SUPPORTED_EXPLAINERS:
        raise NotImplementedError("Experiments are supported only for tabular data!")
    if config['data']['dataset'] not in SUPPORTED_DATASETS:
        raise ValueError("Only datasets adult/imagenet/movie_sentiment are supported")
    if config['classifier']['name'] not in SUPPORTED_CLASSIFIERS:
        raise NotImplementedError("Only random forest classifiers are supported!")

    if config['explainer']['type'] == 'tabular':
        if config['experiment']['verbose']:
            if config['experiment']['show_covered']:
                raise NotImplementedError("Human readable covered true and false examples"
                                          "not implemented for AnchorTabular. Please implement"
                                          "or set show_covered=False in experiment confinguration"
                                          "to continue.")

    if config['data']['split_opts']['method'] == 'auto' and config['data']['dataset'] != 'fashion_mnist':
        raise ValueError('auto splitting method is only implemented for Fashion MNIST dataset!')


def run_experiment(config):

    n_runs = int(config['experiment']['n_runs'])
    check(config)

    with ExplainerExperiment(**config) as exp:
        for _ in range(n_runs):
            with Timer() as time:
                explanation = get_explanation(exp.explainer, exp.instance, exp.explainer_config)
            exp.update(explanation, time.t_elapsed)
            display(config, explanation, exp)


def profile(config):

    prof_dir = config['experiment']['profile_dir']
    if not os.path.exists(prof_dir):
        os.makedirs(prof_dir)
    else:
        print("WARNING: Profiler output directory already exits!"
              "Files may be overwritten!!!")
    prof_fullpath = os.path.join(prof_dir, config['experiment']['profile_out'])
    result = []

    with ExplainerExperiment(**config) as exp:
        cProfile.runctx('result.append(get_explanation(exp.explainer, exp.instance, exp.explainer_config))',
                        locals(),
                        globals(),
                        filename=prof_fullpath,
                        )
        explanation = result[0]

    display(config, explanation, exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anchor Explanations Experiments')
    parser.add_argument("--config",
                        nargs="?",
                        type=str,
                        default="configs/config.yaml",
                        help="Configuration file for the experiment.",
                        )
    parser.add_argument("--hash",
                        type=str,
                        help="If passed, the commit hash is stored with the experimental data "
                             "to allow reproducing experiment results. Use `git rev-parse HEAD`"
                             " to get the commit hash for your current branch.")
    args = parser.parse_args()

    with open(args.config) as fp:
        configuration = yaml.load(fp)

    configuration['experiment']['commit_hash'] = args.hash

    check(configuration)

    if configuration['experiment']['profile']:
        profile(configuration)
    else:
        run_experiment(configuration)
