import argparse
import cProfile
import os
import pickle
import sys
import yaml

import numpy as np

from operator import methodcaller
from timeit import default_timer as timer
from typing import Any, Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from alibi.explainers import AnchorTabular
import alibi.datasets as datasets

SUPPORTED_EXPLAINERS = ['tabular']
SUPPORTED_DATASETS = ['adult', 'imagenet', 'movie_sentiment']
SUPPORTED_CLASSIFIERS = ['rf']

# TODO: Typing and documentation


class Timer:
    def __init__(self):
        pass

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        self.t_elapsed = timer() - self.start


def load_dataset(*, dataset='adult'):

    method = 'fetch_{}'.format(dataset)
    caller = methodcaller(method)
    dataset = caller(datasets)

    return dataset


def split_data(dataset, opts):

    data, target = dataset.data, dataset.target
    np.random.seed(opts['seed'])

    data_perm = np.random.permutation(np.c_[data, target])
    data = data_perm[:, :-1]
    target = data_perm[:, -1]

    idx = opts['max_records']
    X_train, Y_train = data[:idx, :], target[:idx]
    X_test, Y_test = data[idx + 1:, :], target[idx + 1:]

    return {'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test
            }


def preprocess_adult(dataset, splits):

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


def display_performance(splits, predict_fn):
    print('Train accuracy: ', accuracy_score(splits['Y_train'], predict_fn(splits['X_train'])))
    print('Test accuracy: ', accuracy_score(splits['Y_test'], predict_fn(splits['X_test'])))


def predict_fcn(clf, preprocessor=None):
    if preprocessor:
        return lambda x: clf.predict(preprocessor.transform(x))
    return lambda x: clf.predict(x)


def fit_rf(splits, config, preprocessor=None):

    np.random.seed(config['seed'])
    clf = RandomForestClassifier(n_estimators=config['n_estimators'])
    clf.fit(preprocessor.transform(splits['X_train']), splits['Y_train'])

    display_performance(splits, predict_fcn(clf, preprocessor=preprocessor))

    return predict_fcn(clf, preprocessor=preprocessor)


def get_tabular_explainer(predict_fn, dataset, split, config):

    feature_names = dataset.feature_names
    category_map = dataset.category_map
    X_train = split['X_train']

    explainer = AnchorTabular(predict_fn, feature_names, categorical_names=category_map, seed=config['seed'])
    explainer.fit(X_train, disc_perc=config['disc_perc'])

    return explainer


def display_prediction(predict_fn, instance_id, splits, dataset):
    class_names = dataset.classnames
    X_test = splits['X_test']
    print('Prediction: ', class_names[predict_fn(X_test[instance_id].reshape(1, -1))[0]])


def get_explanation(explainer, expln_config, splits, exp_config):
    instance_id = exp_config['instance_idx']

    return explainer.explain(splits['X_test'][instance_id],
                             threshold=expln_config['threshold'],
                             verbose=expln_config['verbose'],
                             parallel=expln_config['parallel'],
                             )


def display_explanation(explanation):
    print('Anchor: %s' % (' AND '.join(explanation['names'])))
    print('Precision: %.2f' % explanation['precision'])
    print('Coverage: %.2f' % explanation['coverage'])


class ExplainerExperiment(object):

    def __init__(self, *, dataset, explainer, classifier, experiment):

        self.dataset = dataset
        self.explainer_config = explainer
        self.experiment_config = experiment
        self.clf_config = classifier

        self._this_module = sys.modules[__name__]
        self._default_data_store = {'commit_hash': self.experiment_config['commit_hash'],
                                    't_elapsed': [],
                                    'clf_config': {},
                                    'exp_config': {},
                                    'expln_config': {},
                                    }
        self._data_fields = self.experiment_config['save']['fields']
        self._data_mapping = self.experiment_config['save']['mapping']
        # self._data_store contains the fields specified in experiment settings
        # in addition to the _default_data_store fields
        self._create_data_store()

        self.explainer = None
        self.splits = None

    def __enter__(self):

        # load and split dataset
        dataset = load_dataset(dataset=self.dataset)
        splits = split_data(dataset, self.experiment_config['data']['split_opts'])
        self.splits = splits

        # optionally preprocess the dataset
        if self.experiment_config['data']['preprocess']:
            preprocess_fcn = 'preprocess_{}'.format(self.dataset)
            preprocessor = getattr(self._this_module, preprocess_fcn)(dataset, splits)
        else:
            preprocessor = None

        # fit classifier
        clf_fcn = 'fit_{}'.format(self.clf_config['name'])
        predict_fn = getattr(self._this_module, clf_fcn)(splits, self.clf_config, preprocessor=preprocessor)
        explainer_fcn = 'get_{}_explainer'.format(self.explainer_config['type'])

        # create and fit explainer instance
        explainer = getattr(self._this_module, explainer_fcn)(predict_fn, dataset, splits, self.explainer_config)
        self.explainer = explainer

        return self

    def __exit__(self, *args):
        self._save_exp_metadata()

        if not os.path.exists(self.experiment_config['ckpt_dir']):
            os.makedirs(self.experiment_config['ckpt_dir'])
        else:
            print("WARNING: Checkpoint directory already exists, "  # TODO: Setup logging
                  "files may be overwritten!")

        fullpath = os.path.join(self.experiment_config['ckpt_dir'],
                                self.experiment_config['ckpt'])
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

    def _save_exp_metadata(self):
        self._data_store['clf_config'] = self.clf_config
        self._data_store['expln_config'] = self.explainer_config
        self._data_store['exp_config'] = self.experiment_config

    def update(self, explanation, t_elapsed):
        for field in self._data_fields:
            if field in self._data_mapping:
                data = self._read_recursive(explanation,
                                            self._data_mapping[field])
                self._data_store[field].append(data)
            else:
                self._data_store[field].append(explanation[field])
        self._data_store['t_elapsed'].append(t_elapsed)


def run_experiment(config):

    n_runs = int(config['experiment']['n_runs'])

    with ExplainerExperiment(**config) as exp:
        for _ in range(n_runs):
            with Timer() as time:
                explanation = get_explanation(exp.explainer,
                                              exp.explainer_config,
                                              exp.splits,
                                              exp.experiment_config,
                                              )
            exp.update(explanation, time.t_elapsed)
            if config['experiment']['verbose']:
                display_explanation(explanation)
    return


def profile(config):

    prof_dir = config['experiment']['profile_dir']
    if not os.path.exists(prof_dir):
        os.makedirs(prof_dir)
    else:
        print("WARNING: Profiler output directory already exits!"
              "Files may be overwritten!!!")
    prof_fullpath = os.path.join(prof_dir, config['experiment']['profile_out'])

    with ExplainerExperiment(**config) as exp:
        cProfile.runctx('get_explanation(exp.explainer, exp.explainer_config, exp.splits, exp.experiment_config)',
                        locals(),
                        globals(),
                        filename=prof_fullpath,
                        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anchor Explanations Experiments')
    parser.add_argument("--config",
                        nargs="?",
                        type=str,
                        default="configs/config.yaml",
                        help="Configuration file for the experiment",
                        )
    parser.add_argument("--hash",
                        type=str,
                        help="If passed, the commit hash is stored with the experimental data "
                             "to allow reproducing experiment results")
    args = parser.parse_args()

    with open(args.config) as fp:
        configuration = yaml.load(fp, Loader=yaml.FullLoader)

    configuration['experiment']['commit_hash'] = args.hash

    if configuration['explainer']['type'] not in SUPPORTED_EXPLAINERS:
        raise NotImplementedError("Experiments are supported only for tabular data!")
    if configuration['dataset'] not in SUPPORTED_DATASETS:
        raise ValueError("Only datasets adult/imagenet/movie_sentiment are supported")
    if configuration['classifier']['name'] not in SUPPORTED_CLASSIFIERS:
        raise NotImplementedError("Only random forest classifiers are supported!")

    if configuration['experiment']['profile']:
        profile(configuration)
    else:
        run_experiment(configuration)
