import argparse
import ast
import json
import os
import pickle
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from operator import methodcaller
from timeit import default_timer as timer

from alibi.explainers import AnchorTabular
import alibi.datasets as datasets

SUPPORTED_EXPLAINERS = ['tabular']
SUPPORTED_DATASETS = ['adult', 'imagenet', 'movie_sentiment']
SUPPORTED_CLASSIFIERS = ['rf']


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


def split_data(dataset, seed=0, idx=30000):

    data, target = dataset.data, dataset.target
    np.random.seed(seed)

    data_perm = np.random.permutation(np.c_[data, target])
    data = data_perm[:, :-1]
    target = data_perm[:, -1]

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

    np.random.seed(int(config['seed']))
    clf = RandomForestClassifier(n_estimators=int(config['n_estimators']))
    clf.fit(preprocessor.transform(splits['X_train']), splits['Y_train'])

    display_performance(splits, predict_fcn(clf, preprocessor=preprocessor))

    return predict_fcn(clf, preprocessor=preprocessor)


def get_tabular_explainer(predict_fn, dataset, split, config):

    feature_names = dataset.feature_names
    category_map = dataset.category_map
    X_train = split['X_train']

    explainer = AnchorTabular(predict_fn, feature_names, categorical_names=category_map, seed=int(config['seed']))
    explainer.fit(X_train, disc_perc=ast.literal_eval(config['disc_perc']))

    return explainer


def display_prediction(predict_fn, instance_id, splits, dataset):
    class_names = dataset.classnames
    X_test = splits['X_test']
    print('Prediction: ', class_names[predict_fn(X_test[instance_id].reshape(1, -1))[0]])


def get_explanation(explainer, expln_config, splits, exp_config):
    instance_id = int(exp_config['instance_idx'])

    return explainer.explain(splits['X_test'][instance_id],
                             threshold=float(expln_config['threshold']),
                             verbose=ast.literal_eval(expln_config['verbose']),
                             parallel=ast.literal_eval(expln_config['parallel']),
                             )


def display_explanation(explanation):
    print('Anchor: %s' % (' AND '.join(explanation['names'])))
    print('Precision: %.2f' % explanation['precision'])
    print('Coverage: %.2f' % explanation['coverage'])


class ExplainerExperiment(object):

    def __init__(self, *, dataset, expln_config, clf_config, exp_config):

        self.dataset = dataset
        self.explainer_config = expln_config
        self.experiment_config = exp_config
        self.clf_config = clf_config

        self._this_module = sys.modules[__name__]
        # TODO: Implement _create_data_store
        self._data_store = {'feat_ids': [],
                            'feat_names': [],
                            'precision': [],
                            'coverage': [],
                            't_elapsed': [],
                            'clf_config': {},
                            'exp_config': {},
                            'expln_config': {},
                            }
        self.explainer = None
        self.splits = None

    def __enter__(self):

        dataset = load_dataset(dataset=self.dataset)
        splits = split_data(dataset)
        self.splits = splits

        preprocess_fcn = 'preprocess_{}'.format(self.dataset)
        preprocessor = getattr(self._this_module, preprocess_fcn)(dataset, splits)
        clf_fcn = 'fit_{}'.format(self.clf_config['name'])
        predict_fn = getattr(self._this_module, clf_fcn)(splits, self.clf_config, preprocessor)
        explainer_fcn = 'get_{}_explainer'.format(self.explainer_config['type'])
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

    def _save_exp_metadata(self):
        self._data_store['clf_config'] = self.clf_config
        self._data_store['expln_config'] = self.explainer_config
        self._data_store['exp_config'] = self.experiment_config

    def _create_data_store(self):
        pass

    def update(self, explanation, t_elapsed):
        self._data_store['feat_ids'].append(explanation['raw']['feature'])
        self._data_store['feat_names'].append(explanation['raw']['names'])
        self._data_store['precision'].append(explanation['precision'])
        self._data_store['coverage'].append(explanation['coverage'])
        self._data_store['t_elapsed'].append(t_elapsed)


def run_experiment(args):

    n_runs = int(args['exp_config']['n_runs'])

    with ExplainerExperiment(**args) as exp:
        for _ in range(n_runs):
            with Timer() as time:
                explanation = get_explanation(exp.explainer,
                                              exp.explainer_config,
                                              exp.splits,
                                              exp.experiment_config,
                                              )
            exp.update(explanation, time.t_elapsed)
            if ast.literal_eval(args['exp_config']['verbose']):
                display_explanation(explanation)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anchor Explanations Experiments')
    parser.add_argument('dataset',
                        type=str,
                        default='adult',
                        help='Dataset from which to select instance to explain.'
                             'Allowed vals: adult/imagenet/movie_sentiment',
                        )
    # TODO: Could have a config file containing these jsons (or something that generates them) and just
    #  specify the file instead - here we'd just load the config
    parser.add_argument('-clf-config',
                        type=json.loads,
                        help=r'JSON with classifier config. Example usage:'
                             r'\'{"name": "rf", "seed": "0", "n_estimators": "5"}\'.'
                             r'Must contain seed value. Only random forest currently supported'
                             r' (name attribute must be rf).',
                        )
    parser.add_argument('-expln-config',
                        type=json.loads,
                        help=r'JSON with explainer config. Example usage:'
                             r'\'{"type":"tabular", "threshold": "0.95", "parallel":"False", "verbose":"False",'
                             r'"disc_perc":"(25, 50, 75)","seed":0"}\'.'
                             r'"type" should be set to "tabular", "image" or "text"',
                        )
    parser.add_argument('-exp-config',
                        type=json.loads,
                        help=r'JSON with experiment config. Example usage:'
                             r'\'{"n_runs": "100", "instance_idx": "6", "ckpt_dir":"/home/alex/my_exp",'
                             r'"ckpt": "awesome.pkl","verbose":"False"}\'.'
                             r'This specifies where to save the data and under what name, which test instance '
                             r'should be explained',
                        )
    configuration = parser.parse_args()

    if configuration.expln_config['type'] not in SUPPORTED_EXPLAINERS:
        raise NotImplementedError("Experiments are supported only for tabular data!")
    if configuration.dataset not in SUPPORTED_DATASETS:
        raise ValueError("Only datasets adult/imagenet/movie_sentiment are supported")
    if configuration.clf_config['name'] not in SUPPORTED_CLASSIFIERS:
        raise NotImplementedError("Only random forest classifiers are supported!")

    run_experiment(vars(configuration))
