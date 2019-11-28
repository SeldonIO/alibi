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

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import spacy

import alibi.datasets as datasets

from alibi.explainers import AnchorTabular, AnchorText
from alibi.utils.download import spacy_model
from alibi.utils.wrappers import Predictor

SUPPORTED_EXPLAINERS = ['tabular', 'text']
SUPPORTED_DATASETS = ['adult', 'imagenet', 'movie_sentiment']
SUPPORTED_CLASSIFIERS = ['rf', 'lr']

# TODO: Typing and documentation
# TODO: Raise NotImplemented error if show_covered=True for Anchor Tabular
# TODO: opts not used currently in preprocess_adult, pipelines hardcoded
# TODO: in the future one should be able to pass their own classifier that's already fitted as opp to using fit_*

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

    seed = opts['seed']
    method = opts['method']
    print("Splitting using {} method ...".format(method))
    if method == 'shuffle':
        n_train_records = opts['n_train_records']
        return _shuffle(dataset, n_train_records, seed)
    else:
        split_fractions = {
            'test': opts['test_size'],
            'val': opts['val_size'],
        }
        return _train_test_val(dataset, seed, split_fractions)



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

def display_performance(splits, predictor):
    print('Train accuracy: ', accuracy_score(splits['Y_train'], predictor(splits['X_train'])))
    print('Test accuracy: ', accuracy_score(splits['Y_test'], predictor(splits['X_test'])))

def predict_fcn(clf, preprocessor=None):
    if preprocessor:
        return lambda x: clf.predict(preprocessor.transform(x))
    return lambda x: clf.predict(x)


def fit_rf(splits, config, preprocessor=None):

    print("Fitting classifier ...")

    np.random.seed(config['seed'])
    clf = RandomForestClassifier(n_estimators=config['n_estimators'])
    if preprocessor:
        clf.fit(preprocessor.transform(splits['X_train']), splits['Y_train'])
    else:
        clf.fit(splits['X_train'], splits['Y_train'])

    display_performance(splits, Predictor(clf, preprocessor=preprocessor))

    return Predictor(clf, preprocessor=preprocessor)


def fit_lr(splits, config, preprocessor=None):

    print("Fitting classifier ...")
    np.random.seed(config['seed'])
    clf = LogisticRegression(solver=config['solver'])
    if preprocessor:
        clf.fit(preprocessor.transform(splits['X_train']), splits['Y_train'])
    else:
        clf.fit(splits['X_train'], splits['Y_train'])

    display_performance(splits, Predictor(clf, preprocessor=preprocessor))

    return Predictor(clf, preprocessor=preprocessor)


def get_tabular_explainer(predictor, config, dataset=None, split=None):

    if not dataset or not split:
        raise ValueError("Anchor Tabular requires both a dataset object and"
                         " a dictionary with datasets split for fitting but"
                         " at least one was not passed to get_tabular_function!")

    feature_names = dataset.feature_names
    category_map = dataset.category_map
    X_train = split['X_train']

    explainer = AnchorTabular(predictor, feature_names, categorical_names=category_map, seed=config['seed'])
    explainer.fit(X_train, disc_perc=config['disc_perc'])

    return explainer


def get_text_explainer(predictor, config, dataset=None, split=None):

    model_name = config['pert_model']
    spacy_model(model=model_name)
    perturbation_model = spacy.load(model_name)

    return AnchorText(perturbation_model, predictor, seed=config['seed'])


def get_explanation(explainer, instance, expln_config):
    return explainer.explain(instance,
                             **expln_config,
                             )


def _display_prediction(predict_fn, instance_id, splits, dataset):
    class_names = dataset.classnames
    X_test = splits['X_test']
    print('Prediction: ', class_names[predict_fn(X_test[instance_id].reshape(1, -1))[0]])


def _tabular_prediction(predictor, instance, dataset):
    class_names = dataset.target_names
    pred =  class_names[predictor(instance.reshape(1, -1))[0]]
    alternative =  class_names[1 - predictor(instance.reshape(1, -1))[0]]
    print('Prediction: ', pred)
    print('Alternative', alternative)
    return pred, alternative

def _text_prediction(predictor, instance, dataset):
    class_names = dataset.target_names
    pred =  class_names[predictor([instance])[0]]
    alternative  = class_names[1 - predictor([instance])[0]]
    print('Prediction: ', pred)
    print('Alternative', alternative)
    return pred, alternative


def display_explanation(pred, alternative, explanation, show_covered=False):
    print('Anchor: %s' % (' AND '.join(explanation['names'])))
    print('Precision: %.2f' % explanation['precision'])
    print('Coverage: %.2f' % explanation['coverage'])

    if show_covered:
        print('\nExamples where anchor applies and model predicts {}'.format(pred))
        print('\n'.join([x[0] for x in explanation['raw']['examples'][-1]['covered_true']]))
        print('\nExamples where anchor applies and model predicts {}'.format(alternative))
        print('\n'.join([x[0] for x in explanation['raw']['examples'][-1]['covered_false']]))


class ExplainerExperiment(object):

    def __init__(self, *, dataset, explainer, classifier, experiment):


        self.dataset_name = dataset
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
        self.instance = None
        self.splits = None


    def __enter__(self):

        # load and split dataset
        dataset = load_dataset(dataset=self.dataset_name)
        splits = split_data(dataset, self.experiment_config['data']['split_opts'])
        self.splits = splits
        self.dataset = dataset

        # optionally preprocess the dataset
        if self.experiment_config['data']['preprocess']:
            preprocess_fcn = 'preprocess_{}'.format(self.dataset_name)
            preproc = getattr(self._this_module, preprocess_fcn)(dataset,
                                                                 splits,
                                                                 opts=self.experiment_config['data']['preprocess_opts'])
        else:
            preproc = None

        # fit classifier
        clf_fcn = 'fit_{}'.format(self.clf_config['name'])
        self.predictor = getattr(self._this_module, clf_fcn)(splits, self.clf_config, preprocessor=preproc)
        explainer_fcn = 'get_{}_explainer'.format(self.explainer_config['type'])

        # create and fit explainer instance
        explainer = getattr(self._this_module, explainer_fcn)(self.predictor,
                                                              self.explainer_config,
                                                              dataset,
                                                              splits)
        self.explainer = explainer

        # retrieve instance to be explained
        instance_idx = self.experiment_config['instance_idx']
        instance_split = self.experiment_config['instance_split']
        if instance_split:
            self.instance = splits['X_{}'.format(instance_split)][instance_idx]
        else:
            self.instance = dataset.data[instance_idx]

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


def display(config, explanation, exp):

    if config['experiment']['verbose']:
        pred_fcn = '_{}_prediction'.format(config['explainer']['type'])
        # display prediction on instance to be explained
        pred, alternative = getattr(exp._this_module, pred_fcn)(exp.predictor, exp.instance, exp.dataset)
        display_explanation(pred,
                            alternative,
                            explanation,
                            show_covered=config['experiment']['show_covered'])
    return

def check(config):

    if config['explainer']['type'] not in SUPPORTED_EXPLAINERS:
        raise NotImplementedError("Experiments are supported only for tabular data!")
    if config['dataset'] not in SUPPORTED_DATASETS:
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
        configuration = yaml.load(fp, Loader=yaml.FullLoader)

    configuration['experiment']['commit_hash'] = args.hash

    check(configuration)

    if configuration['experiment']['profile']:
        profile(configuration)
    else:
        run_experiment(configuration)
