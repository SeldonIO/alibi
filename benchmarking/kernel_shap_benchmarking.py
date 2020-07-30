import argparse
import logging
import os
import pickle

import numpy as np
from functools import partial

from alibi.datasets import fetch_adult
from alibi.explainers.shap_wrappers import KernelShap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from timeit import default_timer as timer
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)


def load_adult_dataset():
    """
    Load the Adult dataset.
    """

    logging.info("Preprocessing data...")
    return fetch_adult()


def preprocess_adult_dataset(dataset, seed=0, n_train_examples=30000) -> Dict[str, Any]:
    """
    Splits dataset into train and test subsets and preprocesses it.
    """

    logging.info("Splitting data...")
    # split data
    np.random.seed(seed)
    data = dataset.data
    target = dataset.target
    data_perm = np.random.permutation(np.c_[data, target])
    data = data_perm[:, :-1]
    target = data_perm[:, -1]

    X_train, y_train = data[:n_train_examples, :], target[:n_train_examples]
    X_test, y_test = data[n_train_examples + 1:, :], target[n_train_examples + 1:]

    category_map = dataset.category_map
    feature_names = dataset.feature_names

    ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
    ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])

    categorical_features = list(category_map.keys())
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                              ('onehot', OneHotEncoder(drop='first', handle_unknown='error'))])

    preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                                   ('cat', categorical_transformer, categorical_features)])
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return {
        'X': {
            'raw': {'train': X_train, 'test': X_test},
            'processed': {'train': X_train_proc, 'test': X_test_proc}},
        'y': {'train': y_train, 'test': y_test},
        'preprocessor': preprocessor,
        'orig_feature_names': feature_names,
    }


def fit_adult_logistic_regression(data_dict):
    """
    Fit a logistic regression model to the processed Adult dataset.
    """

    logging.info("Fitting model ...")
    X_train_proc = data_dict['X']['processed']['train']
    X_test_proc = data_dict['X']['processed']['test']
    y_train = data_dict['y']['train']
    y_test = data_dict['y']['test']

    classifier = LogisticRegression(multi_class='multinomial',
                                    random_state=0,
                                    max_iter=500,
                                    verbose=0,
                                    )
    classifier.fit(X_train_proc, y_train)

    logging.info(f"Test accuracy: {accuracy_score(y_test, classifier.predict(X_test_proc))}")

    return classifier


def split_set(X, y, fraction, random_state=0):
    """
    Given a set X, associated labels y, splits a fraction y from X.
    """
    _, X_split, _, y_split = train_test_split(X,
                                              y,
                                              test_size=fraction,
                                              random_state=random_state,
                                              )
    logging.info(f"Number of records: {X_split.shape[0]}")
    logging.info(f"Number of class {0}: {len(y_split) - y_split.sum()}")
    logging.info(f"Number of class {1}: {y_split.sum()}")

    return X_split, y_split


def group_adult_dataset(preprocessed_dataset: Dict):
    """
    This function:
        - Finds the numerical and categorical variables in the processed data, along with the encoding length for cat. \
        variables
        - Outputs a list of the same length as the number of variable, where each element is a list specifying the \ 
        indices occupied by each variable in the processed (aka encoded) dataset
    """  # noqa

    feature_names = preprocessed_dataset['orig_feature_names']
    preprocessor = preprocessed_dataset['preprocessor']
    numerical_feats_idx = preprocessor.transformers_[0][2]
    categorical_feats_idx = preprocessor.transformers_[1][2]
    ohe = preprocessor.transformers_[1][1].named_steps['onehot']
    # compute encoded dimension; -1 as ohe is setup with drop='first'
    feat_enc_dim = [len(cat_enc) - 1 for cat_enc in ohe.categories_]
    num_feats_names = [feature_names[i] for i in numerical_feats_idx]
    cat_feats_names = [feature_names[i] for i in categorical_feats_idx]
    print(num_feats_names)
    print(cat_feats_names)

    group_names = num_feats_names + cat_feats_names
    groups = []
    cat_var_idx = 0

    for name in group_names:
        if name in num_feats_names:
            groups.append(list(range(len(groups), len(groups) + 1)))
        else:
            start_idx = groups[-1][-1] + 1 if groups else 0
            groups.append(list(range(start_idx, start_idx + feat_enc_dim[cat_var_idx])))
            cat_var_idx += 1

    return group_names, groups


def fit_kernel_shap_explainer(clf, background_data, groups, group_names, distributed_opts):
    """Returns an a fitted explainer for the classifier `clf`"""

    pred_fcn = clf.predict_proba
    explainer = KernelShap(pred_fcn, link='logit', feature_names=group_names, distributed_opts=distributed_opts)
    explainer.fit(background_data, group_names=group_names, groups=groups)

    return explainer


def sparse2ndarray(mat, examples=None):
    """
    Converts a `scipy.sparse.csr.csr_matrix` to `np.ndarray`. If specified, examples is slice object specifying which
    selects a number of rows from mat and converts only the respective slice.
    """

    if examples:
        return mat[examples, :].toarray()

    return mat.toarray()


def get_filename(distributed_opts: dict):
    """Creates a filename for an experiment given `distributed_opts`."""

    if distributed_opts['n_cpus']:
        return f"results/ray_ncpu_{distributed_opts['n_cpus']}_bsize_{distributed_opts['batch_size']}.pkl"
    return "results/sequential.pkl"


def main():

    # load and preprocess data
    adult_dataset = load_adult_dataset()
    preprocessed_data = preprocess_adult_dataset(adult_dataset)
    preprocessor = preprocessed_data['preprocessor']
    lr = fit_adult_logistic_regression(preprocessed_data)
    # treat encoded categorical variables as a single variable during sampling
    group_names, groups = group_adult_dataset(preprocessed_data)
    # define background data as first 100 training examples
    background_data_idx = slice(0, args.n_background_samples)
    background_data = preprocessed_data['X']['processed']['train'][background_data_idx, :]
    background_data = sparse2ndarray(background_data)
    # explain a fraction of the test set
    fraction_explained = args.fraction_explained
    logging.info(f"{background_data.shape[0]} background samples")
    logging.info(f"fraction explained: {fraction_explained}")
    if np.isclose(fraction_explained, 1.0):
        X_explain_proc = sparse2ndarray(preprocessed_data['X']['processed']['test'])
    else:
        X_explain, y_explain = split_set(preprocessed_data['X']['raw']['test'],
                                         preprocessed_data['y']['test'],
                                         fraction_explained,
                                         )
        X_explain_proc = sparse2ndarray(preprocessor.transform(X_explain))
    experiment = partial(
        run_experiment,
        X_explain=X_explain_proc,
        predictor=lr,
        background_data=background_data,
        groups=groups,
        group_names=group_names,
    )
    if args.sequential:
        logging.info("Running sequential mode")
        experiment(batch_size=None, n_cpu=None, n_runs=args.nruns)
    else:
        logging.info("Running in parallel mode")
        batch_sizes = [10, 20, 40, 80]
        for batch_size in batch_sizes:
            for n_cpu in range(2, args.ncpu_max):
                experiment(batch_size=batch_size, n_cpu=n_cpu, n_runs=args.nruns)


def run_experiment(batch_size, n_cpu, n_runs, X_explain, predictor, background_data, groups, group_names):
    """
    Runs an experiment for a certain batch size, number of CPUs combination. Each cobination is run `n_runs` times.
    """
    logging.info(f"Batch_size: {batch_size}")
    logging.info(f"n_cpu: {n_cpu}")
    logging.info(f"n_runs: {n_runs}")
    distributed_opts = {'batch_size': batch_size, 'n_cpus': n_cpu}
    result = {'t_elapsed': [], 'explanations': []}

    for run in range(n_runs):
        logging.info(f"run: {run}")
        # fit explainer
        explainer = fit_kernel_shap_explainer(predictor, background_data, groups, group_names, distributed_opts)
        t_start = timer()
        explanation = explainer.explain(X_explain)
        t_elapsed = timer() - t_start
        logging.info(f"Time elapsed: {t_elapsed}")
        result['t_elapsed'].append(t_elapsed)
        result['explanations'].append(explanation)
    with open(get_filename(distributed_opts), 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    if not os.path.exists('results'):
        os.mkdir('results')
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_background_samples', type=int, default=100, help="Background set size.")
    parser.add_argument('-fraction_explained', type=float, default=1.0, help="Fraction of test set explained.")
    parser.add_argument('-ncpu_max', type=int, default=3, help="Number of cores to run the computation on.")
    parser.add_argument('-nruns', type=int, default=5, help="Number of runs per batch/ncpu setting.")
    parser.add_argument('-sequential', type=bool, default=False, help="Run sequential mode")
    args = parser.parse_args()
    main()
