import argparse
import itertools
import os
import subprocess

from copy import deepcopy

import ruamel.yaml as yaml

from nested_lookup import nested_update

WDIR = os.getcwd()

# TODO: TEST EVERYTHING ON SIMPLE EXAMPLE AND MAKE SURE IT WORKS
# TODO: TIDY UP DEFAULT CONFIG AFTER TESTING AND COMMIT.


def get_experiment_params(config):

    values = []

    for hyperparameter in config['vars']:
        values.append(config['vars'][hyperparameter])

    return itertools.product(*values)


def get_config_path(hyperparameters, values, basepath='configs/tmp/'):

    if not os.path.exists(basepath):
        os.makedirs(basepath)

    config_name = ''

    for name, value in zip(hyperparameters, values):
        config_name += name + "_"
        config_name += str(value) + "_"

    config_name = config_name[:-1] + '.yaml'

    return os.path.join(WDIR, basepath, config_name)


def run_experiment(setting, hyperparameters, default_config, cleanup=False):

    print("")
    print("Running experiment with setting: {}".format(setting))
    print("The settings are for hyperparameters: {}".format(list(hyperparameters)))

    base_config = deepcopy(default_config)
    yml = yaml.YAML(typ='unsafe')

    # Update the default configuration with new hyperparameter settings and save
    # the updated .yaml file
    for name, value in zip(hyperparameters, setting):
        base_config = nested_update(base_config, name, value)

    config_path = get_config_path(hyperparameters, setting)

    with open(config_path, 'w') as f:
        yml.dump(base_config, f)

    # Run the experiment with the configuration created
    try:
        subprocess.run(["python", "{}/experiment.py".format(WDIR), "--config", "{}".format(config_path)],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=True,
                       )
        print("Experiment complete!")
    except subprocess.CalledProcessError as e:
        print("The command {} threw an error".format(e.cmd), sep=' ')
        print("The stdout is: {}".format(e.stdout), sep=' ')
        print("The stderr is: {}".format(e.stderr), sep=' ', end=' ')
        print("")

    # Remove temporary config file
    if cleanup:
        os.remove(config_path)


def main(default_config, experiments_dict, cleanup=False):

    hyperparams = experiments_dict['vars'].keys()

    print(hyperparams)
    print(default_config)
    exp_settings = get_experiment_params(paramters_dict)

    for setting in exp_settings:
        run_experiment(setting, hyperparams, default_config, cleanup=cleanup)
    print("All experiments complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple experiments with a single script')
    parser.add_argument("--config",
                        nargs="?",
                        type=str,
                        default="configs/config.yaml",
                        help="Default experiment configuration file for the experiment. "
                             "This configuration is to be modified to launch experiments",
                        )
    parser.add_argument("--parameters",
                        nargs="?",
                        type=str,
                        default="exp_settings.yaml",
                        help="Configuration file where variables that are to be changed are specified"
                        )
    parser.add_argument("--cleanup",
                        type=bool,
                        default=False,
                        help="If True, deletes the temporary configurations in configs/tmp that are "
                             "created during the experiments run."
                        )
    args = parser.parse_args()

    with open(args.config) as fp:
        default_config = yaml.load(fp)

    with open(args.parameters) as fp:
        paramters_dict = yaml.load(fp, Loader=yaml.SafeLoader)

    main(default_config, paramters_dict, cleanup=args.cleanup)
