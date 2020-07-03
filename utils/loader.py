#!/usr/bin/env python
import os
import glob

import types
import importlib.machinery

"""
Helper functions
"""


def _source_file(_file_path):
    """
    Dynamically "sources" a provided file
    """
    basename = os.path.basename(_file_path)
    filename = basename.replace(".py", "")
    # Load the module
    loader = importlib.machinery.SourceFileLoader(filename, _file_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)


"""
A loder utility, which takes an experiment directory
path, and loads necessary things into the ModelRegistry.

This imposes an opinionated directory structure on the
users, which looks something like :

- envs/
    - my_env_1.py
    - my_env_2.py
    ....
    - my_env_N.py
- models/
    - my_model_1.py
    - my_model_2.py
    .....
    - my_model_N.py
"""


def load_envs(local_dir="."):
    """
    This function takes a path to a local directory
    and looks for an `envs` folder, and imports
    all the available files in there.
    """
    for _file_path in glob.glob(os.path.join(
            local_dir, "envs", "*.py")):
        """
        Sources a file expected to implement a said
        gym env wrapper.

        The respective files are expected to do a
        `registry.register_env` call to ensure that
        the implemented envs are available in the
        ray registry.
        """
        _source_file(_file_path)


def load_models(local_dir="."):
    """
    This function takes a path to a local directory
    and looks for a `models` folder, and imports
    all the available files in there.
    """
    for _file_path in glob.glob(os.path.join(
        local_dir, "models", "*.py")):
        """
        Sources a file expected to implement a
        custom model.

        The respective files are expected to do a
        `registry.register_env` call to ensure that
        the implemented envs are available in the
        ray registry.
        """
        _source_file(_file_path)


def load_algorithms(CUSTOM_ALGORITHMS):
    """
    This function loads the custom algorithms implemented in this 
    repository, and registers them with the tune registry
    """
    from ray.tune import registry
    
    for _custom_algorithm_name in CUSTOM_ALGORITHMS:
        _class = CUSTOM_ALGORITHMS[_custom_algorithm_name]()
        registry.register_trainable(
            _custom_algorithm_name,
            _class)


def load_preprocessors(CUSTOM_PREPROCESSORS):
    """Function to register custom preprocessors
    """
    from ray.rllib.models import ModelCatalog

    for _precessor_name, _processor_class in CUSTOM_PREPROCESSORS.items():
        ModelCatalog.register_custom_preprocessor(_precessor_name, _processor_class)
