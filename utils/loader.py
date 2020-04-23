#!/usr/bin/env python
import os
import sys
import glob

import types
import importlib.machinery

import humps

import gym

from ray import tune

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

def _uncache(exclude=[]):
    """Remove package modules from cache except excluded ones.
    On next import they will be reloaded.

    Original Source : https://gist.github.com/schipiga/482de016fa749bc08c7b36cf5323fd1b#file-uncache-py

    Args:
        exclude (iter<str>): Sequence of module paths.
    """
    pkgs = []
    for mod in exclude:
        pkg = mod.split('.', 1)[0]
        pkgs.append(pkg)

    to_uncache = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_uncache.append(mod)
            continue

            for pkg in pkgs:
                if mod.startswith(pkg + '.'):
                    to_uncache.append(mod)
                    break

        for mod in to_uncache:
            del sys.modules[mod]

def load_algorithms(CUSTOM_ALGORITHMS):
    """
    This function loads the custom algorithms implemented in this 
    repository, and then ultimately monkey-patches rllib to add the 
    said custom algorithms to the CONTRIBUTED_ALGORITHMS embedded 
    in rllib
    """
    import ray.rllib
    ray.rllib.contrib.registry.CONTRIBUTED_ALGORITHMS.update(
        CUSTOM_ALGORITHMS
    )
    _uncache()

