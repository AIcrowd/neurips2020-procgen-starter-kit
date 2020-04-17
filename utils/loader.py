#!/usr/bin/env python
import os
import glob

import types
import importlib.machinery

import humps

import gym

from ray import tune
from ray.tune import registry

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
        Determine the filename, env_name and class_name

        # Convention :
            - filename : snake_case
            - classname : PascalCase

            the class implementation, should be an inheritance
            of gym.Env
        """
        basename = os.path.basename(_file_path)
        env_name = basename.replace(".py", "")
        class_name = humps.pascalize(env_name)

        # Load the module
        loader = importlib.machinery.SourceFileLoader(env_name, _file_path)
        mod = types.ModuleType(loader.name)
        loader.exec_module(mod)
        try:
            _class = getattr(mod, class_name)
        except KeyError:
            # TODO : Add a better error message
            raise Exception(
                "Looking for a class named {} in the file {}."
                "Did you name the class correctly ?".format(
                    env_name, class_name
                ))
        env = _class
        # Validate the class
        if not issubclass(_class, gym.Env):
            raise Exception(
                "We expected the class named {} to be "
                "a subclass of gym.Env. "
                "Please read more here : <insert-link>"
                .format(
                    class_name
                ))
        # Finally Register Env in Tune
        registry.register_env(env_name, lambda: env)
