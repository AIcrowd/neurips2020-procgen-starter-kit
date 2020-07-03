#!/usr/bin/env python

from ray.rllib.models.preprocessors import Preprocessor


class MyPreprocessorClass(Preprocessor):
    """Custom preprocessing for observations

    Adopted from https://docs.ray.io/en/master/rllib-models.html#custom-preprocessors
    """

    def _init_shape(self, obs_space, options):
        return obs_space.shape  # New shape after preprocessing

    def transform(self, observation):
        # Do your custom stuff
        return observation
