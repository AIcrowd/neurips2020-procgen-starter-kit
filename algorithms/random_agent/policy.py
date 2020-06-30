#!/usr/bin/env python

from ray.rllib.policy import Policy
import numpy as np


class RandomPolicy(Policy):
    """Example of a random policy

    If you are using tensorflow/pytorch to build custom policies,
    you might find `build_tf_policy` and `build_torch_policy` to
    be useful.

    Adopted from examples from https://docs.ray.io/en/master/rllib-concepts.html
    """
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        
        # You can replace this with whatever variable you want to save
        # the state of the policy in. `get_weights` and `set_weights`
        # help to restore the state of the policy.
        self.w = []

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """Return the action for a batch

        Returns:
            action_batch: List of actions for the batch
            rnn_states: List of RNN states if any
            info: Additional info
        """
        action_batch = []
        rnn_states = []
        info = {}
        for _ in obs_batch:
            action_batch.append(self.action_space.sample())
        return action_batch, rnn_states, info

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]