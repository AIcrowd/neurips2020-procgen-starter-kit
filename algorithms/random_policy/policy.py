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
        # are used for checkpointing the states and restoring the states
        # from a checkpoint.
        self.w = []

    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
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
        """Fused compute gradients and apply gradients call.

        Either this or the combination of compute/apply grads must be
        implemented by subclasses.

        Returns:
            grad_info: dictionary of extra metadata from compute_gradients().
        Examples:
            >>> batch = ev.sample()
            >>> ev.learn_on_batch(samples)

        Reference: https://github.com/ray-project/ray/blob/master/rllib/policy/policy.py#L279-L316
        """
        # implement your learning code here
        return {}

    def get_weights(self):
        """Returns model weights.

        Returns:
            weights (obj): Serializable copy or view of model weights
        """
        return {"w": self.w}

    def set_weights(self, weights):
        """Returns the current exploration information of this policy.

        This information depends on the policy's Exploration object.
        
        Returns:
            any: Serializable information on the `self.exploration` object.
        """
        self.w = weights["w"]
