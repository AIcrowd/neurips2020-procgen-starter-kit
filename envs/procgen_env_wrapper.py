import gym
from gym.spaces import Discrete, Box
import numpy as np

from ray.tune import registry

from procgen import ProcgenEnv

class ProcgenEnvWrapper(gym.Env):
    """
    Procgen Wrapper file
    """
    def __init__(self, config):
        self.config = config
        self.env_name = self.config["env_name"]
        
        self.env = ProcgenEnv(
            num_envs=1, env_name=self.env_name
        )

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

# Register Env in Ray
registry.register_env(
    "ProcGenEnv",
    lambda config: ProcgenEnvWrapper(config)
)