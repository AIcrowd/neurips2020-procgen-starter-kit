import time
import gym
import numpy as np

from ray.tune import registry
from procgen.env import ENV_NAMES as VALID_ENV_NAMES

class ProcgenEnvWrapper(gym.Env):
    """
    Procgen Wrapper file
    """
    def __init__(self, config):
        self._default_config = {
            "num_levels" : 0,  # The number of unique levels that can be generated. Set to 0 to use unlimited levels.
            "env_name" : "coinrun",  # Name of environment, or comma-separate list of environment names to instantiate as each env in the VecEnv
            "start_level" : 0,  # The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels
            "paint_vel_info" : False,  # Paint player velocity info in the top left corner. Only supported by certain games.
            "use_generated_assets" : False,  # Use randomly generated assets in place of human designed assets
            "center_agent" : True,  # Determines whether observations are centered on the agent or display the full level. Override at your own risk.
            "use_sequential_levels" : False,  # When you reach the end of a level, the episode is ended and a new level is selected. If use_sequential_levels is set to True, reaching the end of a level does not end the episode, and the seed for the new level is derived from the current level seed. If you combine this with start_level=<some seed> and num_levels=1, you can have a single linear series of levels similar to a gym-retro or ALE game.
            "distribution_mode" : "easy"  # What variant of the levels to use, the options are "easy", "hard", "extreme", "memory", "exploration". All games support "easy" and "hard", while other options are game-specific. The default is "hard". Switching to "easy" will reduce the number of timesteps required to solve each game and is useful for testing or when working with limited compute resources. NOTE : During the evaluation phase (rollout), this will always be overriden to "easy"
        }
        self.config = self._default_config
        self.config.update(config)

        self.env_name = self.config.pop("env_name")

        assert self.env_name in VALID_ENV_NAMES

        env = gym.make(f"procgen:procgen-{self.env_name}-v0", **self.config)
        self.env = env
        # Enable video recording features
        self.metadata = self.env.metadata

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._done = True

    def reset(self):
        assert self._done, "procgen envs cannot be early-restarted"
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._done = done
        return obs, rew, done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __repr__(self):
        return self.env.__repr()

    @property
    def spec(self):
        return self.env.spec

# Register Env in Ray
registry.register_env(
    "procgen_env_wrapper",
    lambda config: ProcgenEnvWrapper(config)
)