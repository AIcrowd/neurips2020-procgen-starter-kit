import gym
from gym.spaces import Discrete, Box
import numpy as np

from ray.tune import registry

from procgen import ProcgenEnv
from procgen.scalarize import Scalarize
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
            "debug_mode" : False,  # A useful flag that's passed through to procgen envs. Use however you want during debugging.
            "center_agent" : False,  # Determines whether observations are centered on the agent or display the full level. Override at your own risk.
            "use_sequential_levels" : False,  # When you reach the end of a level, the episode is ended and a new level is selected. If use_sequential_levels is set to True, reaching the end of a level does not end the episode, and the seed for the new level is derived from the current level seed. If you combine this with start_level=<some seed> and num_levels=1, you can have a single linear series of levels similar to a gym-retro or ALE game.
            "distribution_mode" : "easy"  # What variant of the levels to use, the options are "easy", "hard", "extreme", "memory", "exploration". All games support "easy" and "hard", while other options are game-specific. The default is "hard". Switching to "easy" will reduce the number of timesteps required to solve each game and is useful for testing or when working with limited compute resources. NOTE : During the evaluation phase (rollout), this will always be overriden to "easy"
        }
        self.config = self._default_config
        self.config.update(config)

        self.env_name = self.config["env_name"]
        
        assert self.env_name in VALID_ENV_NAMES
        # TODO: Add custom env_names via an env variable to this list

        self.env = None
        self.instantiate_new_env()

    def instantiate_new_env(self):
        if self.env:
            del self.env

        self.env = ProcgenEnv(
            num_envs=1,
            env_name=self.env_name,
            options=self.config
        )
        """
        More on the need to scalarize here : 
        https://github.com/openai/procgen/blob/master/procgen/scalarize.py#L7
        """
        self.env = Scalarize(self.env)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


    def reset(self):
        """
        VecEnvs cannot be reset after done is True,
        hence we need to instantiate a new environment
        on every reset.
        """        
        self.instantiate_new_env()
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

# Register Env in Ray
registry.register_env(
    "procgen_env_wrapper",
    lambda config: ProcgenEnvWrapper(config)
)