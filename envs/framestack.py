from gym.wrappers import FrameStack
from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

# Register Env in Ray
registry.register_env(
    "stacked_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: FrameStack(ProcgenEnvWrapper(config), 4),
)
