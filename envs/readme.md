# Custom gym environments

This directory contains the custom gym environments that will be used by
`rllib`.

## Using gym wrappers

You can use custom gym wrappers for better control over the environment.
These wrappers should be used on `ProcgenEnvWrapper` class. You should
not use `gym.make` to create an environment but use the gym env provided
in the starter kit.

### Example

A simple example to use framestack will be

```python
from gym.wrappers import FrameStack
from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

# Register Env in Ray
registry.register_env(
    "stacked_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: FrameStack(ProcgenEnvWrapper(config), 4)
)
```

You can point to `stacked_procgen_env` instead of `procgen_env_wrapper` in your
experiment config file in order to use the env with the wrapper.

### Note
- If you do not use `ProcgenEnvWrapper` as your base env, the
rollouts will fail.
- Please do not edit `procgen_env_wrapper.py` file. All the changes
you make to this file will be dropped during the evaluation.
