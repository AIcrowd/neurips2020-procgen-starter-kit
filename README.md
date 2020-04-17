# rl-experiments

# Installation
```
pip install ray[rllib]
pip install tensorflow # or tensorflow-gpu
```

# Usage
```
Training example:
    python ./train.py --run DQN --env CartPole-v0

Training with Config:
    python ./train.py -f experiments/simple-corridor-0.yaml


Note that -f overrides all other trial-specific command-line options.
```

## How to add a custom environment ?

To add a custom environment, create a file inside `envs/` To add a custom model, create a file inside `models/` directory.
,
and name is `my_new_amazing_custom_env.py`.   
The file, should implement a class `MyNewAmazingCustomEnv`, which should be inherited from `gym.Env`. 
Note that the class name here should be a PascalCase version of the filename (without the extension).

Then, you can readily use `my_new_amazing_custom_env` as the env_name in the YAML configs in `experiments/` directory.
Please refer [envs/simple_corridor.py](envs/simple_corridor.py) for an example.

## How to add a custom Model ?
To add a custom model, create a file inside `models/` directory and name is `my_amazing_model.py`. The file, should implement
a class `MyAmazingModel`, which should be inherited from `ray.rllib.models.tf.tf_modelv2.TFModelV2`.
Note that the class name here should be a PascalCase version of the filename (without the extension).

Then, you can readily use `my_amazing_model` as the Model name in the YAML configs in the `experiments/` directory.

## What configs parameters do I have available ?
This is a very thin wrapper over RLLib. Hence, you have all the RLLib config options available in the experiment configuration (YAML) file to play with.

# Author
[Sharada Mohanty](https://twitter.com/MeMohanty/)