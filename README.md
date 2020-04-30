# NeurIPS 2020 - Procgen Starter Kit
![AIcrowd-Logo](https://raw.githubusercontent.com/AIcrowd/AIcrowd/master/app/assets/images/misc/aicrowd-horizontal.png)

This is the starter kit for the [NeurIPS 2020 - Procgen competition](LINK-TO-BE-INSERTED) hosted on [AIcrowd](https://www.aicrowd.com/)

16 simple-to-use procedurally-generated [gym](https://github.com/openai/gym) environments which provide a direct measure of how quickly a reinforcement learning agent learns generalizable skills.  The environments run at high speed (thousands of steps per second) on a single core.

![](https://raw.githubusercontent.com/openai/procgen/master/screenshots/procgen.gif)

These environments are associated with the paper [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://cdn.openai.com/procgen.pdf) [(citation)](#citation).  
Compared to [Gym Retro](https://github.com/openai/retro), these environments are:

* Faster: Gym Retro environments are already fast, but Procgen environments can run >4x faster.
* Non-deterministic: Gym Retro environments are always the same, so you can memorize a sequence of actions that will get the highest reward.  Procgen environments are randomized so this is not possible.
* Customizable: If you install from source, you can perform experiments where you change the environments, or build your own environments.  The environment-specific code for each environment is often less than 300 lines.  This is almost impossible with Gym Retro.

Supported platforms:
- Windows 10
- macOS 10.14 (Mojave), 10.15 (Catalina)
- Linux (manylinux2010)

Supported Pythons:
- 3.6 64-bit
- 3.7 64-bit

Supported CPUs:
- Must have at least AVX

# Installation
```
pip install ray[rllib]
pip install tensorflow # or tensorflow-gpu
pip install procgen
pip install humps
```

# Usage
```

git clone git@github.com:AIcrowd/neurips2020-procgen-starter-kit.git
cd neurips2020-procgen-starter-kit

# Training example:
python ./train.py --run PPO -f experiments/procgen-0.yaml

# Rollout example:
# the env name and configuration are automatically picked up from 
# the experiment config.

python ./rollout.py \
    /tmp/ray/checkpoint_dir/checkpoint-0 \
    --run PPO \
    --episodes 100

# NOTE : The path to the checkpoint will have the following path in case of default options :
# ~/ray_results/procgen-ppo/<experiment-name>-<uuid>/checkpoint_1/checkpoint-1
```

## How do I add a custom Model ?
To add a custom model, create a file inside `models/` directory and name is `models/my_vision_network.py`. 
Please refer [here](https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/models/my_vision_network.py ) for a working implementation of how to add a custom model.

## How do I add a custom Algorithm/Trainable/Agent ?
Please refer to the instructions [here](https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/experiments/procgen-0.yaml#L4)

## What configs parameters do I have available ?
This is a very thin wrapper over RLLib. Hence, you have all the RLLib config options available in the experiment configuration (YAML) file to play with.
Please refer to [this example](https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/experiments/procgen-0.yaml) for an exaplanation on the structure and available options in the experiment config file.

## How will my code be evaluated ?
TODO - Mohanty to elaborate later

# Author(s)
- [Sharada Mohanty](https://twitter.com/MeMohanty/)
- [Karl Cobbe](https://github.com/kcobbe)
