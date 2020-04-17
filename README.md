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

# Author
[Sharada Mohanty](https://twitter.com/MeMohanty/)