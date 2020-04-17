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

Grid search example:
    python ./train.py -f tuned_examples/cartpole-grid-search-example.yaml


Note that -f overrides all other trial-specific command-line options.
```

# Author
[Sharada Mohanty](https://twitter.com/MeMohanty/)