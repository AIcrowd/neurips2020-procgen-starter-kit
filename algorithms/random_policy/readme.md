# Writing a custom policy

For more information on writing custom policies, please refer https://docs.ray.io/en/master/rllib-concepts.html

This directory contains the example code for implementing a custom random policy. Here, the agent never learns and outputs random actions for every observation.

## Directory structure

```
.
└── algorithms            # Directory containing code for custom algorithms
    ├── __init__.py
    ├── random_policy     # Python module for random policy
    │   ├── __init__.py
    │   ├── policy.py     # Code for random policy
    │   └── trainer.py    # Training wrapper for the random policy
    └── registry.py
```

## How to start?

- Go through `policy.py` that has most of what you are looking for. `trainer.py` is just a training wrapper around the policy.
- Once the policy is implemented, you need to register the policy with `rllib`. You can do this by adding your policy trainer class to `registry.py`.