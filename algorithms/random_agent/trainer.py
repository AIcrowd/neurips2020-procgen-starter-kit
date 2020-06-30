#!/usr/bin/env python

from ray.rllib.agents.trainer_template import build_trainer
from .policy import RandomPolicy

DEFAULT_CONFIG = {}

RandomAgentTrainer = build_trainer(
    name="RandomAgentTrainer",
    default_policy=RandomPolicy,
    default_config=DEFAULT_CONFIG,
)