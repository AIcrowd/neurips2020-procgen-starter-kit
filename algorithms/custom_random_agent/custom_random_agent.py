import numpy as np

from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.utils.annotations import override

"""
Note : This implementation has been adapted from : 
    https://github.com/ray-project/ray/blob/master/rllib/contrib/random_agent/random_agent.py
"""

# yapf: disable
# __sphinx_doc_begin__
class CustomRandomAgent(Trainer):
    """Policy that takes random actions and never learns."""

    _name = "CustomRandomAgent"
    _default_config = with_common_config({
        "rollouts_per_iteration": 10,
    })

    @override(Trainer)
    def _init(self, config, env_creator):
        self.env = env_creator(config["env_config"])

    @override(Trainer)
    def _train(self):
        rewards = []
        steps = 0
        for _ in range(self.config["rollouts_per_iteration"]):
            obs = self.env.reset()
            done = False
            reward = 0.0
            while not done:
                action = self.env.action_space.sample()
                obs, r, done, info = self.env.step(action)
                reward += r
                steps += 1
            rewards.append(reward)
        return {
            "episode_reward_mean": np.mean(rewards),
            "timesteps_this_iter": steps,
        }
# __sphinx_doc_end__
# don't enable yapf after, it's buggy here


if __name__ == "__main__":
    trainer = CustomRandomAgent(
        env="CartPole-v0", config={"rollouts_per_iteration": 10})
    result = trainer.train()
    assert result["episode_reward_mean"] > 10, result
    print("Test: OK")