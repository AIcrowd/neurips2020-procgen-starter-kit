import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple, Dict
import numpy as np

from ray.tune import registry

class VeronikaAndriusEnv(gym.Env):
    """
    Procgen Wrapper file
    """
    def __init__(self, config={
            "MAX_INT":4,
            "STATE_SIZE":3
        }):
        self.MAX_INT = config["MAX_INT"]
        self.STATE_SIZE = config["STATE_SIZE"]

        self.config = config

        self.state = self.reset()

        self.action_space = Discrete(self.STATE_SIZE)

        _observation_array = []
        
        # os_dict = {}
        # for idx in range(self.STATE_SIZE):
        #     os_dict[idx] = Discrete(self.MAX_INT)
        # self.observation_space = Dict(os_dict)
        # print(self.observation_space.sample())
        ob_tuple = []
        for idx in range(self.STATE_SIZE):
            ob_tuple.append(
                Discrete(self.MAX_INT)
            )
        self.observation_space = Tuple(ob_tuple)
    
    def _get_observation(self):
        return self.state

    def reset(self):
        self.state = np.array([
            np.random.randint(0, self.MAX_INT) for _ in range(self.STATE_SIZE)
        ])
        return self._get_observation()

    def step(self, action):
        """
            Action in this case has to be the index in the array
        """
        action = int(action)
        assert action < self.STATE_SIZE, "Invalid Action Provided. Action has to be less than action size"

        for _idx in range(self.STATE_SIZE):
            if _idx == action:
                continue
            else:
                self.state[_idx] += 1

        # observation, reward, done, info
        observation = self._get_observation()

        done = False
        reward = 0
        if np.any(self.state >= self.MAX_INT):
            done = True
        elif len(set(self.state)) == 1:
            done = True
            reward = 1
        info = {}

        return observation, reward, done, info

registry.register_env(
    "veronika_andrius_env",
    lambda config: VeronikaAndriusEnv(config)
)

if __name__ =="__main__":
    env = VeronikaAndriusEnv()

    observation = env.reset()
    done = False
    while done == False:
        print("Observation : ", observation)
        action = int(input("Enter action index : "))
        observation, reward, done, info = env.step(
            action
        )
        print("Reward : ", reward)
    