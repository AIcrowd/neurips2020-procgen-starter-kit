import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple, Dict
import numpy as np

from ray.tune import registry

class VeronikaAndriusEnv(gym.Env):
    """
    Procgen Wrapper file
    """
    def __init__(self, config={}):
        self.MAX_INT = 10
        self.STATE_SIZE = 3

        self.state = self.reset()

        self.action_space = Discrete(self.STATE_SIZE)

        _observation_array = []
        
        os_dict = {}
        for idx in range(self.STATE_SIZE):
            os_dict[idx] = Discrete(self.MAX_INT)
        self.observation_space = Dict(os_dict)
        print(self.observation_space.sample())
    
    def _get_observation(self):
        _OBS = {}
        for _ in range(self.STATE_SIZE):
            _OBS[_] = self.state[_]
        """
        {
            0: 1 + np.random.randint(-2, 2)
            1: 4 + np.random.randint(-2, 2), 
            2: 9 + np.random.randint(-2, 2),
        }
        """
        return _OBS

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

        _mean = np.array(self.state.values()).mean()
        score = np.abs((self.state.values() - _mean).mean())

        _temp_state = self.state
        for _key in enumerate(_temp_state.keys()):

            if _key == action:
                continue
            else:
                _temp_state[_key] += 1

        self.state = _temp_state
        # observation, reward, done, info
        observation = self.get_observation()

        _mean = np.array(self.state.values()).mean()
        new_score = np.abs((np.array(self.state.values()) - _mean).mean())

        reward = new_score - score

        done = False
        if len(set(self.state)) == 1:
            done == True
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
    