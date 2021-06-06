import gym
from gym.spaces import Tuple

class MultiActionEnv(gym.Wrapper):

    def __init__(self, env, actions_per_step):
        super().__init__(env)

        # multiply the action space by the number of actions
        self.action_space = Tuple((self.action_space,)*actions_per_step)