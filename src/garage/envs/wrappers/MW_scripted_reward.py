"""
Wrapper for replacing env reward with -l2 distance
of scripted action and action
"""

import numpy as np
import gym

class MWScriptedReward(gym.Wrapper):
    """Modifies the metaworld env reward

    Replaces the environment's reward with the negative l2 distance between the action
    that is taken by the policy, and the action that should be taken by the scripted policy.

    Args:
        env(gym.env): The metaworld environment that is having its reward modified/
        scripted_policy(metaworld.policy): The metaworld scripted policy that corresponds to env
    """
    def __init__(self, env, scripted_policy, c=100):
        super().__init__(env)
        self.c = c
        self._scripted_policy = scripted_policy

    def step(self, action):
        """Call step on wrapped env.

        This method will intercept the reward returned by the environment
        and replace it with the negative l2 distance between the action
        and scripted action.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            np.ndarray: Agent's observation of the current environment
            float: Amount of reward returned after previous action
            bool: Whether the episode has ended, in which case further step()
                calls will return undefined results
            dict: Contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning)

        """
        curr_obs = self.env._get_obs()
        scripted_action = self._scripted_policy.get_action(curr_obs)
        observation, _, done, info = self.env.step(action)
        reward = - np.linalg.norm(action - scripted_action) * self.c
        return observation, reward, done, info
