import random
import math

import torch
import numpy as np

from garage.core import Serializable
from garage.torch.policies import Policy


class EpsilonGreedyPolicy(Policy, Serializable):
    def __init__(self,
                 env_spec,
                 qfunction,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=200):
        super().__init__(env_spec)
        Serializable.quick_init(self, locals())
        self._steps = 0
        self.qfunction = qfunction
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.epsilon = 1

    def reset(self):
        # self._steps = 0
        pass

    def get_action(self, observation):
        sample = random.random()
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                            math.exp(-1. * self._steps / self.eps_decay)
        self._steps += 1
        if sample > self.epsilon:
            with torch.no_grad():
                qval = self.qfunction.get_qval(torch.tensor(observation, dtype=torch.float))
                # action = np.argmax(qval.detach().numpy())
                action = qval.max(0)[1].item()
                return action, {}
        else:
            return self._env_spec.action_space.sample(), {}

    def get_params_internal(self, **tags):
        """
        Internal method to be implemented which does not perform caching
        """
        return self.qfunction.get_params_internal(**tags)
