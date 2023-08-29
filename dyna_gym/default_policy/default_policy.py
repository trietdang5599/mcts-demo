from abc import abstractmethod

import gym


class DefaultPolicy:
    def __init__(self, k: int, env: gym.Env, horizon: int):
        """
        Args:
            k: number of top k predictions to return
            env: environment
            horizon: horizon of the environment (the maximum number of steps in an episode)
        """
        self.k = k
        self.env = env
        self.horizon = horizon

    @abstractmethod
    def get_predicted_sequence(self, state, horizon: int = None):
        pass

    @abstractmethod
    def get_top_k_tokens(self, state):
        pass
