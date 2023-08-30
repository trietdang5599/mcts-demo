from collections import OrderedDict

import gym
import torch


class LanguageEnv(gym.Env):
    """
    Langauge generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Transition: the next state is the current state concatenated with the action.
    Reward: an external function that evaluates a state (pass rate for programs, alignment score for natural language, etc.)
    Terminal state: the program reaches the maximum length or the terminal token is generated.
    """
    def __init__(self, terminal_token, horizon, reward_func):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_token = terminal_token
        self.horizon = horizon

        self.get_reward = reward_func

    def reset(self, init_state):
        self.state = init_state
        return self.state

    def transition(self, s, a, is_model_dynamic=False):
        # s is a one-dimensional tensor, a is a token id (scalar), concatenate them to form a new state
        next_state = torch.cat([s, torch.tensor([a]).to(s.device)])

        if a == self.terminal_token or len(next_state) == self.horizon:
            # either the text finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            reward = self.get_reward(next_state)
        else:
            reward = 0  # no intermediate reward

        return next_state, reward, done

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)

        return self.state, reward, done, {}

    def equality_operator(self, s1, s2):
        # s1 and s2 are two tensors
        return torch.equal(s1, s2)
