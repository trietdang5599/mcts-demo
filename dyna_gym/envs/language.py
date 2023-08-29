from collections import OrderedDict

import gym


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
        self._reward_func = reward_func

        # state -> reward
        # reward function may be expensive to compute (possibly using another neural model), so we cache the results
        self.cached_reward = OrderedDict()

    def reset(self, init_state):
        self.state = init_state
        return self.state

    def transition(self, s, a, is_model_dynamic=False):
        next_state = s + [a]

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

    def get_reward(self, state):
        if tuple(state) in self.cached_reward.keys():
            # cache rewards for training
            return self.cached_reward[tuple(state)]

        reward = self._reward_func(state)

        self.cached_reward[tuple(state)] = reward

        return reward

    def equality_operator(self, s1, s2):
        return s1 == s2
