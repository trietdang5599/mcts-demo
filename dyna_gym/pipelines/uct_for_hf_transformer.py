from typing import Callable, Sequence

import gym
import transformers

from dyna_gym.agents import uct
from dyna_gym.default_policy.hf_default_policy import HuggingFaceDefaultPolicy


def uct_for_hf_transformer_pipeline(
        model_name: str,
        horizon: int,
        reward_func: Callable,
        uct_args: dict,
        model_generation_args: dict = {},
        should_plot_tree: bool = False,
        reward_func_takes_str_as_input: bool = True,
) -> Callable[[str], Sequence[str]]:
    """
    A wrapped UCT agent for HuggingFace transformer.

    Args:
        model_name: The name of the Transformer model to use.
        horizon: The maximum number of steps to take.
        reward_func: A function that evaluate the reward of a sequence.
        uct_args: Arguments for the UCT agent.
        model_generation_args: Arguments for the model generation.
        should_plot_tree: Whether to plot the tree after generation.
        reward_func_takes_str_as_input: Whether the reward function takes a string as input.
            if so, need to call tokenizer.decode() before calling the reward function.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    terminal_token = tokenizer.eos_token_id

    if reward_func_takes_str_as_input:
        _reward_func = lambda seq: reward_func(tokenizer.decode(seq))
    else:
        _reward_func = reward_func

    env = gym.make('LanguageEnv-v0', terminal_token=terminal_token, horizon=horizon, reward_func=_reward_func)

    default_policy = HuggingFaceDefaultPolicy(
        k=uct_args['width'],
        env=env,
        horizon=horizon,
        model=model,
        generation_args=model_generation_args,
    )

    agent = uct.UCT(
        default_policy=default_policy,
        **uct_args
    )

    ### Run
    def generate(input_str):
        init_state = tokenizer.encode(input_str)
        env.reset(init_state)

        env.step(agent.act(env, done=False))
        trajectories = agent.rolled_out_trajectories

        seqs = [tokenizer.decode(seq) for seq in trajectories]

        if should_plot_tree:
            from dyna_gym.utils.tree_search_utils import plot_tree
            # plot (and print) the tree
            plot_tree(agent.root, env, tokenizer,"tree")

        return seqs

    return generate
