from datetime import datetime
from typing import Callable, Sequence

import gym
import torch
import transformers

from dyna_gym.agents import uct
from dyna_gym.default_policy.hf_default_policy import HuggingFaceDefaultPolicy
from dyna_gym.utils.tree_search_utils import print_tree

# --- Compat helpers for old Gym (0.20/0.21) vs new Gym/Gymnasium (>=0.26) ---
def _compat_reset(env, input_ids, attention_mask):
    """
    Try multiple reset signatures:
      1) new API: reset(options={...})
      2) old-like keyword passthrough: reset(input_ids=..., attention_mask=...)
      3) bypass wrapper: env.unwrapped.reset(input_ids, attention_mask)
    Return: obs (or (obs, info) -> we normalize to (obs, info))
    """
    # 1) New API (keyword-only): seed=None, options=dict
    try:
        out = env.reset(options={"input_ids": input_ids, "attention_mask": attention_mask})
        # Gymnasium: (obs, info); Old gym w/ wrapper might return obs
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
        else:
            return out, {}
    except TypeError:
        pass

    # 2) Some old wrappers forward **kwargs
    try:
        out = env.reset(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
        else:
            return out, {}
    except TypeError:
        pass

    # 3) Fallback: call unwrapped (custom env reset signature)
    out = env.unwrapped.reset(input_ids, attention_mask)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    else:
        return out, {}

def _compat_step(env, action):
    """
    Normalize step returns to (obs, reward, done, info).
    - New API: (obs, reward, terminated, truncated, info)
    - Old API: (obs, reward, done, info)
    """
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return obs, reward, done, info
        elif len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, bool(done), info
    raise RuntimeError("Unexpected env.step() return format: {}".format(type(out)))


def uct_for_hf_transformer_pipeline(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        horizon: int = 100,
        reward_func: Callable = None,
        uct_args: dict = {},
        model_generation_args: dict = {},
        should_plot_tree: bool = False,
        reward_func_input_is_state: bool = False,
) -> Callable:
    """
    A wrapped UCT agent for HuggingFace transformer.

    Args:
        model_name: The name of a HuggingFace transformer model. If provided, will load the model and tokenizer.
        model: A HuggingFace transformer model.
        tokenizer: A HuggingFace tokenizer.
        horizon: The maximum number of steps to take.
        reward_func: A function that evaluate the reward of a sequence.
        value_func: A function that evaluate the value of a sequence.
        uct_args: Arguments for the UCT agent.
        model_generation_args: Arguments for the model generation.
        should_plot_tree: Whether to plot the tree after generation.
        reward_func_input_is_state: Whether the input of the reward function is (token ids, attention masks) or tokenized text.
    """
    eos_token_id = tokenizer.eos_token_id

    if not reward_func_input_is_state:
        # by default reward function takes tokenized text as input
        # if reward function takes texts as input, wrap it here to take (token ids, attention masks) as input
        def reward_func_(state):
            ids, attention_mask = state
            text = tokenizer.decode(ids, skip_special_tokens=True)
            return reward_func(text)
    else:
        reward_func_ = reward_func

    env = gym.make(
        'LanguageEnv-v0',
        terminal_token=eos_token_id,
        horizon=horizon,
        reward_func=reward_func_,
    )

    default_policy = HuggingFaceDefaultPolicy(
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
    def generate(input_ids=None, input_str=None, attention_mask=None):
        assert (input_ids is None) != (input_str is None), "Only provide one of input_ids and input_str"

        if input_str is not None:
            input_ids = tokenizer.encode(input_str)
            input_ids = torch.tensor(input_ids).to(model.device)

        if attention_mask is None:
            # attend to tokens that are not padding
            if tokenizer.pad_token_id is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
            attention_mask = attention_mask.to(model.device)

        # env.reset(input_ids, attention_mask)
        obs, info = _compat_reset(env, input_ids, attention_mask)
        # do all rollouts in one step
        # env.step(agent.act(env, done=False))
        obs, reward, done, info = _compat_step(env, agent.act(env, done=False))

        # print tree
        print_tree(agent.root, tokenizer)
        # optionally, plot the tree and save to a pdf file
        if should_plot_tree:
            # plot (and print) the tree
            from dyna_gym.utils.tree_search_utils import plot_tree
            filename = f"tree-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            plot_tree(agent.root, tokenizer, filename)
            print(f"Tree plotted and saved to {filename}.pdf")

        results = {
            'output_ids': agent.rolled_out_trajectories,
            'rewards': agent.rolled_out_rewards,
            'texts': [tokenizer.decode(ids, skip_special_tokens=True) for ids in agent.rolled_out_trajectories],
        }

        # clear for the next generation call
        agent.reset()

        return results

    return generate
