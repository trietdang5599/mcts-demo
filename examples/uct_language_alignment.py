import gym
import dyna_gym.agents.uct as uct
import transformers

from dyna_gym.default_policy.hf_default_policy import HuggingFaceDefaultPolicy
from dyna_gym.utils.tree_search_utils import plot_tree


model_name = 'gpt2'

model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
terminal_token = tokenizer.eos_token_id

def length_reward_func(state: list) -> float:
    # token_ids -> reward
    # the shorter the better
    return -len(state)

horizon = 100

### Parameters
env = gym.make('LanguageEnv-v0', terminal_token=terminal_token, horizon=horizon, reward_func=length_reward_func)
agent = uct.UCT(
    rollouts=5,
    gamma=1.,
    width=3,
    default_policy=HuggingFaceDefaultPolicy(k=3, env=env, horizon=horizon, model=model),
)

### Run
input_str = "Hello, I'm a language model,"
init_state = tokenizer.encode(input_str)
env.reset(init_state)

done = False
__, __, done, __ = env.step(agent.act(env,done))
sequences = agent.rolled_out_trajectories

for seq in sequences:
    print(tokenizer.decode(seq))

# plot (and print) the tree
plot_tree(agent.root, env, tokenizer,"tree")
