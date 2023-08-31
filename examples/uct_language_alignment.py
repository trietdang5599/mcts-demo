import pprint

import transformers

from dyna_gym.pipelines import uct_for_hf_transformer_pipeline


def length_reward_func(input_str: list) -> float:
    # input_ids -> reward
    # the shorter, the better
    return -len(input_str)

horizon = 100

# arguments for UCT agent
uct_args = dict(
    rollouts = 5,
    gamma = 1.,
    width = 3,
    alg = 'uct',
)

# will be passed to huggingface model.generate()
model_generation_args = dict(
    top_k = 3,
    top_p = 0.9,
    do_sample = True,
    temperature = 0.7,
)

model_name = "gpt2"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

pipeline = uct_for_hf_transformer_pipeline(
    model = model,
    tokenizer = tokenizer,
    horizon = horizon,
    reward_func = length_reward_func,
    uct_args = uct_args,
    model_generation_args = model_generation_args,
    should_plot_tree = True,
)
input_ids = tokenizer.encode("Hello, I'm a language model,")
outputs = pipeline(input_ids=input_ids)

# decoded texts
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
