import pprint

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

# will be passed to model.generate()
model_generation_args = dict(
    top_k = 3,
    top_p = 0.2,
    do_sample = True,
    temperature = 0.7,
)

pipeline = uct_for_hf_transformer_pipeline(
    model_name = "gpt2",
    # alternatively, you can pass in a model and a tokenizer
    horizon = horizon,
    reward_func = length_reward_func,
    uct_args = uct_args,
    model_generation_args = model_generation_args,
    should_plot_tree = True,
)
outputs = pipeline(input_str="Hello, I'm a language model,")

# output tokens
print(outputs['output_ids'])
# decoded texts
print(outputs['texts'])
