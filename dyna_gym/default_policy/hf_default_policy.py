from dyna_gym.default_policy.default_policy import DefaultPolicy

import gym
import torch
from transformers import PreTrainedModel


class HuggingFaceDefaultPolicy(DefaultPolicy):
    """
    Default policy that uses a HuggingFace transformer model.
    """
    def __init__(self, k: int, env: gym.Env, horizon: int, model: PreTrainedModel):
        super().__init__(k, env, horizon)
        self.model = model

    @torch.no_grad()
    def get_predicted_sequence(self, state, horizon=None):
        horizon = horizon if horizon is not None else self.horizon

        # Convert the tokenized state into a PyTorch tensor
        input_data = torch.tensor([state])

        outputs = self.model.generate(
            input_data,
            top_k=self.k,
            max_length=horizon,
            early_stopping=True,
            use_cache=True,
        )

        return outputs[0].tolist()

    @torch.no_grad()
    def get_top_k_tokens(self, state):
        # Convert the tokenized state into a PyTorch tensor
        input_data = torch.tensor([state])

        outputs = self.model(**{'input_ids': input_data})

        # Assuming the model returns logits for tokens
        logits = outputs.logits[0][-1]  # First (and only) batch, last token

        # Get the top k logits and their indices
        topk_values, topk_indices = torch.topk(logits, self.k)

        # Convert PyTorch tensors to Python lists
        topk_values_list = topk_values.tolist()
        topk_indices_list = topk_indices.tolist()

        return topk_indices_list, topk_values_list
