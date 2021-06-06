from gym.spaces import Discrete, MultiDiscrete, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import numpy as np
import torch
from torch import nn


class MultiActionAutoregressiveAgent(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        assert isinstance(action_space,
                          Tuple), 'action space is not a tuple. make sure to use the MultiActionEnv wrapper.'
        single_action_space = action_space[0]

        self.action_space_parts = None

        if isinstance(single_action_space, Discrete):
            self._num_action_logits = single_action_space.n
            self.action_space_parts = [self._num_action_logits]
        elif isinstance(single_action_space, MultiDiscrete):
            self._num_action_logits = np.sum(single_action_space.nvec)
            self.action_space_parts = [*single_action_space.nvec]
        else:
            raise RuntimeError('Can only be used with discrete and multi-discrete action spaces')

        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

    def _encode_one_hot_actions(self, action):
        batch_size = action.shape[0]
        one_hot_actions = torch.zeros([batch_size, self._num_action_logits]).to(action.device)
        offset = 0
        for i, num_logits in enumerate(self.action_space_parts):
            oh_idxs = (offset + action[:, i]).type(torch.LongTensor)
            one_hot_actions[torch.arange(batch_size), oh_idxs] = 1
            offset += num_logits
        return one_hot_actions

    def observation_features_module(self, input_dict, states, seq_lens, **kwargs):
        raise NotImplementedError

    def action_features_module(self, input_dict, states, seq_lens, **kwargs):
        raise NotImplementedError

    def action_module(self, action_features, embedded_action, **kwargs):
        raise NotImplementedError

    def embed_action_module(self, action, **kwargs):
        raise NotImplementedError

    def from_batch(self, train_batch, is_training=True):

        input_dict = train_batch.copy()
        input_dict["is_training"] = is_training
        states = []
        i = 0
        while "state_in_{}".format(i) in input_dict:
            states.append(input_dict["state_in_{}".format(i)])
            i += 1
        return self.observation_features_module(input_dict, states, input_dict.get("seq_lens"))

    def forward(self, input_dict, state, seq_lens):
        # Just do the state embedding here, actions are decoded as part of the distribution
        raise NotImplementedError

    def value_function(self):
        """
        This is V(s) depending on whatever the last state was.
        """
        return self._value
