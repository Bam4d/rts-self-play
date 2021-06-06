import torch
from griddly.util.rllib.torch.agents.common import layer_init
from griddly.util.rllib.torch.agents.impala_cnn import ImpalaCNNAgent
from torch import nn

from autocats.torch.models.multi_action_autoregressive_model import MultiActionAutoregressiveAgent


class MACombined(MultiActionAutoregressiveAgent):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_model_config = model_config['custom_model_config']

        self._observation_features_module_class = custom_model_config.get('observation_features_module_class', ImpalaCNNAgent)
        self._observation_features_size = custom_model_config.get('osbervation_features_size', 256)

        self._observation_features_module = self._observation_features_module_class(
            obs_space,
            action_space,
            self._observation_features_size,
            model_config,
            name
        )

        self._action_embedding_network = nn.Sequential(
            layer_init(nn.Linear(self._num_action_logits, self._observation_features_size))
        )

        self._action_logits_network = nn.Sequential(
            layer_init(nn.Linear(self._observation_features_size * 2, self._observation_features_size)),
            nn.ReLU(),
            layer_init(nn.Linear(self._observation_features_size, self._observation_features_size)),
            nn.ReLU(),
            layer_init(nn.Linear(self._observation_features_size, self._num_action_logits), std=0.01)
        )

    def action_features_module(self, input_dict, states, seq_lens, **kwargs):
        return self._observation_features, self._state_out

    def observation_features_module(self, input_dict, states, seq_lens, **kwargs):
        self._observation_features, self._state_out = self._observation_features_module(input_dict)
        self._value = self._observation_features_module.value_function()
        return self._observation_features, self._state_out

    def action_module(self, action_features, embedded_action, **kwargs):

        action_network_inputs = torch.cat([action_features, embedded_action], dim=1)
        return self._action_logits_network(action_network_inputs)

    def embed_action_module(self, action, **kwargs):
        one_hot_actions = self._encode_one_hot_actions(action)
        return self._action_embedding_network(one_hot_actions)
