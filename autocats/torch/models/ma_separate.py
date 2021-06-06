import torch
from griddly.util.rllib.torch.agents.common import layer_init
from griddly.util.rllib.torch.agents.impala_cnn import ImpalaCNNAgent
from torch import nn

from autocats.torch.models.multi_action_autoregressive_model import MultiActionAutoregressiveAgent


class MASeparate(MultiActionAutoregressiveAgent):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_model_config = model_config['custom_model_config']

        self._observation_features_module_class = custom_model_config.get('observation_features_module_class', ImpalaCNNAgent)
        self._action_features_module_class = custom_model_config.get('action_features_module_class', ImpalaCNNAgent)
        self._observation_features_size = custom_model_config.get('osbervation_features_size', 256)

        self._observation_features_module = self._observation_features_module_class(
            obs_space,
            action_space,
            self._observation_features_size,
            model_config,
            name
        )

        self._action_features_module = self._action_features_module_class(
            obs_space,
            action_space,
            self._observation_features_size,
            model_config,
            name
        )

        self._action_embedding_network = nn.Sequential(
            layer_init(nn.Linear(self._num_action_logits, self._observation_features_size), std=0.1)
        )

        self._action_logits_network = nn.Sequential(
            layer_init(nn.Linear(self._observation_features_size, self._observation_features_size), std=0.1),
            nn.ReLU(),
            layer_init(nn.Linear(self._observation_features_size, self._num_action_logits), std=0.01)
        )

        self._action_conditioner = nn.Sequential(
            layer_init(nn.Linear(self._observation_features_size*2, self._observation_features_size), std=0.1),
        )

    def action_features_module(self, input_dict, states, seq_lens, **kwargs):
        return self._action_features_module(input_dict)

    def observation_features_module(self, input_dict, states, seq_lens, **kwargs):
        observation_features, state_out = self._observation_features_module(input_dict)
        self._value = self._observation_features_module.value_function()
        return observation_features, state_out

    def action_module(self, action_features, embedded_action, **kwargs):
        if embedded_action is not None:
            features_and_embedding = torch.cat([action_features, embedded_action], dim=1)
            action_network_inputs = self._action_conditioner(features_and_embedding)
        else:
            action_network_inputs = action_features

        return self._action_logits_network(action_network_inputs)

    def embed_action_module(self, action, **kwargs):
        one_hot_actions = self._encode_one_hot_actions(action)
        return self._action_embedding_network(one_hot_actions)
