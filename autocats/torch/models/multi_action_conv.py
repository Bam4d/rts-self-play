import torch
from autocats.torch.models.multi_action_autoregressive_model import MultiActionAutoregressiveAgent
from griddly.util.rllib.torch.agents.common import layer_init
from torch import nn
from torch.nn import Flatten


class SimpleConvEncoderModule(nn.Module):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, num_out_channels):
        nn.Module.__init__(self)

        self._num_objects = obs_space.shape[2]
        self._num_outputs = obs_space.shape[0] * obs_space.shape[1] * num_out_channels

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self._num_objects, num_out_channels, 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(num_out_channels, num_out_channels, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(num_out_channels, num_out_channels, 3, padding=1)),
        )

        self._critic_head = nn.Sequential(
            Flatten(),
            layer_init(nn.Linear(self._num_outputs, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.01)
        )

    def forward(self, input):
        encoded_obs = self.network(input)

        value = self._critic_head(encoded_obs)
        self._value = value.reshape(-1)
        return encoded_obs

    def value_function(self):
        return self._value


class ActionEmbedderModule(nn.Module):

    def __init__(self, obs_space, embedding_features, num_action_logits):
        super().__init__()

        self._height = obs_space.shape[0]
        self._width = obs_space.shape[1]

        self._embedding_features = embedding_features
        self._num_action_logits = num_action_logits

        self._network = nn.Sequential(
            nn.Conv2d(num_action_logits, embedding_features, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, action):
        tiled_action = torch.tile(
            action.reshape(-1, self._num_action_logits, 1, 1),
            (1, 1, self._height, self._width)
        )

        return self._network(tiled_action)


class ActionModule(nn.Module):

    def __init__(self, obs_space, num_channels, num_logits):
        super().__init__()

        num_outputs = obs_space.shape[0] * obs_space.shape[1] * num_channels

        # might be better to be a cgru?
        self._network = nn.Sequential(
            layer_init(nn.Conv2d(num_channels, num_channels, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(num_channels, num_channels, 3, padding=1)),
            nn.ReLU(),
        )

        self._action_head = nn.Sequential(
            Flatten(),
            layer_init(nn.Linear(num_outputs, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_logits), std=0.01)
        )

    def forward(self, observation_embedding, action_embedding):
        input = observation_embedding + action_embedding
        state = self._network(input) + input
        return self._action_head(state)


class MAConvs(MultiActionAutoregressiveAgent):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        num_channels = model_config.get('num_channels', 64)
        self._observation_features_module = SimpleConvEncoderModule(obs_space, num_channels)

        # Action embedding network
        self._action_embedding_module = ActionEmbedderModule(obs_space, num_channels, self._num_action_logits)

        # Actor head
        self._action_module = ActionModule(obs_space, num_channels, self._num_action_logits)

    def embed_action_module(self, action, **kwargs):
        # One-hot encode the action selection
        one_hot_actions = self._encode_one_hot_actions(action)
        return self._action_embedding_module(one_hot_actions)

    def action_features_module(self, input_dict, states, seq_lens, **kwargs):
        return self._observation_features, self._state_out

    def observation_features_module(self, input_dict, state, seq_lens, **kwargs):
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2)
        self._observation_features = self._observation_features_module(obs_transformed)
        self._value = self._observation_features_module.value_function()
        return self._observation_features, state

    def action_module(self, observation_features, embedded_action=None, **kwargs):
        if embedded_action is None:
            batch_size = observation_features.shape[0]
            zero_actions = torch.zeros([batch_size, len(self._action_space_parts)]).to(observation_features.device)
            embedded_action = self.embed_action_module(zero_actions)
        return self._action_module(observation_features, embedded_action)

