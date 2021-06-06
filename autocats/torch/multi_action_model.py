import numpy as np
import torch
from griddly.util.rllib.torch.agents.common import layer_init
from griddly.util.rllib.torch.agents.impala_cnn import ImpalaCNNAgent
from gym.spaces import MultiDiscrete, Discrete, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
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


class MultiActionAutoregressiveModel(TorchModelV2, nn.Module):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        custom_model_config = model_config['custom_model_config']
        # self._observation_features_module_class = custom_model_config.get('observation_features_class', ImpalaCNNAgent)
        self._observation_features_size = custom_model_config.get('observation_features_size', 256)

        assert isinstance(action_space,
                          Tuple), 'action space is not a tuple. make sure to use the MultiActionEnv wrapper.'
        single_action_space = action_space[0]

        self._action_space_parts = None

        if isinstance(single_action_space, Discrete):
            self._num_action_logits = single_action_space.n
            self._action_space_parts = [self._num_action_logits]
        elif isinstance(single_action_space, MultiDiscrete):
            self._num_action_logits = np.sum(single_action_space.nvec)
            self._action_space_parts = [*single_action_space.nvec]
        else:
            raise RuntimeError('Can only be used with discrete and multi-discrete action spaces')

        # assert self._observation_features_size == num_outputs, 'bruh.'

        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Create the observation features network (By default use IMPALA CNN)
        # self._observation_features_module = self._observation_features_module_class(
        #     obs_space,
        #     action_space,
        #     self._observation_features_size,
        #     model_config,
        #     name
        # )

        num_channels = 64
        self._observation_features_module = SimpleConvEncoderModule(obs_space, num_channels)

        # Action embedding network
        self._action_embedding_module = ActionEmbedderModule(obs_space, num_channels, self._num_action_logits)

        # Actor head
        self._action_module = ActionModule(obs_space, num_channels, self._num_action_logits)

    def embed_action(self, action):
        # One-hot encode the action selection
        batch_size = action.shape[0]
        one_hot_actions = torch.zeros([batch_size, self._num_action_logits]).to(action.device)
        offset = 0
        for i, num_logits in enumerate(self._action_space_parts):
            oh_idxs = (offset + action[:, i]).type(torch.LongTensor)
            one_hot_actions[torch.arange(batch_size), oh_idxs] = 1
            offset += num_logits

        return self._action_embedding_module(one_hot_actions)

    def observation_features_module(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2)
        observation_features = self._observation_features_module(obs_transformed)
        self._value = self._observation_features_module.value_function()

        return observation_features, state

    def action_module(self, observation_features, embedded_action=None):
        if embedded_action is None:
            batch_size = observation_features.shape[0]
            zero_actions = torch.zeros([batch_size, len(self._action_space_parts)]).to(observation_features.device)
            embedded_action = self.embed_action(zero_actions)
        return self._action_module(observation_features, embedded_action)

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
