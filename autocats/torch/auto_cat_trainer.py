import gym
import numpy as np
import torch
from gym.spaces import Tuple, Discrete, MultiDiscrete
from ray.rllib import SampleBatch
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.impala.vtrace_torch_policy import build_vtrace_loss
from ray.rllib.agents.impala.vtrace_torch_policy import VTraceTorchPolicy, VTraceLoss, make_time_major
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, TorchMultiCategorical
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils.torch_ops import sequence_mask

from autocats.torch.auto_cat_mixin import AutoCATMixin

DEFAULT_CONFIG = with_common_config({
    # V-trace params (see vtrace_tf/torch.py).
    "vtrace": True,
    "vtrace_clip_rho_threshold": 1.0,
    "vtrace_clip_pg_rho_threshold": 1.0,
    # System params.
    #
    # == Overview of data flow in IMPALA ==
    # 1. Policy evaluation in parallel across `num_workers` actors produces
    #    batches of size `rollout_fragment_length * num_envs_per_worker`.
    # 2. If enabled, the replay buffer stores and produces batches of size
    #    `rollout_fragment_length * num_envs_per_worker`.
    # 3. If enabled, the minibatch ring buffer stores and replays batches of
    #    size `train_batch_size` up to `num_sgd_iter` times per batch.
    # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
    #    on batches of size `train_batch_size`.
    #
    "rollout_fragment_length": 50,
    "train_batch_size": 500,
    "min_iter_time_s": 10,
    "num_workers": 2,
    # number of GPUs the learner should use.
    "num_gpus": 1,
    # set >1 to load data into GPUs in parallel. Increases GPU memory usage
    # proportionally with the number of buffers.
    "num_data_loader_buffers": 1,
    # how many train batches should be retained for minibatching. This conf
    # only has an effect if `num_sgd_iter > 1`.
    "minibatch_buffer_size": 1,
    # number of passes to make over each train batch
    "num_sgd_iter": 1,
    # set >0 to enable experience replay. Saved samples will be replayed with
    # a p:1 proportion to new data samples.
    "replay_proportion": 0.0,
    # number of sample batches to store for replay. The number of transitions
    # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
    "replay_buffer_num_slots": 0,
    # max queue size for train batches feeding into the learner
    "learner_queue_size": 16,
    # wait for train batches to be available in minibatch buffer queue
    # this many seconds. This may need to be increased e.g. when training
    # with a slow environment
    "learner_queue_timeout": 300,
    # level of queuing for sampling.
    "max_sample_requests_in_flight_per_worker": 2,
    # max number of workers to broadcast one set of weights to
    "broadcast_interval": 1,
    # Use n (`num_aggregation_workers`) extra Actors for multi-level
    # aggregation of the data produced by the m RolloutWorkers
    # (`num_workers`). Note that n should be much smaller than m.
    # This can make sense if ingesting >2GB/s of samples, or if
    # the data requires decompression.
    "num_aggregation_workers": 0,

    # Learning params.
    "grad_clip": 40.0,
    # either "adam" or "rmsprop"
    "opt_type": "adam",
    "lr": 0.0005,
    "lr_schedule": None,
    # rmsprop considered
    "decay": 0.99,
    "momentum": 0.0,
    "epsilon": 0.1,
    # balancing the three losses
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "entropy_coeff_schedule": None,

    # Callback for APPO to use to update KL, target network periodically.
    # The input to the callback is the learner fetches dict.
    "after_train_step": None,

    # AutoCATParams
    "actions_per_step": 1,
    "autoregressive_actions": False,
})


def build_CAT_vtrace_loss(policy, model, dist_class, train_batch):
    action_space_parts = model.action_space_parts

    def _make_time_major(*args, **kw):
        return make_time_major(policy, train_batch.get("seq_lens"), *args,
                               **kw)

    # Repeat the output_hidden_shape depending on the number of actions that have been generated
    # output_hidden_shape = np.tile(output_hidden_shape, action_repeats)

    actions = train_batch[SampleBatch.ACTIONS]
    dones = train_batch[SampleBatch.DONES]
    rewards = train_batch[SampleBatch.REWARDS]
    behaviour_action_logp = train_batch[SampleBatch.ACTION_LOGP]
    behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]

    invalid_action_mask = train_batch['invalid_action_mask']
    autoregressive_actions = policy.config['autoregressive_actions']

    if 'seq_lens' in train_batch:
        max_seq_len = policy.config['rollout_fragment_length']
        mask_orig = sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = torch.reshape(mask_orig, [-1])
    else:
        mask = torch.ones_like(rewards)

    actions_per_step = policy.config["actions_per_step"]

    states = []
    i = 0
    while "state_in_{}".format(i) in train_batch:
        states.append(train_batch["state_in_{}".format(i)])
        i += 1

    seq_lens = train_batch["seq_lens"] if "seq_lens" in train_batch else []

    model.observation_features_module(train_batch, states, seq_lens)
    action_features, _ = model.action_features_module(train_batch, states, seq_lens)

    previous_action = None
    embedded_action = None
    logp_list = []
    entropy_list = []
    logits_list = []

    multi_actions = torch.chunk(actions, actions_per_step, dim=1)
    multi_invalid_action_mask = torch.chunk(invalid_action_mask, actions_per_step, dim=1)
    for a in range(actions_per_step):
        if autoregressive_actions:
            if a == 0:
                batch_size = action_features.shape[0]
                previous_action = torch.zeros([batch_size, len(action_space_parts)]).to(action_features.device)
            else:
                previous_action = multi_actions[a-1]

            embedded_action = model.embed_action_module(previous_action)

        logits = model.action_module(action_features, embedded_action)
        logits += torch.maximum(torch.tensor(torch.finfo().min), torch.log(multi_invalid_action_mask[a]))
        cat = TorchMultiCategorical(logits, model, action_space_parts)

        logits_list.append(logits)
        logp_list.append(cat.logp(multi_actions[a]))
        entropy_list.append(cat.entropy())

    logp = torch.stack(logp_list, dim=1).sum(dim=1)
    entropy = torch.stack(entropy_list, dim=1).sum(dim=1)
    target_logits = torch.hstack(logits_list)

    unpack_shape = np.tile(action_space_parts, actions_per_step)

    unpacked_behaviour_logits = torch.split(behaviour_logits, list(unpack_shape), dim=1)
    unpacked_outputs = torch.split(target_logits, list(unpack_shape), dim=1)

    values = model.value_function()

    # Inputs are reshaped from [B * T] => [T - 1, B] for V-trace calc.
    policy.loss = VTraceLoss(
        actions=_make_time_major(actions, drop_last=True),
        actions_logp=_make_time_major(logp, drop_last=True),
        actions_entropy=_make_time_major(entropy, drop_last=True),
        dones=_make_time_major(dones, drop_last=True),
        behaviour_action_logp=_make_time_major(
            behaviour_action_logp, drop_last=True),
        behaviour_logits=_make_time_major(
            unpacked_behaviour_logits, drop_last=True),
        target_logits=_make_time_major(unpacked_outputs, drop_last=True),
        discount=policy.config["gamma"],
        rewards=_make_time_major(rewards, drop_last=True),
        values=_make_time_major(values, drop_last=True),
        bootstrap_value=_make_time_major(values)[-1],
        dist_class=TorchCategorical,
        model=model,
        valid_mask=_make_time_major(mask, drop_last=True),
        config=policy.config,
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        entropy_coeff=policy.entropy_coeff,
        clip_rho_threshold=policy.config["vtrace_clip_rho_threshold"],
        clip_pg_rho_threshold=policy.config["vtrace_clip_pg_rho_threshold"])

    return policy.loss.total_loss


def setup_mixins(policy, obs_space, action_space, config):
    AutoCATMixin.__init__(policy)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


AutoCATVTraceTorchPolicy = VTraceTorchPolicy.with_updates(
    name="AutoCATVTraceTorchPolicy",
    loss_fn=build_CAT_vtrace_loss,
    before_init=setup_mixins,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule, AutoCATMixin]
)


def get_vtrace_policy_class(config):
    if config['framework'] == 'torch':
        return AutoCATVTraceTorchPolicy
    else:
        raise NotImplementedError('Tensorflow not supported')


AutoCATTrainer = ImpalaTrainer.with_updates(name='AutoCATTrainer',
                                            default_config=DEFAULT_CONFIG,
                                            default_policy=AutoCATVTraceTorchPolicy,
                                            get_policy_class=get_vtrace_policy_class)
