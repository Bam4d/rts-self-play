import numpy as np
import torch
from ray.rllib import Policy, SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override
from ray.rllib.utils.torch_ops import convert_to_non_torch_type

from autocats.torch.torch_auto_cat_exploration import TorchAutoCATExploration


class AutoCATMixin:
    def __init__(self):
        self.view_requirements = {
            SampleBatch.INFOS: ViewRequirement(data_col=SampleBatch.INFOS, shift=-1)
        }

    @override(Policy)
    def compute_actions_from_input_dict(
            self,
            input_dict,
            explore=None,
            timestep=None,
            **kwargs):

        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        with torch.no_grad():
            # Pass lazy (torch) tensor dict to Model as `input_dict`.
            input_dict = self._lazy_tensor_dict(input_dict)
            # Pack internal state inputs into (separate) list.
            state_batches = [
                input_dict[k] for k in input_dict.keys() if "state_in" in k[:8]
            ]
            # Calculate RNN sequence lengths.
            seq_lens = np.array([1] * len(input_dict["obs"])) \
                if state_batches else None

            self._is_recurrent = state_batches is not None and state_batches != []

            # Switch to eval mode.
            self.model.eval()

            infos = input_dict[SampleBatch.INFOS] if SampleBatch.INFOS in input_dict else {}

            valid_action_trees = []
            for info in infos:
                if isinstance(info, dict) and 'valid_action_tree' in info:
                    valid_action_trees.append(info['valid_action_tree'])
                else:
                    valid_action_trees.append({})

            extra_fetches = {}

            actions_per_step = self.config['actions_per_step']
            autoregressive_actions = self.config['autoregressive_actions']

            step_actions_list = []
            step_masked_logits_list = []
            step_logp_list = []
            step_mask_list = []

            observation_features, state_out = self.model.observation_features_module(input_dict, state_batches,
                                                                                     seq_lens)
            action_features, _ = self.model.action_features_module(input_dict, state_batches, seq_lens)

            embedded_action = None
            for a in range(actions_per_step):
                if autoregressive_actions:
                    if a == 0:
                        batch_size = action_features.shape[0]
                        previous_action = torch.zeros([batch_size, len(self.model.action_space_parts)]).to(action_features.device)
                    else:
                        previous_action = actions

                    embedded_action = self.model.embed_action_module(previous_action)

                dist_inputs = self.model.action_module(action_features, embedded_action)

                exploration = TorchAutoCATExploration(
                    self.model,
                    dist_inputs,
                    valid_action_trees,
                )

                actions, masked_logits, logp, mask = exploration.get_actions_and_mask()

                # Remove the performed action from the trees
                for batch_action, batch_tree in zip(actions, valid_action_trees):
                    x = int(batch_action[0])
                    y = int(batch_action[1])
                    # Assuming we have x,y coordinates
                    del batch_tree[x][y]
                    if len(batch_tree[x]) == 0:
                        del batch_tree[x]

                step_actions_list.append(actions)
                step_masked_logits_list.append(masked_logits)
                step_logp_list.append(logp)
                step_mask_list.append(mask)

            step_actions = tuple(step_actions_list)
            step_masked_logits = torch.hstack(step_masked_logits_list)
            step_logp = torch.sum(torch.stack(step_logp_list,dim=1), dim=1)
            step_mask = torch.hstack(step_mask_list)

            extra_fetches.update({
                'invalid_action_mask': step_mask
            })

            input_dict[SampleBatch.ACTIONS] = step_actions

            extra_fetches.update({
                SampleBatch.ACTION_DIST_INPUTS: step_masked_logits,
                SampleBatch.ACTION_PROB: torch.exp(step_logp.float()),
                SampleBatch.ACTION_LOGP: step_logp,
            })

            # Update our global timestep by the batch size.
            self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])

            return convert_to_non_torch_type((step_actions, state_out, extra_fetches))
