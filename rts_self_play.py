import argparse
import os
import sys

import ray
from griddly import gd
from griddly.util.rllib.callbacks import VideoCallbacks, ActionTrackerCallbacks, WinLoseMetricCallbacks
from griddly.util.rllib.environment.core import RLlibMultiAgentWrapper, RLlibEnv
from ray import tune
from ray.rllib.agents.callbacks import MultiCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env

from autocats.torch.auto_cat_trainer import AutoCATTrainer
from autocats.torch.models.ma_separate import MASeparate
from autocats.torch.multi_action_model import MultiActionAutoregressiveModel
from autocats.wrappers.multi_action_env import MultiActionEnv

parser = argparse.ArgumentParser(description='Run experiments')

parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--yaml-file', help='YAML file containing GDY for the game')
parser.add_argument('--root-directory', default=os.path.expanduser("~/ray_results"),
                    help='root directory for all data associated with the run')
parser.add_argument('--num-gpus', default=1, type=int, help='Number of GPUs to make available to ray.')
parser.add_argument('--num-cpus', default=8, type=int, help='Number of CPUs to make available to ray.')

parser.add_argument('--num-workers', default=7, type=int, help='Number of workers')
parser.add_argument('--num-envs-per-worker', default=5, type=int, help='Number of workers')
parser.add_argument('--num-gpus-per-worker', default=0, type=float, help='Number of gpus per worker')
parser.add_argument('--num-cpus-per-worker', default=1, type=float, help='Number of gpus per worker')
parser.add_argument('--max-training-steps', default=20000000, type=int, help='Number of workers')
parser.add_argument('--train-batch-size', default=500, type=int, help='Training batch size')

parser.add_argument('--capture-video', action='store_true', help='enable video capture')
parser.add_argument('--video-directory', default='videos', help='directory of video')
parser.add_argument('--video-frequency', type=int, default=1000000, help='Frequency of videos')
parser.add_argument('--actions-per-step', default=1, type=int, help='Number of actions to produce per time-step')

parser.add_argument('--seed', type=int, default=1, help='seed for experiments')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--entropy-coeff', type=float, default=0.01, help='entropy coefficient')


if __name__ == '__main__':

    args = parser.parse_args()

    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    if args.debug:
        ray.init(include_dashboard=False, num_gpus=args.num_gpus, num_cpus=args.num_cpus, local_mode=True)
    else:
        ray.init(include_dashboard=False, num_gpus=args.num_gpus, num_cpus=args.num_cpus)

    env_name = "griddly-rts-env"


    def _create_env(env_config):
        env = RLlibEnv(env_config)
        env = MultiActionEnv(env, env_config['actions_per_step'])
        return RLlibMultiAgentWrapper(env, env_config)


    register_env(env_name, _create_env)
    ModelCatalog.register_custom_model("AutoCat", MASeparate)

    wandbLoggerCallback = WandbLoggerCallback(
        project='rts_experiments',
        api_key_file='~/.wandb_rc',
        dir=args.root_directory
    )

    max_training_steps = args.max_training_steps

    config = {
        'framework': 'torch',
        'seed': args.seed,
        'num_workers': args.num_workers,
        'num_envs_per_worker': args.num_envs_per_worker,
        'num_gpus_per_worker': float(args.num_gpus_per_worker),
        'num_cpus_per_worker': args.num_cpus_per_worker,

        'train_batch_size': args.train_batch_size,

        'callbacks': MultiCallbacks([
            VideoCallbacks,
            ActionTrackerCallbacks,
            WinLoseMetricCallbacks
        ]),

        'model': {
            'custom_model': 'AutoCat',
            'custom_model_config': {}
        },
        'env': env_name,
        'env_config': {
            'generate_valid_action_trees': True,
            'invalid_action_masking': 'conditional',
            'yaml_file': args.yaml_file,
            'global_observer_type': gd.ObserverType.ISOMETRIC,
            'level': 0,
            'record_actions': True,
            'actions_per_step': args.actions_per_step,
            'max_steps': 1000,
        },
        'actions_per_step': args.actions_per_step,
        # 'entropy_coeff': tune.grid_search([0.0005, 0.001, 0.002, 0.005]),
        # 'lr': tune.grid_search([0.0005, 0.0002, 0.0001, 0.00005])
        'entropy_coeff_schedule': [
            [0, args.entropy_coeff],
            [max_training_steps, 0.0]
        ],
        'lr_schedule': [
            [0, args.lr],
            [max_training_steps, 0.0]
        ],

    }

    if args.capture_video:
        real_video_frequency = int(args.video_frequency / (args.num_envs_per_worker * args.num_workers))
        config['env_config']['record_video_config'] = {
            'frequency': real_video_frequency,
            'directory': os.path.join(args.root_directory, args.video_directory)
        }

    stop = {
        "timesteps_total": max_training_steps,
    }

    trial_name_creator = lambda trial: f'RTS-self-play'

    result = tune.run(
        AutoCATTrainer,
        local_dir=args.root_directory,
        config=config,
        stop=stop,
        callbacks=[wandbLoggerCallback],
        trial_name_creator=trial_name_creator
    )
