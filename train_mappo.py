from algorithms.MARL.MAPPO import MAPPO
from algorithms.MARL.common.utils import agg_double_list, copy_file_ppo, init_dir

import numpy as np
from sumo_env import SUMOEnv
import argparse
import os
import torch
from tqdm import tqdm
import wandb
from pathlib import Path


reward_type = "Global"
# reward_type = "Local"
mode = False
testFlag = False
USE_CUDA = False  # torch.cuda.is_available()


def train(config):
    use_wandb = os.environ.get('WANDB_MODE', 'online') # can be online, offline, or disabled
    wandb.init(
    project="prioritylane",
    tags=["MultiAgent", "RL"],
    mode=use_wandb
    )
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'mappo_run1'
    else:
        exst_run_nums = [int(str(folder.name).split('mappo_run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('mappo_run')]
        if len(exst_run_nums) == 0:
            curr_run = 'mappo_run1'
        else:
            curr_run = 'mappo_run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
    #                         config.discrete_action, num_agents=config.n_agents, action_step=config.action_step)
    # unwrapped_env = env.envs[0]

    env = SUMOEnv(mode=mode,testFlag=testFlag, num_agents=config.n_agents,action_step=config.action_step,
                  episode_duration=config.episode_duration)
    unwrapped_env = env
    assert unwrapped_env.n == config.n_agents

    normalize_rewards = True
    # Log configs
    wandb.config.algorithm = 'MAPPO'
    wandb.config.lr = config.lr
    wandb.config.gamma = config.gamma
    wandb.config.batch_size = config.batch_size
    wandb.config.n_rl_agents = config.n_agents
    wandb.config.action_step = config.action_step
    wandb.config.normalize_rewards = normalize_rewards



    MEMORY_CAPACITY = config.buffer_length
    state_dim = unwrapped_env._num_observation
    action_dim = unwrapped_env._num_actions
    BATCH_SIZE = config.batch_size
    ROLL_OUT_N_STEPS = 20 #(config.episode_duration//config.action_step)//4 # so we can call a train every 20 steps 
    actor_hidden_size = config.hidden_dim
    critic_hidden_size = config.hidden_dim
    actor_lr = config.lr
    critic_lr = config.lr
    reward_gamma = config.gamma
    EPISODES_BEFORE_TRAIN = 10

    MAX_EPISODES = config.n_episodes
    mappo = MAPPO(env=env, memory_capacity=MEMORY_CAPACITY,
                  state_dim=state_dim, action_dim=action_dim,
                  batch_size=BATCH_SIZE, 
                  roll_out_n_steps=ROLL_OUT_N_STEPS,
                  actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                  actor_lr=actor_lr, critic_lr=critic_lr, 
                  reward_scale=1,
                  reward_gamma=reward_gamma, reward_type='global_R',
                  episodes_before_train=EPISODES_BEFORE_TRAIN, 
                  )

    t = 0
    scores = []    
    smoothed_total_reward = 0
    EVAL_INTERVAL = 1
    with tqdm(total=config.n_episodes) as pbar:
        while mappo.n_episodes<config.n_episodes:
            ep_i = mappo.n_episodes
            total_reward = mappo.interact()
            if mappo.n_episodes >= EPISODES_BEFORE_TRAIN:
                print("---------------Training----------------")
                mappo.train()

            if ep_i == 0:
                smoothed_total_reward = total_reward
            smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
            scores.append(smoothed_total_reward)
            wandb.log({'# Episodes': ep_i, 
                    "Average Smooth Reward": smoothed_total_reward,
                    "Average Raw Reward": total_reward})
            
            if ep_i % config.save_interval < config.n_rollout_threads:
                mappo.save(run_dir, ep_i + 1 )
            if (mappo.n_episodes-ep_i) >= 1:
                print("Episodes %i-%i of %i" % (ep_i + 1,
                                ep_i + 1 + config.n_rollout_threads,
                                config.n_episodes))
                pbar.update(1)
    mappo.save(run_dir, ep_i + 1)
    env.close()


def evaluate(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    curr_run = config.run_id #+ config.model_id
    run_dir = model_dir / curr_run

    if not os.path.exists(model_dir):
        raise Exception("Sorry, no pretrained models")

    env = SUMOEnv(mode=True,testFlag=testFlag, num_agents=config.n_agents,action_step=config.action_step,
                  episode_duration=config.episode_duration)
    unwrapped_env = env
    assert unwrapped_env.n == config.n_agents

    MEMORY_CAPACITY = config.buffer_length
    state_dim = unwrapped_env._num_observation
    action_dim = unwrapped_env._num_actions
    BATCH_SIZE = config.batch_size
    ROLL_OUT_N_STEPS = (config.episode_duration//config.action_step)//4 # so we can call a train every 20 steps 
    actor_hidden_size = config.hidden_dim
    critic_hidden_size = config.hidden_dim
    actor_lr = config.lr
    critic_lr = config.lr
    reward_gamma = config.gamma
    EPISODES_BEFORE_TRAIN = 10

    MAX_EPISODES = config.n_episodes
    mappo = MAPPO(env=env, 
                  state_dim=state_dim, action_dim=action_dim,
                  roll_out_n_steps=ROLL_OUT_N_STEPS,
                  actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                  actor_lr=actor_lr, critic_lr=critic_lr)
    

    seeds = [1]
    # load the model if exist
    mappo.load(f'{run_dir}{os.sep}', train_mode=False)
    rewards, steps, avg_speeds = mappo.evaluation(env, len(seeds), is_train=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="PL", type=str)
    parser.add_argument("--run_id", default="mappo_run28", type=str) # runXX is performing the best on training data
    parser.add_argument("--model_name", default="priority_lane", type=str)
    parser.add_argument("--seed",
                        default=42, type=int,
                        help="Random seed")
    parser.add_argument("--option", default='train', choices=['train', 'test'], type=str)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--n_agents", default=1, type=int)
    parser.add_argument("--buffer_length", default=1024, type=int)
    parser.add_argument("--n_episodes", default=5000, type=int)
    parser.add_argument("--episode_duration", default=400, type=int)
    parser.add_argument("--action_step", default=5, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--steps_per_update", default=128, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--save_interval", default=30, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.01, type=float)

    config = parser.parse_args()

    # train or eval
    if config.option == 'train':
        train(config)
    else:
        evaluate(config)
