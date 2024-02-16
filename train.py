import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG

import numpy as np
import gym
from sumo_env import SUMOEnv
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import wandb
from argparse import ArgumentParser
import time
import os
from tqdm import tqdm
import csv
from copy import deepcopy
from gym.vector.utils import concatenate

class MASyncVectorEnv(gym.vector.SyncVectorEnv):
    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        super().__init__(env_fns, observation_space=None, action_space=None, copy=True)
        self._rewards = np.zeros((self.num_envs,self.envs[0].n,), dtype=float)
        self._dones = np.zeros((self.num_envs,self.envs[0].n,), dtype=np.bool_)

    def reset_wait(self): # override
        obs = super().reset_wait()
        transpose_idx = (1,0,2)
        return np.transpose(obs, transpose_idx)
    
    def step_wait(self): # override
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            try:
                if all(self._dones[i]):
                    observation = env.reset()
            except TypeError: # probably not multiagent?
                if self._dones[i]:
                    observation = env.reset()
            observations.append(observation)
            infos.append(info)
        self.observations = concatenate(
            observations, self.observations, self.single_observation_space
        )

        obs, rews, dones = (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones)
        )
        transpose_idx = (1,0,2)
        return np.transpose(obs, transpose_idx), rews, dones, infos
    
use_wandb = os.environ.get('WANDB_MODE', 'online') # can be online, offline, or disabled
wandb.init(
  project="prioritylane",
  tags=["MultiAgent", "RL"],
  mode=use_wandb
)

reward_type = "Global"
# reward_type = "Local"
mode = False
testFlag = False
USE_CUDA = False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action, num_agents=50,action_step=30):
    def get_env_fn(rank):
        def init_env():
            env = SUMOEnv(mode=mode,testFlag=testFlag, num_agents=num_agents,action_step=action_step,
                          episode_duration=config.episode_duration)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return MASyncVectorEnv([get_env_fn(0)])
    else:
        return MASyncVectorEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def runner(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    # logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action, num_agents=config.n_agents, action_step=config.action_step)
    # print(env.action_space)
    # print(env.observation_space)
    normalize_rewards = True
    # Log configs
    wandb.config.algorithm = 'MADDPG'
    wandb.config.lr = config.lr
    wandb.config.gamma = config.gamma
    wandb.config.batch_size = config.batch_size
    wandb.config.n_rl_agents = config.n_agents
    wandb.config.action_step = config.action_step
    wandb.config.normalize_rewards = normalize_rewards

    unwrapped_env = env.envs[0]
    assert unwrapped_env.n == config.n_agents
    
    maddpg = MADDPG.init_from_env(unwrapped_env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  gamma=config.gamma,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in unwrapped_env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in unwrapped_env.action_space])
    t = 0
    scores = []    
    smoothed_total_reward = 0
    pid = os.getpid()
    train_counter = 0
    for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        total_reward = 0
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset() # is an cast to array of (agents, rollouts, observations)
        step = 0
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()
        # obs = 
        # oo = np.hstack(obs)
        # obs = [i[np.newaxis,:] for i in obs]
        episode_length = int(config.episode_duration/config.action_step)
        
        for et_i in range(episode_length):
            step += 1
            
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                requires_grad=False)
                        for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if reward_type=="Global":
                total_reward += float(rewards[0][0])
            else:
                total = np.sum(rewards)
                temp_reward = total/50
                total_reward += temp_reward
            
            val_losses = []
            pol_losses = []
            # print(len(replay_buffer))
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    device = 'gpu'
                    maddpg.prep_training(device=device)
                else:
                    device = 'cpu'                    
                    maddpg.prep_training(device=device)
                for u_i in range(config.n_rollout_threads):
                    print("---------------Training----------------")
                    train_counter+=1
                    print(train_counter)
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                    to_gpu=USE_CUDA, norm_rews=normalize_rewards)
                        val_loss, pol_loss = maddpg.update(sample, a_i)
                        val_losses.append(val_loss)
                        pol_losses.append(pol_loss)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device=device)
        ep_rews = replay_buffer.get_average_rewards(
            episode_length * config.n_rollout_threads)
        # for a_i, a_ep_rew in enumerate(ep_rews):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        
        
        total_reward = total_reward/step
        if ep_i == 0:
            smoothed_total_reward = total_reward
        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        scores.append(smoothed_total_reward)
        
        wandb.log({'# Episodes': ep_i, 
                "Average Smooth Reward": smoothed_total_reward,
                "Average Raw Reward": total_reward})
        
        
    

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="PL", type=str)
    parser.add_argument("--model_name", default="priority_lane", type=str)
    parser.add_argument("--seed",
                        default=42, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--n_agents", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1000, type=int)
    parser.add_argument("--episode_duration", default=400, type=int)
    parser.add_argument("--action_step", default=2, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--steps_per_update", default=40, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=30, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    runner(config)
