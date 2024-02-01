import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
# from tensorboardX import SummaryWriter

from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG

import numpy as np
import sys

from sumo_env import SUMOEnv
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import wandb
from argparse import ArgumentParser
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import time
import os
from tqdm import tqdm
import csv

mode = False
testFlag = True
USE_CUDA = False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = SUMOEnv(mode=mode,testFlag=testFlag)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)],mode)
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
    

def sample_agents(model, num_agents):
    curr_num_agents = model.nagents
    model.agents = np.random.choice(model.agents, size=num_agents)
    return model

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    curr_run = config.run_id + config.model_id
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    # os.makedirs(log_dir)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    NUM_AGENTS = 229
    env = make_parallel_env(config.env_id, 1, config.seed, config.discrete_action)

    # env.set_Testing(True)
    maddpg = MADDPG.init_from_save(run_dir)
    maddpg = sample_agents(maddpg, NUM_AGENTS)
    assert maddpg.nagents==env.envs[0].n

    t = 0
    scores = []    
    smoothed_total_reward = 0
    pid = os.getpid()
    # testResultFilePath = f"results/Run18_Density1_CAV20.csv" 
    testResultFilePath = f"results/Debugging.csv" 
    # testResultFilePath = f"results/MultiAgent_Test_{config.run_id}.csv"  
    with open(testResultFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        written_headers = False

        for seed in list(range(42,43)): # realizations for averaging - 47
            env.set_sumo_seed(seed)
            for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
                total_reward = 0
                print("Episodes %i-%i of %i" % (ep_i + 1,
                                                ep_i + 1 + config.n_rollout_threads,
                                                config.n_episodes))
                obs = env.reset(mode)
                step = 0
                maddpg.prep_rollouts(device='cpu')
                for et_i in range(config.episode_length):
                    step += 1
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                        requires_grad=False)
                                for i in range(maddpg.nagents)]
                  
                    torch_agent_actions = maddpg.step(torch_obs, explore=False)
                    # convert actions to numpy arrays
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                    # rearrange actions to be per environment
                    actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                    # print(actions)
                    next_obs, rewards, dones, infos = env.step(actions)
                    obs = next_obs
                    t += config.n_rollout_threads
                    total_reward += rewards[0]
                    if et_i%10==0:
                        headers, values = env.getTestStats()
                        if not written_headers:
                            writer.writerow(headers)
                            written_headers = True
                        writer.writerow(values)

                total_reward = total_reward/step
                # show reward
                smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
                scores.append(smoothed_total_reward)
        
        env.close()
      
    plt.plot(scores)
    plt.xlabel('episodes')
    plt.ylabel('ave rewards')
    plt.savefig('avgScore.jpg')
    plt.show()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="PL", type=str)
    parser.add_argument("--run_id", default="run18", type=str) # runXX is performing the best on training data
    parser.add_argument("--model_id", default="/model.pt", type=str)
    parser.add_argument("--model_name", default="priority_lane", type=str)
    parser.add_argument("--seed",
                        default=42, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=150, type=int)
    parser.add_argument("--steps_per_update", default=10, type=int)
    parser.add_argument("--batch_size",
                        default=64, type=int,
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

    run(config)
