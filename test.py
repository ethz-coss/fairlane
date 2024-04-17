import argparse
import torch
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from algorithms.maddpg import MADDPG

from sumo_env import SUMOEnv
from matplotlib import pyplot as plt
# import wandb
from tqdm import tqdm
import csv
from utils.common import make_parallel_env


GUI = False     
testFlag = True
USE_CUDA = False  # torch.cuda.is_available()
# SotaFlag = False
# ModelFlag = True
# Baseline1Flag = False
# Baseline2Flag = False

modelToRlDict = {}

folders = {
    'baseline1': 'Baseline1',
    'baseline2': 'Baseline2',
    'model': 'Model',
    'sota': 'SOTA'
}

def sample_agents(model, num_agents):
    if model.nagents!=num_agents:
        model.agents = np.random.choice(model.backupAgents, size=num_agents)
    return model

def dynamic_agents(model,agents):
    for agent in agents:
        if agent.id not in modelToRlDict:
            mod = np.random.choice(model.backupAgents, size=1)[0]
            modelToRlDict[agent.id] = mod


    for key in list(modelToRlDict.keys()):
        if key not in [agent.id for agent in agents]:
            del modelToRlDict[key]

    model.agents = np.array(list(modelToRlDict.values()))
    return model

def run(config):
    scenario_flag = config.scenario # can be model, sota, baseline1, baseline2
    model_dir = Path('./models') / config.env_id / config.model_name
    curr_run = config.run_id + config.model_id
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    # os.makedirs(log_dir)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    CAV=config.cav
    HDV=config.hdv
    NPC=100-(HDV+CAV)

    print(CAV,HDV,NPC)
    env = make_parallel_env(SUMOEnv, config.n_rollout_threads, config.seed, GUI, testFlag,
                            config.episode_duration, num_agents=config.n_agents, action_step=config.action_step,
                            cav_rate=CAV, hdv_rate=HDV, scenario_flag=scenario_flag)

    # env.set_Testing(True)
    maddpg = MADDPG.init_from_save(run_dir)
    maddpg.backupAgents = maddpg.agents
    maddpg = sample_agents(maddpg, config.n_agents)
    # maddpg = dynamic_agents(maddpg, env.envs[0].agents)

    # assert maddpg.nagents==env.envs[0].n
    # maddpg.nagents=env.envs[0].n
    t = 0
    scores = []    
    smoothed_total_reward = 0
    pid = os.getpid()
    # testResultFilePath = f"results/Run22_Density1_CAV20.csv" 

    # folder = config.scenario
    # folder = "Baseline2"
    folder = folders[scenario_flag]
    # folder = "SOTA"
    # folder = "dummy"

    os.makedirs(f'results/{config.network}/{folder}/', exist_ok=True)
    testResultFilePath = f"results/{config.network}/{folder}/{folder}_CAV{CAV}_HDV{HDV}_NPC{NPC}_test_stats.csv"
    # testResultFilePath = "dummy.csv"
    with open(testResultFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        written_headers = False

        for seed in list(range(3,13)): # realizations for averaging - 47
            # seed = 2
            # env.seed(seed)
            env.envs[0].set_sumo_seed(seed)
            for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
                total_reward = 0
                print("Episodes %i-%i of %i" % (ep_i + 1,
                                                ep_i + 1 + config.n_rollout_threads,
                                                config.n_episodes))
                obs = env.reset()
                step = 0
                maddpg.prep_rollouts(device='cpu')
                
                episode_length = int(config.episode_duration/(config.action_step + 1))
                for et_i in range(episode_length):
                    step += 1
                    number_of_agents = env.envs[0].n
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                        requires_grad=False)
                                for i in range(number_of_agents)]
                    
                    #assign models to RL agent
                    maddpg = dynamic_agents(maddpg, env.envs[0].agents)
                  
                    torch_agent_actions = maddpg.step(torch_obs, explore=False)
                    while len(torch_agent_actions)<number_of_agents:
                        torch_agent_actions.append(torch.zeros_like(torch_agent_actions[0]))
                    # convert actions to numpy arrays
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                    # rearrange actions to be per environment
                    actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                    # print(actions)
                    next_obs, rewards, dones, infos = env.step(actions)

                    obs = next_obs                    
                    t += config.n_rollout_threads
                    total_reward += rewards[0]
                    # if et_i%testStatAccumulation==0 and et_i>0:
                    if step%100==0:
                        headers, values = env.envs[0].getTestStats()
                        if step!=1:
                            if not written_headers:
                                writer.writerow(headers)
                                written_headers = True
                            writer.writerow(values)

                total_reward = total_reward/step
                # show reward
                smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
                scores.append(smoothed_total_reward)
               

                # if scenario_flag=='sota':   
                #     os.rename('sumo_configs/Test/edge_stats.xml',  f"results/{config.network}/{folder}/SOTA_CAV{CAV}_HDV{HDV}_NPC{NPC}_edge_stats_{seed}.xml")
                # elif scenario_flag=='model':                 
                #     os.rename('sumo_configs/Test/edge_stats.xml',  f"results/{config.network}/{folder}/Model_CAV{CAV}_HDV{HDV}_NPC{NPC}_edge_stats_{seed}.xml")
                # elif scenario_flag=='baseline1':                 
                #     os.rename('sumo_configs/Test/edge_stats.xml',  f"results/{config.network}/{folder}/Baseline1_CAV{CAV}_HDV{HDV}_NPC{NPC}_edge_stats_{seed}.xml")
                # elif scenario_flag=='baseline2':                 
                #     os.rename('sumo_configs/Test/edge_stats.xml',  f"results/{config.network}/{folder}/Baseline2_CAV{CAV}_HDV{HDV}_NPC{NPC}_edge_stats_{seed}.xml")

   
        env.close()
        
      
    plt.plot(scores)
    plt.xlabel('episodes')
    plt.ylabel('ave rewards')
    plt.savefig('avgScore.jpg')
    plt.show()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="PL", type=str)
    parser.add_argument("--run_id", default="run37", type=str) # runXX is performing the best on training data 
    parser.add_argument("--model_id", default="/model.pt", type=str)
    parser.add_argument("--model_name", default="priority_lane", type=str)
    parser.add_argument("--network", default="MSN", type=str)
    parser.add_argument("--scenario", default="baseline1", type=str, choices=folders.keys())
    parser.add_argument("--cav", default=20, type=int)
    parser.add_argument("--hdv", default=20, type=int)
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    
    parser.add_argument("--n_agents", default=200, type=int) #219
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_duration", default=3600, type=int) # 100 for warmup
    parser.add_argument("--action_step", default=2, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--steps_per_update", default=128, type=int)
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

    run(config)
