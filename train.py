import argparse
import torch
import os
import numpy as np
from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG
from sumo_env import SUMOEnv
import wandb
import os
from tqdm import tqdm
from utils.common import make_parallel_env
    

reward_type = "Global"
# reward_type = "Local"
# reward_type = "Individual"
GUI = False
testFlag = False
USE_CUDA = True and torch.cuda.is_available()

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

def logit2ohe(x):
    return 1*(x==x.max(axis=1, keepdims=True))
    
def runner(config, run_dir, wandb_run):
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    # logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    CAV=config.cav
    HDV=config.hdv
    NPC=100-(HDV+CAV)

    print(CAV,HDV,NPC)

    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # env = make_parallel_env(SUMOEnv, config.n_rollout_threads, config.seed, GUI, testFlag,
    #                         config.episode_duration, num_agents=config.n_agents, action_step=config.action_step)
    
    env = make_parallel_env(SUMOEnv, config.n_rollout_threads, config.seed, GUI, testFlag, 
                            config.episode_duration, num_agents=config.n_agents, action_step=config.action_step,
                            cav_rate=CAV, hdv_rate=HDV, scenario_flag='model')
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

    unwrapped_env = env.env_fns[0]()
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
    # maddpg.backupAgents = maddpg.agents
    # maddpg = sample_agents(maddpg, config.n_agents)

    t = 0
    scores = []    
    smoothed_total_reward = 0
    pid = os.getpid()
    train_counter = 0
    env.seed(config.seed)
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
            number_of_agents = config.n_agents
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                requires_grad=False)
                        for i in range(number_of_agents)]
            
            #assign models to RL agent
            # maddpg = dynamic_agents(maddpg, env.envs[0].agents)
            
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [logit2ohe(np.array([ac[i] for ac in agent_actions])) for i in range(config.n_rollout_threads)]
            # actions = np.array(actions)
            next_obs, rewards, dones, infos = env.step(actions)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if reward_type=="Global":
                total_reward += float(rewards[0][0])
            elif reward_type=="Individual":
                temp_reward = np.mean(rewards)
                total_reward += temp_reward
            else:
                temp_reward = np.mean(rewards)
                total_reward += temp_reward
            
            val_losses = []
            pol_losses = []
            # print(len(replay_buffer))
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                print(f"---------------Training----------------:{et_i}")
                if USE_CUDA:
                    device = 'gpu'
                else:
                    device = 'cpu'                    
                maddpg.prep_training(device=device)
                train_counter+=1
                for a_i in range(maddpg.nagents):
                    sample = replay_buffer.sample(config.batch_size,
                                                to_gpu=USE_CUDA, norm_rews=normalize_rewards)
                    val_loss, pol_loss = maddpg.update(sample, a_i)
                    val_losses.append(val_loss)
                    pol_losses.append(pol_loss)
                maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
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
    parser.add_argument("--cav", default=20, type=int)
    parser.add_argument("--hdv", default=50, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--n_agents", default=50, type=int)
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--n_episodes", default=5000, type=int)
    parser.add_argument("--episode_duration", default=400, type=int)
    parser.add_argument("--action_step", default=3, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--steps_per_update", default=40, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=1000, type=int)
    parser.add_argument("--init_noise_scale", default=1.0, type=float)
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
    parser.add_argument("--discrete_action", default=True)

    config = parser.parse_args()

    model_dir = Path('./models') / config.env_id / config.model_name
    base_run_name = f'maddpg_run_{config.seed}'
    if not model_dir.exists():
        curr_run = f'{base_run_name}_1'
    else:
        exst_run_nums = [int(str(folder.name).rsplit('_', 1)[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith(base_run_name)]
        if len(exst_run_nums) == 0:
            curr_run =  f'{base_run_name}_1'
        else:
            curr_run =  f'{base_run_name}_{(max(exst_run_nums) + 1)}'
    run_dir = model_dir / curr_run

    use_wandb = os.environ.get('WANDB_MODE', 'online') # can be online, offline, or disabled
    wandb_run = wandb.init(
            project="prioritylane",
            tags=["MultiAgent", "RL"],
            mode=use_wandb, 
            name=curr_run
        )
    wandb.define_metric("*", step_metric="# Episodes")


    runner(config, run_dir, wandb_run)

