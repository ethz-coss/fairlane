from __future__ import annotations
from typing import TYPE_CHECKING
from gym.vector import SyncVectorEnv
from gym.vector.utils import concatenate
import numpy as np

if TYPE_CHECKING:
    from sumo_env import SUMOEnv

from copy import deepcopy


class MASyncVectorEnv(SyncVectorEnv):
    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        super().__init__(env_fns, observation_space=None, action_space=None, copy=True)
        self._rewards = [np.zeros((self.num_envs, self.envs[0].n,), dtype=float)]
        self._dones = np.zeros(
            (self.num_envs, self.envs[0].n,), dtype=np.bool_)
        # self._rewards = [None] * self.num_envs
        # self._dones = [None] * self.num_envs

    def reset_wait(self):  # override
        obs = super().reset_wait()
        transpose_idx = (1, 0, 2)
        return np.transpose(obs, transpose_idx)

    def step_wait(self):  # override
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            try:
                if all(self._dones[i]):
                    observation = env.reset()
            except TypeError:  # probably not multiagent?
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
        transpose_idx = (1, 0, 2)
        return np.transpose(obs, transpose_idx), rews, dones, infos


def make_parallel_env(sumoenv: SUMOEnv, n_rollout_threads, seed, mode, testFlag, episode_duration, num_agents=50, action_step=30, **kwargs):
    def get_env_fn(rank):
        def init_env():
            env = sumoenv(mode=mode, testFlag=testFlag, num_agents=num_agents, action_step=action_step,
                          episode_duration=episode_duration, **kwargs)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return MASyncVectorEnv([get_env_fn(0)])
    else:
        return MASyncVectorEnv([get_env_fn(i) for i in range(n_rollout_threads)])


# def convertToFlows(cpn, hpn,scenario):
#     flowToRouteDict = {0: 1, 210: 36, 420: 80, 630: 120, 840: 175, 1050: 228, 1260: 250, 1470: 289, 1680: 300, 1890: 319, 2100: 338}
#     vph = 2400 #2400 seems to be in out favor, followed by 2100

#     cav_count = (vph/100)*cpn
#     hdv_count = (vph/100)*hpn
#     npc_count = vph - (cav_count+hdv_count)

#     # if scenario=='baseline1' or scenario=='baseline2':
#     #     n_agents = 1      
#     #     npc_count = npc_count + hdv_count
#     # else:
#     #     n_agents = flowToRouteDict[hdv_count]

#     cav_rate = cav_count/3600
#     npc_rate = npc_count/3600
#     hdv_rate = hdv_count/3600
#     if npc_count <= 0:
#         npc_period = 0
#         npc_count = 0
#     print(f"CAV_Rate={cav_count}")
#     print(f"NPC_Rate={npc_count}")
#     print(f"HDV_Rate={hdv_count}")
#     return cav_rate, npc_rate, hdv_rate
#     # return cav_count, npc_count, hdv_count

def convertToFlows(cpn, hpn,scenario):
    # flowToRouteDict = {0: 1, 210: 36, 420: 80, 630: 120, 840: 175, 1050: 228, 1260: 250, 1470: 289, 1680: 300, 1890: 319, 2100: 338}
    flowToRouteDict = {0: 1, 210: 40, 420: 80, 630: 120, 840: 160, 1050: 200, 1260: 240, 1470: 280, 1680: 320, 1890: 360, 2100: 400}

    vph = 2100

    cav_count = (vph/100)*cpn
    hdv_count = (vph/100)*hpn
    npc_count = vph - (cav_count+hdv_count)

    cav_agents = flowToRouteDict[cav_count]
    hdv_agents = flowToRouteDict[hdv_count]
    npc_agents = flowToRouteDict[npc_count]

    # if scenario=='baseline1' or scenario=='baseline2':
    #     n_agents = 1      
    #     npc_count = npc_count + hdv_count
    # else:
    #     n_agents = flowToRouteDict[hdv_count]

    # cav_rate = cav_count/3600
    # npc_rate = npc_count/3600
    # if npc_count <= 0:
    #     npc_period = 0
    #     npc_count = 0

    return cav_agents, npc_agents, hdv_agents
