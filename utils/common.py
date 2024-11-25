from __future__ import annotations
from typing import TYPE_CHECKING
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from gym.vector.async_vector_env import AsyncState, AlreadyPendingCallError
from gym.vector.utils import concatenate, write_to_shared_memory
import numpy as np
import sys

if TYPE_CHECKING:
    from sumo_env import SUMOEnv

from copy import deepcopy


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send((None, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                if all(done):
                    observation = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send(((None, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_check_observation_space":
                pipe.send((data == observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()

class MAAsyncVectorEnv(AsyncVectorEnv):
    def __init__(self, env_fns, num_agents=1, observation_space=None, action_space=None, copy=True):
        super().__init__(env_fns, observation_space=None, action_space=None, copy=True, worker=_worker_shared_memory)
        self._rewards = np.zeros((self.num_envs, num_agents,), dtype=float)
        self._dones = np.zeros((self.num_envs, num_agents,), dtype=np.bool_)
        # self._rewards = [None] * self.num_envs

    def reset_wait(self, *args, **kwargs):  # override
        obs = super().reset_wait(*args, **kwargs)
        transpose_idx = (1, 0, 2)
        return np.transpose(obs, transpose_idx)

    def step_wait(self):  # override
        obs, rews, dones, infos = super().step_wait()
        transpose_idx = (1, 0, 2)
        return np.transpose(obs, transpose_idx), rews, dones, infos

class MASyncVectorEnv(SyncVectorEnv):
    def __init__(self, env_fns, num_agents=1, observation_space=None, action_space=None, copy=True):
        super().__init__(env_fns, observation_space=None, action_space=None, copy=True)
        self._rewards = np.zeros((self.num_envs, num_agents,), dtype=float)
        self._dones = np.zeros((self.num_envs, num_agents,), dtype=np.bool_)

    def reset_wait(self, *args, **kwargs):  # override
        obs = super().reset_wait(*args, **kwargs)
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
                print("WARNING, NOT USING MULTIAGENT")
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
                          episode_duration=episode_duration, default_seed=seed, **kwargs)
            env.seed(seed + rank * 1000)
            # np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return MASyncVectorEnv([get_env_fn(0)], num_agents=num_agents)
    else:
        return MAAsyncVectorEnv([get_env_fn(i) for i in range(n_rollout_threads)], num_agents=num_agents)


VPH = 13416
def convertToFlows(cpn, hpn, scenario):
    vph = VPH
    cav_count = int((vph/100)*cpn)
    hdv_count = int((vph/100)*hpn)
    npc_count = int(vph - (cav_count+hdv_count))

    # if scenario=='baseline1' or scenario=='baseline2':
    #     n_agents = 1      
    #     npc_count = npc_count + hdv_count
    # else:
    #     n_agents = flowToRouteDict[hdv_count]

    # cav_rate = cav_count/3600
    # npc_rate = npc_count/3600
    # hdv_rate = hdv_count/3600
    if npc_count <= 0:
        npc_period = 0
        npc_count = 0
    # print(f"CAV_Rate={cav_count}")
    # print(f"NPC_Rate={npc_count}")
    # print(f"HDV_Rate={hdv_count}")
    # return cav_rate, npc_rate, hdv_rate
    return cav_count, npc_count, hdv_count

# def convertToFlows(cpn, hpn,scenario):
#     # flowToRouteDict = {0: 1, 210: 36, 420: 80, 630: 120, 840: 175, 1050: 228, 1260: 250, 1470: 289, 1680: 300, 1890: 319, 2100: 338}
#     flowToRouteDict = {0: 1, 210: 40, 420: 80, 630: 120, 840: 160, 1050: 200, 1260: 240, 1470: 280, 1680: 320, 1890: 360, 2100: 400}

#     vph = 2100

#     cav_count = (vph/100)*cpn
#     hdv_count = (vph/100)*hpn
#     npc_count = vph - (cav_count+hdv_count)

#     cav_agents = flowToRouteDict[cav_count]
#     hdv_agents = flowToRouteDict[hdv_count]
#     npc_agents = flowToRouteDict[npc_count]

#     # if scenario=='baseline1' or scenario=='baseline2':
#     #     n_agents = 1      
#     #     npc_count = npc_count + hdv_count
#     # else:
#     #     n_agents = flowToRouteDict[hdv_count]

#     # cav_rate = cav_count/3600
#     # npc_rate = npc_count/3600
#     # if npc_count <= 0:
#     #     npc_period = 0
#     #     npc_count = 0

#     return cav_agents, npc_agents, hdv_agents
