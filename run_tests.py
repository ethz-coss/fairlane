import os
import signal
from subprocess import Popen
from shutil import which
import numpy as np
from test import folders
import itertools
from utils.common import convertToFlows

child_processes = []

def sigterm_handler(signum, frame):
    print("Parent process received termination signal. Killing child processes...")
    for process in child_processes:
        if process.poll() is None:  # Check if the child process is still running
            process.terminate()
    exit(0)

# Register SIGTERM (termination) signal handler
signal.signal(signal.SIGTERM, sigterm_handler)


IS_SLURM = which('sbatch') is not None
DEFAULT_SEED = 42

NUM_SEEDS = 20
seeds = range(DEFAULT_SEED, DEFAULT_SEED+NUM_SEEDS)

def flush_commands(commands):
    if not IS_SLURM:
        pycalls = " &\n".join(commands)
        pycalls += "\n wait"
        proc = Popen(f'{pycalls}', shell=True) #, preexec_fn=os.setsid
        child_processes.append(proc)
        proc.communicate()
        return
    pycalls = "\n".join(commands)
    proc = Popen(f"sbatch -n 1 --mem-per-cpu=4G --time=10:00:00 -o job.out -e job.err  --wrap '{pycalls}'", shell=True)
    child_processes.append(proc)


# cav_rates = np.arange(0,101,10)
# # hdv_rates = [40]
# hdv_rates = np.arange(0,101,10)


if __name__=="__main__":
    NCPU = 16
    calls = []

    ## OVERRIDES
    # scenarios = ['baseline1', 'baseline2','model','sota', 'mappo']
    scenarios = ['baseline1', 'baseline2','model','sota']
    # scenarios = ['baseline2',"baseline1"]
    # scenarios = ['mappo']
    cav_rates = np.arange(0,91,10)
    hdv_rates = np.arange(10,91,10)
    compliances = np.arange(0, 1.01, 0.1)
    cav_rates = [10,20]
    hdv_rates = [20,30,40]
    for scenario in scenarios:
        for cav, hdv, compliance in itertools.product(cav_rates, hdv_rates, compliances):
            if (cav+hdv)<=100:
                _, _, n_agents = convertToFlows(cav,hdv,scenario)   
                # n_agents = 100         
                # if scenario == 'baseline1' or scenario == 'baseline2':
                #     n_agents = 1
                if scenario=='mappo':
                    calls.append(f"python run_mappo.py --run_id mappo_run_45_1 --waiting_time_memory 3600 --n_episodes 1 --option test --episode_duration 3600 --cav {cav} --hdv {hdv}")
                else:
                    calls.append(f"python test.py --run_id maddpg_run_45_2 --waiting_time_memory 3600 --scenario {scenario} --cav {cav} --hdv {hdv} --n_agents {n_agents} --compliance {compliance}")
                # proc = Popen(f"python test.py --run_id run16 --scenario {scenario} --cav {cav} --hdv {hdv} --n_agents {n_agents}", shell=True)
                # proc.wait()
            if len(calls)==NCPU:
                flush_commands(calls)
                calls = []
    if calls:
        flush_commands(calls)
