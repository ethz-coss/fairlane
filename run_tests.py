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

NUM_SEEDS = 10
seeds = range(DEFAULT_SEED, DEFAULT_SEED+NUM_SEEDS)

def flush_commands(commands):
    if not IS_SLURM:
        pycalls = " &\n".join(commands)
        pycalls += "\n wait"
        proc = Popen(f'{pycalls}', shell=True, preexec_fn=os.setsid)
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
    NCPU = 1
    calls = []

    ## OVERRIDES
    scenarios = ['baseline1', 'baseline2','model','sota']
    # scenarios = ["model","sota","baseline1"]
    # scenarios = ['sota']
    cav_rates = np.arange(0,101,10)
    hdv_rates = np.arange(0,101,10)

   
    for scenario in scenarios:
        for cav, hdv in itertools.product(cav_rates, hdv_rates):
            if (cav+hdv)<=100:
                _, _, n_agents = convertToFlows(cav,hdv,scenario)   
                # n_agents = 100         
                # if scenario == 'baseline1' or scenario == 'baseline2':
                #     n_agents = 1
                calls.append(f"python test.py --run_id run37 --scenario {scenario} --cav {cav} --hdv {hdv} --n_agents {n_agents}")
                # proc = Popen(f"python test.py --run_id run16 --scenario {scenario} --cav {cav} --hdv {hdv} --n_agents {n_agents}", shell=True)
                # proc.wait()
            if len(calls)==NCPU:
                flush_commands(calls)
                calls = []
    if calls:
        flush_commands(calls)
