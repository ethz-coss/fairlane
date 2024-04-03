import os
from shutil import which
import numpy as np
from test import folders
import itertools
from utils.common import convertToFlows
IS_SLURM = which('sbatch') is not None
DEFAULT_SEED = 42

NUM_SEEDS = 10
seeds = range(DEFAULT_SEED, DEFAULT_SEED+NUM_SEEDS)

def flush_commands(commands):
    if not IS_SLURM:
        pycalls = " &\n".join(commands)
        pycalls += "\n wait"
        os.system(f'{pycalls}')
        return
    pycalls = "\n".join(commands)
    os.system(f"sbatch -n 1 --mem-per-cpu=4G --time=20:00:00 -o job.out -e job.err  --wrap '{pycalls}'")

if __name__=="__main__":
    NCPU = 4
    calls = []

    ## OVERRIDES
    scenarios = ['baseline1', 'baseline2', "model", "sota"]
    cav_rates = [10, 30, 50]
    hdv_rates = [25, 50, 75, 100]

    for scenario in scenarios:
        for cav, hdv in itertools.product(cav_rates, hdv_rates):
            _, _, n_agents = convertToFlows(cav,hdv)
            for seed in seeds:
                calls.append(f"python test.py --run_id run16 --scenario {scenario} --seed {seed} --cav {cav} --hdv {hdv} --n_agents {n_agents}")

                if len(calls)==NCPU:
                    flush_commands(calls)
                    calls = []
    if calls:
        flush_commands(calls)
