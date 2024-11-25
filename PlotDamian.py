import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
import sumolib
import glob
from run_tests import cav_rates, hdv_rates

TARGET_EDGES = ['-15','-23','-5','-3']
NETWORK = 'MSN'
models = ['Model','SOTA','Baseline1','Baseline2']
seeds = list(range(3,4))
model = 'run14' # can be baseline or SOTA


# attr_list = ['id', 'traveltime','waitingTime', 'timeLoss', 'speed', 'left', 'teleported']
attr_list = ['id','traveltime']

def parse_edgedata(models, seeds):
    trip_data = []
    
    for cav in cav_rates:
        for hdv in hdv_rates:
            for model, seed in itertools.product(models, seeds):
                npc = 100-cav-hdv
                if npc<0:
                    continue
            # for scale, experiment, seed in itertools.product(scales, experiments, seeds): 
                pattern = f"results/{NETWORK}/{model}/{model}_CAV{cav}_HDV{hdv}_NPC{npc}_edge_stats_{seed}.xml"
                
                # for filename in glob.glob(pattern):
                try:
                    parser = sumolib.xml.parse_fast_nested(pattern, 'interval', ['end', 'id'], 'edge', attr_list)
                except FileNotFoundError:
                    print('File not Found: ', pattern)
                except Exception as e:
                    raise e
                else:
                    for interval, edge in parser:
                    
                        traveltime = float(edge.traveltime) 
                        time = float(interval.end)
                        edge_id = edge.id
                        veh_type = interval.id.rsplit('_')[1]
                        
                        data = {'time': time,
                                'traveltime':traveltime,
                                'edge_id': edge_id,
                                'model': model,
                                'seed': seed,
                                'veh_type': veh_type,
                                'cav': cav,
                                'hdv': hdv
                                }
                        if veh_type == 'cav':             
                            trip_data.append(data)
            #     continue

    return trip_data

run_data = parse_edgedata(models, seeds)

_dataframe = pd.DataFrame(run_data)

STARTTIME = 300
ENDTIME = 3900
filtered_df = _dataframe.query('edge_id in @TARGET_EDGES and time<@ENDTIME and time>@STARTTIME')

df = filtered_df.melt(id_vars=['time', 'model', 'seed', 'veh_type', 'cav', 'hdv'], var_name='stats')

# df = df.groupby(['model', 'stats', 'veh_type', 'cav', 'hdv', 'seed']).mean()['value'].reset_index()
print(df.head())
g = sns.FacetGrid(df, row='cav', hue='model', col='veh_type', sharey=False)
g.map_dataframe(sns.lineplot, x='hdv', y='value')
g.fig.axes[0].invert_yaxis()
g.add_legend()
g.savefig("all_plot.pdf")
