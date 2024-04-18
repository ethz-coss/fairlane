import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

flow = 'Model37'
keys_to_plot = ['Avg_WaitingTime_All (5 mins)','avg_speed_all','Avg_WaitingTime_CAV (5 mins)', 'Avg_WaitingTime_RL+NPC (5 mins)', 'avg_speed_CAV','avg_speed_RL+NPC','Throughput','total_lane_change_number_all (5 mins)']
for key in keys_to_plot:
    fig, axes = plt.subplots(2,5,figsize=(30,10), dpi=300,sharey='row')
    # fig, axes = plt.subplots(2,5,figsize=(30,10), dpi=300)
    # axes = axes.flatten()

    CAV=[0,10,20,30,40,50,60,70,80,90]
    # CAV = [30,40,50]
    # CAV = [10,20,30]
    j = -1
    k = 0
    for cav in CAV:
        j +=1
        if j == 5:
            j = 0
            k = 1
        file_path = "C:/Users/rodubey/Downloads/result/results/MSN"
        # file_path = "D:/prioritylane/results/MSN"
        mapping = {
                'Exclusive CAV': f'{file_path}/Baseline1_{cav}_consolidated',
                'Model': f'{file_path}/Model_{cav}_consolidated', 
                'SOTA': f'{file_path}/SOTA_{cav}_consolidated',    
                'No Managed Lane': f'{file_path}/Baseline2_{cav}_consolidated',  
            }
        subtitle_to_plot = [f'CAV Penetration Rate = {cav}']
        data = []
        for i, (label, _filename) in enumerate(mapping.items()):
            filename = f'{_filename}.csv'
            df = pd.read_csv(filename)
            df['baseline'] = label
            # df['Seed'] = range(len(df))
            data.append(df)
        super_df = pd.concat(data).reset_index()
        # print(super_df.shape)
        # print(cav)
        sns.lineplot(ax=axes[k,j],data=super_df, hue='baseline', x="HDV", estimator='mean', y=key,style="baseline",markers=True).set(title=subtitle_to_plot[0])
    # plt.show()
    plt.savefig(f'C:/D/SUMO/PriorityLane/ResultPlots/{flow}/{flow}_{key}.jpg')
    # plt.savefig('rohit.jpg')



# import os
# import img2pdf

# flow = 2100
# folder_dir = 'D:/prioritylane/ResultPlots/2100/'

# with open(f'ResultPlots/{flow}.pdf', "wb") as f:
#     f.write(img2pdf.convert([i for i in os.listdir(folder_dir) if i.endswith(".jpg")]))
