import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

flow = 'Model37'
keys_to_plot = ['Average Waiting Time (All)','Average Speed (All)','Average Waiting Time (CAV)','Average Waiting Time (HDV & NPC)','Average Speed (CAV)','Average Speed (HDV & NPC)','Throughput','Average Lane Change Number (All)','Average Lane Change Number (HDV)']
for key in keys_to_plot:
    fig, axes = plt.subplots(3,3,figsize=(10,10), dpi=300,sharey='row')
    # fig, axes = plt.subplots(2,5,figsize=(30,10), dpi=300)
    # axes = axes.flatten()

    CAV=[10,20,30,40,50,60,70,80,90]
    # CAV = [30,40,50]
    # CAV = [10,20,30]
    # j = -1
    # k = 0
    for cav, ax in zip(CAV, axes.flatten()):
        # j +=1
        # if j == 5:
        #     j = 0
        #     k = 1
        file_path = "results/MSN"
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
        legend_flag = cav==90 # place legend only on last box
        sns.lineplot(ax=ax,data=super_df, hue='baseline', x="HDV", estimator='median', errorbar=('pi', 50), y=key,style="baseline",markers=True, legend=legend_flag).set(title=subtitle_to_plot[0],
                                                                                                                                            xlabel=None,
                                                                                                                                            ylabel=None)
    # plt.show()
        
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))
    fig.supylabel(key, ha='right')
    fig.supxlabel('HDV')
    fig.tight_layout()
    fig.savefig(f'ResultPlots/{flow}/{key}.jpg', bbox_inches='tight')
    # plt.savefig('rohit.jpg')


# import os
# import img2pdf

# flow = 2100
# folder_dir = 'D:/prioritylane/ResultPlots/2100/'

# with open(f'ResultPlots/{flow}.pdf', "wb") as f:
#     f.write(img2pdf.convert([i for i in os.listdir(folder_dir) if i.endswith(".jpg")]))
