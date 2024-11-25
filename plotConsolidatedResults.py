import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import csv
import itertools


		
# folder = ['SOTA']
folder = ['Baseline1','Model','SOTA','Baseline2']
CAV=[0,10,20,30,40,50,60,70,80,90,100]
# CAV = [30,40,50]
HDV=[0,10,20,30,40,50,60,70,80,90,100]
# HDV = [50]

for cav in CAV:
    for fold in folder:
        file_path = f"results/MSN/{fold}_{cav}_consolidated.csv"
        header = [ 'Avg_WaitingTime_All (5 mins)','avg_speed_all','Avg_WaitingTime_CAV (5 mins)', 'Avg_WaitingTime_RL+NPC (5 mins)', 'avg_speed_CAV','avg_speed_RL+NPC','Throughput','total_lane_change_number_all (5 mins)','total_lane_change_number_RL (5 mins)','total_collision (5 mins)','CAV','HDV','NPC','Seed']
        data = []
        with open('dummy.csv', 'wt', newline ='') as file:    
            # writer = csv.writer(file, delimiter=',')
            # writer.writerow(i for i in header)
            for hdv in HDV:
                if 100-(hdv+cav) >=0:
                    filename = f"results/MSN/{fold}/{fold}_CAV{cav}_HDV{hdv}_NPC{100-(hdv+cav)}_test_stats.csv"
                    if os.path.isfile(filename):
                        if os.stat(filename).st_size > 0:             # print(filename)
                            df = pd.read_csv(filename)             
                            seed = [3,4]
                            # for s in seed:
                            #     Avg_WaitingTime_ALL = df.loc[df['Seed'] == s]['Avg_WaitingTime_All (5 mins)'].mean()
                            #     avg_speed_all = df.loc[df['Seed'] == s]['avg_speed_all'].mean()
                            #     Avg_WaitingTime_CAV = df.loc[df['Seed'] == s]['Avg_WaitingTime_CAV (5 mins)'].mean()
                            #     Avg_WaitingTime_RL = df.loc[df['Seed'] == s]['Avg_WaitingTime_RL+NPC (5 mins)'].mean()
                            #     avg_speed_CAV = df.loc[df['Seed'] == s]['avg_speed_CAV'].mean()
                            #     avg_speed_RL = df.loc[df['Seed'] == s]['avg_speed_RL+NPC'].mean()
                            #     Throughput = df.loc[df['Seed'] == s]['Throughput'].mean()
                            #     total_lane_change_number_all = df.loc[df['Seed'] == s]['total_lane_change_number_all (5 mins)'].mean()
                            #     total_lane_change_number_RL = df.loc[df['Seed'] == s]['total_lane_change_number_RL (5 mins)'].mean()
                            #     total_collision = df.loc[(df['Seed'] == s)]['total_collision (5 mins)'].mean()

                            #     row_list=[Avg_WaitingTime_ALL,avg_speed_all,Avg_WaitingTime_CAV,Avg_WaitingTime_RL,avg_speed_CAV,avg_speed_RL,Throughput,total_lane_change_number_all,total_lane_change_number_RL,total_collision,cav,hdv,100-hdv+cav,s]          
                                
                            #     writer.writerow(row_list)
                            averaged = df.groupby('Seed').mean().reset_index()
                            if len(df.Seed.unique())<10:
                                print(len(df.Seed.unique()), filename)
                            averaged['CAV'] = cav
                            averaged['HDV'] = hdv
                            averaged['NPC'] = 100-hdv
                            data.append(averaged)
                        else:
                             print(filename, "--is not written properly")
                    else:
                        print(filename, "--is missing")
        temp = pd.concat(data).reset_index()
        temp = temp.drop(['index'],axis=1)
        temp.to_csv(file_path, index=False)

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import os
# import csv
# import itertools


		
# folder = ['Baseline1','Baseline2','Model','SOTA']
# # folder = ['Baseline1','Baseline2','Model']
# # CAV=[0,10,20,30,40,50,60,70,80,90,100]
# CAV = [20]
# HDV=[10,20,30,40,50,60,70,80,90,100]
# # HDV = [40]

# epsteps = np.arange(300,3900,300)
# for cav in CAV:
#     for fold in folder:
#         file_path = f"D:/prioritylane/results/MSN/{fold}_{cav}_consolidated.csv"
#         header = [ 'Avg_WaitingTime_CAV (5 mins)', 'Avg_WaitingTime_RL+NPC (5 mins)', 'avg_speed_CAV','avg_speed_RL+NPC','Throughput','total_lane_change_number_all (5 mins)','total_lane_change_number_RL (5 mins)','total_collision (5 mins)','CAV','HDV','NPC','Seed']

#         with open(file_path, 'wt', newline ='') as file:    
#             writer = csv.writer(file, delimiter=',')
#             writer.writerow(i for i in header)
#             for hdv in HDV:
#                 if 100-(hdv+cav) >=0:
#                     filename = f"D:/prioritylane/results/MSN/{fold}/{fold}_CAV{cav}_HDV{hdv}_NPC{100-(hdv+cav)}_test_stats.csv"
#                     print(filename)
#                     df = pd.read_csv(filename)  
#                     seed = [3]
#                     for s, epstep in itertools.product(seed, epsteps):
#                         Avg_WaitingTime_CAV = df.loc[(df['Seed'] == s) & (df['Episode_Step']==epstep)]['Avg_WaitingTime_CAV (5 mins)'].mean()
#                         Avg_WaitingTime_RL = df.loc[(df['Seed'] == s) & (df['Episode_Step']==epstep)]['Avg_WaitingTime_RL+NPC (5 mins)'].mean()
#                         avg_speed_CAV = df.loc[(df['Seed'] == s) & (df['Episode_Step']==epstep)]['avg_speed_CAV'].mean()
#                         avg_speed_RL = df.loc[(df['Seed'] == s) & (df['Episode_Step']==epstep)]['avg_speed_RL+NPC'].mean()
#                         Throughput = df.loc[(df['Seed'] == s) & (df['Episode_Step']==epstep)]['Throughput'].mean()
#                         total_lane_change_number_all = df.loc[(df['Seed'] == s) & (df['Episode_Step']==epstep)]['total_lane_change_number_all (5 mins)'].mean()
#                         total_lane_change_number_RL = df.loc[(df['Seed'] == s) & (df['Episode_Step']==epstep)]['total_lane_change_number_RL (5 mins)'].mean()
#                         total_collision = df.loc[(df['Seed'] == s) & (df['Episode_Step']==epstep)]['total_collision (5 mins)'].mean()

#                         row_list=[Avg_WaitingTime_CAV,Avg_WaitingTime_RL,avg_speed_CAV,avg_speed_RL,Throughput,total_lane_change_number_all,total_lane_change_number_RL,total_collision,cav,hdv,100-hdv+cav,s]          
                        
#                         writer.writerow(row_list)


