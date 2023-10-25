import numpy as np
import pandas as pd
import os
import csv
from csv import writer
import sys

if len(sys.argv) > 1:
    timesteps = int(sys.argv[1])
else:
    #default timesteps
    timesteps = 30

path = 'data_trajectories/FDR_weather_fixed'

data = {}

minsteps = 1000
maxsteps = -1000
stepcounter = 0


features = ["lat", "lon", "h", "tempr", "uwind", "vwind"]
flights_counter = 0

with open(f'data_trajectories/FDR-{timesteps}-timestep-trajectories6v2phases.csv', 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    #header = ['ID','CI', 'PL']

    #for i in features:
        #header.append(f"{i}")

    #writer.writerow(header)

    for subdir, dirs, files in os.walk(path):

        for file in files:
            
            file_list = file.split("_")
            flight_id = file_list[1]

            if len(file_list)!=7:
                continue

            CI = file_list[3]
            PL = file_list[5]

            df = pd.read_csv(os.path.join(subdir, file))

            if(np.isnan(np.array(df.iloc[:,8]).tolist()).any()):
                continue

            flights_counter += 1

            df = df[(df.phase != 'climbing_BELOW-FL100') & (df.phase != 'descending_BELOW-FL100')]

            df_cl = df[(df.phase == 'climbing_ABOVE-FL100')]
            df_cr = df[(df.phase == 'cruising')]
            df_ds = df[(df.phase == 'descending_ABOVE-FL100')]

            df_list = [df_cl, df_cr, df_ds]

            print(f"flight {flights_counter}")

            for df_i in df_list:

                data = []

                posix = np.array(df_i.iloc[:,3])

                min_posix = np.min(posix)
                max_posix = np.max(posix)
                
                interval = (max_posix - min_posix) / (timesteps/3)

                accumulative_interval = interval
             
                previous_index = 0
                counter = 0
                
                for i in range(int(timesteps/3)):

                    index = np.argmax(posix >= min_posix+interval*(i+1))
                    if index == 0 and previous_index != 0:
                        #print(index,lon,(min_lon+interval*(i+1)))
                        index = len(posix)-1

                    if previous_index == index:
                        data.append(np.asarray(df_i.iloc[index,1], dtype = np.float32))
                        data.append(np.asarray(df_i.iloc[index,2], dtype = np.float32))
                        data.append(np.asarray(df_i.iloc[index,4], dtype = np.float32))
                        data.append(np.asarray(df_i.iloc[index,6], dtype = np.float32))
                        data.append(np.asarray(df_i.iloc[index,7], dtype = np.float32))
                        data.append(np.asarray(df_i.iloc[index,8], dtype = np.float32))
                        counter += 1
                    else:
                        data.append(np.mean(np.asarray(df_i.iloc[previous_index:index,1].tolist(), dtype = np.float32)))
                        data.append(np.mean(np.asarray(df_i.iloc[previous_index:index,2].tolist(), dtype = np.float32)))
                        data.append(np.mean(np.asarray(df_i.iloc[previous_index:index,4].tolist(), dtype = np.float32)))
                        data.append(np.mean(np.asarray(df_i.iloc[previous_index:index,6].tolist(), dtype = np.float32)))
                        data.append(np.mean(np.asarray(df_i.iloc[previous_index:index,7].tolist(), dtype = np.float32)))
                        data.append(np.mean(np.asarray(df_i.iloc[previous_index:index,8].tolist(), dtype = np.float32)))
                        counter += 1

                    prev_data = [flight_id, CI, PL]
                    prev_data.extend(data)
                    writer.writerow(prev_data)
                    data = []
                    previous_index = index

print("--- Trajectory dataset has been created! ---")