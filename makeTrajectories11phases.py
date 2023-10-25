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

path = 'data_trajectories/FDR_v6/v6'

data = {}
counter = 0

minsteps = 1000
maxsteps = -1000
stepcounter = 0


features = ["lat", "lon", "h", "tempr", "press", "wn", "we", "ws", "wx", "vdot", "hdot"]

flights_counter = 0

with open(f'data_trajectories/FDR-{timesteps}-timestep-trajectories11v2phases.csv', 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

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

            df = df[df['h[ft]'] > 10000]

            split1 = 'CRUISE'
            split2 = 'DES'

            # Find the index of the first occurrence of the split string in column 'A'
            split_index = (df['Phase'].str.startswith(split1)).idxmax()

            # Split the DataFrame into two separate DataFrames based on the split index
            df_cl = df.loc[:split_index]
            df_cr_ds = df.loc[split_index+1:]

            split_index = (df_cr_ds['Phase'].str.startswith(split2)).idxmax()

            df_cr = df.loc[:split_index]
            df_ds = df.loc[split_index+1:]

            print(f"flight {flights_counter}")

            posix = np.array(df.iloc[:,3])

            min_posix = np.min(posix)
            max_posix = np.max(posix)
            
            interval = (max_posix - min_posix) / timesteps

            accumulative_interval = interval
         
            previous_index = 0
            data = []
            counter = 0
            
            for i in range(timesteps):

                index = np.argmax(posix >= min_posix+interval*(i+1))
                if index == 0 and previous_index != 0:
                    #print(index,lon,(min_lon+interval*(i+1)))
                    index = len(posix)-1

                if previous_index == index:
                    
                    data.append(np.asarray(df.iloc[index,30], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,31], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,6], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,21], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,22], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,23], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,24], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,25], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,26], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,70], dtype = np.float32))
                    data.append(np.asarray(df.iloc[index,72], dtype = np.float32))
                    counter += 1
                else:
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,30].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,31].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,6].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,21].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,22].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,23].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,24].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,25].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,26].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,70].tolist(), dtype = np.float32)))
                    data.append(np.mean(np.asarray(df.iloc[previous_index:index,72].tolist(), dtype = np.float32)))
                    counter += 1

                prev_data = [flight_id, CI, PL]
                prev_data.extend(data)
                writer.writerow(prev_data)
                data = []
                previous_index = index

print("--- Trajectory dataset has been created! ---")