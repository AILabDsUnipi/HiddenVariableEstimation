import pandas as pd 
import os
import csv
import sys
import platform
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import mean_absolute_error as mae
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers import Dense,Lambda,TimeDistributed, Dropout, Bidirectional
from keras.layers import concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from tensorflow.python.framework.ops import disable_eager_execution
tf.config.set_visible_devices([],'GPU')

disable_eager_execution()

def make_directory(directory_path):
    try:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    except FileExistsError:
        print(f"Error: Directory '{directory_path}' already exists.")
        raise

#Creates the LSTM architecture with seperated output networks.
def create_lstm_seperated(input_dim, timesteps, latent_dim, dense_layer, learning_rate):

    inp = Input((timesteps, input_dim))
    bidirectional = Bidirectional(LSTM(latent_dim, return_sequences=True, activation = 'relu', name='Bidirectional-Layer'))(inp) 
    latent_ci = LSTM(latent_dim, activation='relu', name = 'LSTM-layer-ci')(bidirectional)
    latent_pl = LSTM(latent_dim, activation='relu', name = 'LSTM-layer-pl')(bidirectional)
    #ci_out = Dense(1, activation='relu')(x)
    #pl_out = Dense(1, activation='relu')(x)
    ci_dense = Dense(dense_layer, activation = 'relu', name = 'CI-dense-layer')(latent_ci)
    pl_dense = Dense(dense_layer, activation = 'relu', name = 'PL-dense-layer')(latent_pl)

    ci_out = Dense(1, activation='sigmoid', name = 'CI-out')(ci_dense)
    pl_out = Dense(1, activation='sigmoid', name = 'PL-out')(pl_dense)

    model_lstm = Model(inp, [ci_out, pl_out])
    #model_lstm = Model(inp, out)
    model_lstm.compile(loss='mse', optimizer=Adam(learning_rate = learning_rate), metrics=['mse'])

    return model_lstm

#Creates the LSTM architecture that outputs from a single network
def create_lstm(input_dim, timesteps, latent_dim, dense_layer, learning_rate):

    inp = Input((timesteps, input_dim))
    bidirectional = Bidirectional(LSTM(latent_dim, return_sequences=True, activation = 'relu', name='Bidirectional-Layer'))(inp) 
    latent = LSTM(latent_dim, activation='relu', name = 'LSTM-layer')(bidirectional)

    dense = Dense(dense_layer, activation = 'relu', name = 'PL-dense-layer')(latent)

    out = Dense(2, activation='sigmoid', name = 'out')(dense)

    model_lstm = Model(inp, out)

    model_lstm.compile(loss='mse', optimizer=Adam(learning_rate = learning_rate), metrics=['mse'])

    return model_lstm

#The function that trains the LSTM
def train_model(lstm, x_train, y_train, epochs,batch_size, x_val, y_val):

    lstm.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,shuffle=(True), validation_data=(x_val, y_val))

def main():
          
    #initialization of model parameters
    #Random number used to give an id to the experiment
    experiment = random.randint(0, 1000)

    #Pipeline 1 or 2 (11 or 6 features)
    pipeline = int(sys.argv[1])

    #Learning Epochs
    epochs = int(sys.argv[2])

    #Batch Size
    batch_size = int(sys.argv[3])

    #Number of neurons in the dense layer(s)
    dense_layer = int(sys.argv[4])

    #Number of neurons in the intermediate latent layer
    latent_dim = int(sys.argv[5])

    #Timesteps of the Aerotrajectories
    timesteps = int(sys.argv[6])

    #Learning Rate
    learning_rate = float(sys.argv[7])

    #Choose between the two different LSTM architectures
    seperated_output = str(sys.argv[8])

    if pipeline == 1:
        features = 11
    else:
        features = 6
    
    #You have the option to add an extra arbitrary augment that will
    #be considered as if you want to include phase information into
    #the data

    if seperated_output == 'Yes':
        seperated_output = True
    else:
        seperated_output = False

    #Change this internally if you need other scaling method. Other scaling methods
    #not implemented
    scale = 'MinMax'

    #Random seed for reproducibility
    seed = 42

    header = ['Experiment', 'Pipeline', 'Timesteps', 'Epochs', 'batch_size', 'LSTM1', 'LSTM2', 'Dense', 'learning_rate', 'scale', 'Seperate Output', 'CI MAE TRAIN', 'CI MAE STD TRAIN', 'PL MAE TRAIN', 'PL MAE STD TRAIN', 'CI MAE TEST', 'CI MAE STD TEST', 'PL MAE TEST', 'PL MAE STD TEST', 'CI TRAIN MIN', 'CI TRAIN MAX', 'CI TRAIN IQR', 'PL TRAIN MIN', 'PL TRAIN MAX', 'PL TRAIN IQR', 'CI TEST MIN', 'CI TEST MAX', 'CI TEST IQR', 'PL TEST MIN', 'PL TEST MAX', 'PL TEST IQR']

    #Directory that results will be saved.
    make_directory(f'lstm{pipeline}_{experiment}')


    #Loadings files that contain Train and Test IDs
    train_file = open('data_trajectories/train_ids.txt')
    test_file = open('data_trajectories/test_ids.txt')

    #Making the respective Dataframes
    if pipeline == 2:
        df = pd.read_csv(f"data_trajectories/FDR-{timesteps}-timestep-trajectories6v2phases.csv",dtype='float32',header=None)
        df_original = pd.read_csv(f"data_trajectories/Original-{timesteps}.csv",dtype='float32',header=None)
        df_predicted = pd.read_csv(f"data_trajectories/Predicted-Split-{timesteps}.csv",dtype='float32',header=None)
    if pipeline == 1:
        df = pd.read_csv(f"data_trajectories/FDR-{timesteps}-timestep-trajectories11v2phases.csv",dtype='float32',header=None)

    #------------------------#
    #DATA & MODEL PREPARATION#
    #------ STARTS HERE -----#

    ID_df = df.iloc[:,0]
    CI_df = df.iloc[:,1]
    PL_df = df.iloc[:,2]

    ID_original_df = df.iloc[:,0]
    ID_predicted_df = df.iloc[:,0]

    mae_CI_test_box = []
    mae_CI_train_box = []
    mae_PL_train_box = []
    mae_PL_test_box = []

    IDs = pd.array(ID_df)
    IDs_original = pd.array(ID_original_df)
    IDs_predicted = pd.array(ID_predicted_df)

    x_df = df.iloc[:,3:]
    x_df.columns = range(x_df.shape[1])

    x_original_df = df_original.iloc[:,1:]
    x_predicted_df = df_predicted.iloc[:,1:]

    x_original_df.columns = range(x_original_df.shape[1])
    x_predicted_df.columns = range(x_predicted_df.shape[1])

    pl_min = PL_df.min(axis=0)
    ci_min = CI_df.min(axis=0)
    pl_max = PL_df.max(axis=0)
    ci_max = CI_df.max(axis=0)

    x_min_df = x_df.min(axis=0)
    x_max_df = x_df.max(axis=0)

    print(x_df[:2])
    print(x_original_df[:2])
    print(x_min_df)

    x_norm_df = (x_df - x_min_df) / (x_max_df - x_min_df)
    x_norm_original_df = (x_original_df - x_min_df) / (x_max_df - x_min_df)
    x_norm_predicted_df = (x_predicted_df - x_min_df) / (x_max_df - x_min_df)

    print(x_norm_original_df[:2])
    print(x_norm_predicted_df[:2])

    CI_norm_df = (CI_df - ci_min) / (ci_max - ci_min)
    PL_norm_df = (PL_df - pl_min) / (pl_max - pl_min)

    CI_array = CI_df.to_numpy(dtype='float32')
    PL_array = PL_df.to_numpy(dtype='float32')

    CI_norm_array = CI_norm_df.to_numpy(dtype='float32')
    PL_norm_array = PL_norm_df.to_numpy(dtype='float32')

    y_array = np.column_stack((CI_array, PL_array))
    y_norm_array = np.column_stack((CI_norm_array, PL_norm_array))
    y_df = pd.DataFrame(y_array)
    y_norm_df = pd.DataFrame(y_norm_array)

    ID_array = ID_df.to_numpy(dtype='float32')

    train_ids = []
    test_ids = []

    for line in train_file:
        train_ids.append(line.split("_")[1].strip())

    for line in test_file:
        test_ids.append(line.strip())
    
    CI_train = []
    PL_train = []
    ID_train = []
    ID_predicted = []
    ID_original = []

    x_train = []
    y_train = []
    x_train_norm = []
    y_train_norm = []

    x_original_test = []
    x_predicted_test = []
    x_original_norm_test = []
    x_predicted_norm_test = []

    CI_test = []
    PL_test = []
    ID_test = []
    
    x_test = []
    y_test = []
    x_test_norm = []
    y_test_norm = []

    index_train = []
    counter = 0
    index_test = []
    data_train = []
    data_test = []

    start = 0
    end = timesteps

    while(counter != x_df.shape[0]/timesteps):
            
            tmp_x = x_df.iloc[start:end,:].to_numpy()
            tmp_x_norm = x_norm_df.iloc[start:end,:].to_numpy()
            tmp_y = y_df.iloc[start].to_numpy()
            tmp_y_norm = y_norm_df.iloc[start].to_numpy()
            tmp_id = str(int(IDs[start]))

            if tmp_id in train_ids:
                x_train.append(tmp_x)
                x_train_norm.append(tmp_x_norm)
                y_train.append(tmp_y)
                y_train_norm.append(tmp_y_norm)
                ID_train.append(tmp_id)

            if tmp_id in test_ids:
                x_test.append(tmp_x)
                x_test_norm.append(tmp_x_norm)
                y_test.append(tmp_y)
                y_test_norm.append(tmp_y_norm)
                ID_test.append(tmp_id)
      
            start += timesteps
            end += timesteps
            counter += 1

    start = 0
    end = timesteps
    counter = 0

    while(counter != x_original_df.shape[0]/timesteps):
            
            tmp_x = x_original_df.iloc[start:end,:].to_numpy()
            tmp_x_norm = x_norm_original_df.iloc[start:end,:].to_numpy()

            tmp_id = str(int(IDs_original[start]))

            x_original_test.append(tmp_x)
            x_original_norm_test.append(tmp_x_norm)
            ID_original.append(tmp_id)
      
            start += timesteps
            end += timesteps
            counter += 1

    start = 0
    end = timesteps
    counter = 0

    while(counter != x_predicted_df.shape[0]/timesteps):
            
            tmp_x = x_predicted_df.iloc[start:end,:].to_numpy()
            tmp_x_norm = x_norm_predicted_df.iloc[start:end,:].to_numpy()

            tmp_id = str(int(IDs_predicted[start]))

            x_predicted_test.append(tmp_x)
            x_predicted_norm_test.append(tmp_x_norm)
            ID_predicted.append(tmp_id)
      
            start += timesteps
            end += timesteps
            counter += 1



    x_train = np.array(x_train, dtype='float32')
    x_train_norm = np.array(x_train_norm, dtype='float32')
    y_train = np.array(y_train, dtype='float32')
    y_train_norm = np.array(y_train_norm, dtype='float32')

    x_test = np.array(x_test, dtype='float32')
    x_test_norm = np.array(x_test_norm, dtype='float32')
    y_test = np.array(y_test, dtype='float32')
    y_test_norm = np.array(y_test_norm, dtype='float32')

    x_original = np.array(x_original_test, dtype='float32')
    x_original_norm = np.array(x_original_norm_test, dtype='float32')

    x_predicted = np.array(x_predicted_test, dtype='float32')
    x_predicted_norm = np.array(x_predicted_norm_test, dtype='float32')   

    if seperated_output:
        model = create_lstm_seperated(features, timesteps, latent_dim, dense_layer, learning_rate)
    else:
        model = create_lstm(features, timesteps, latent_dim, dense_layer, learning_rate)

    CI_col = y_train_norm[:,0]
    PL_col = y_train_norm[:,1]

    CI_col_test = y_test_norm[:,0]
    PL_col_test = y_test_norm[:,1]

    #------------------------#
    #DATA & MODEL PREPARATION#
    #-------ENDS HERE--------#

    #Model Training
    if seperated_output:
        train_model(model,x_train_norm, [CI_col,PL_col], epochs, batch_size, x_test_norm, [CI_col_test, PL_col_test])
    else:
        train_model(model,x_train_norm, y_train_norm, epochs, batch_size, x_test_norm, y_test_norm)
    
    #------------------------#
    #---TESTING AND RESULTS--#
    #------STARTS HERE-------#

    # Predict Train Set
    y_pred = model.predict(x_train_norm)
    y_pred_original = model.predict(x_original_norm)
    y_pred_predicted = model.predict(x_predicted_norm)

    if seperated_output:
        y_pred = np.column_stack(y_pred)
        y_pred_original = np.column_stack(y_pred_original)
        y_pred_predicted = np.column_stack(y_pred_predicted)

    #Model output is linear so we clip and round it in the valid
    #value ranges.

    y_pred_original[:,0] = y_pred_original[:,0] * (ci_max - ci_min) + ci_min
    y_pred_original[:,1] = y_pred_original[:,1] * (pl_max - pl_min) + pl_min
    
    y_pred_original[:,0] = y_pred_original[:,0].clip(ci_min, ci_max)
    y_pred_original[:,0] = np.round(y_pred_original[:,0])
    
    y_pred_original[:,1] = y_pred_original[:,1].clip(pl_min, pl_max)
    y_pred_original[:,1] = np.round(y_pred_original[:,1],1)

    #------

    y_pred_predicted[:,0] = y_pred_predicted[:,0] * (ci_max - ci_min) + ci_min
    y_pred_predicted[:,1] = y_pred_predicted[:,1] * (pl_max - pl_min) + pl_min
    
    y_pred_predicted[:,0] = y_pred_predicted[:,0].clip(ci_min, ci_max)
    y_pred_predicted[:,0] = np.round(y_pred_predicted[:,0])
    
    y_pred_predicted[:,1] = y_pred_predicted[:,1].clip(pl_min, pl_max)
    y_pred_predicted[:,1] = np.round(y_pred_predicted[:,1],1)

    #------

    y_pred[:,0] = y_pred[:,0] * (ci_max - ci_min) + ci_min
    y_pred[:,1] = y_pred[:,1] * (pl_max - pl_min) + pl_min
    
    y_pred[:,0] = y_pred[:,0].clip(ci_min, ci_max)
    y_pred[:,0] = np.round(y_pred[:,0])
    
    y_pred[:,1] = y_pred[:,1].clip(pl_min, pl_max)
    y_pred[:,1] = np.round(y_pred[:,1],1)

    mae_CI_train = mean_absolute_error(y_train[:,0], y_pred[:,0])
    mae_PL_train = mean_absolute_error(y_train[:,1], y_pred[:,1])

    print("Train Results: \n")
    print(f'CI Train MAE: {mae_CI_train}')
    print(f'PL Train MAE: {mae_PL_train}')
        
    std_CI_train = []
    std_PL_train = []
    
    for i in range(len(y_pred)):
    
        CI_loss_train = np.abs(y_train[i,0] - y_pred[i,0])
        PL_loss_train = np.abs(y_train[i,1] - y_pred[i,1])
    
    
        std_CI_train.append(CI_loss_train)
        std_PL_train.append(PL_loss_train)
    
    std_CI_train = np.std(std_CI_train)
    std_PL_train = np.std(std_PL_train)

    for i in range(len(y_pred)):
        mae_CI_train_box.append(abs(y_train[i,0] - y_pred[i,0]))
        mae_PL_train_box.append(abs(y_train[i,1] - y_pred[i,1]))

    plt.clf()
    plt.boxplot(mae_CI_train_box)
    plt.savefig(f"lstm{pipeline}_{experiment}/mae_CI_train_boxplot.png")

    plt.clf()
    plt.boxplot(mae_PL_train_box)
    plt.savefig(f"lstm{pipeline}_{experiment}/mae_PL_train_boxplot.png")

    mae_CI_train = mean_absolute_error(y_train[:,0], y_pred[:,0])
    mae_PL_train = mean_absolute_error(y_train[:,1], y_pred[:,1])

    print(f'Std CI Train: {std_CI_train}')
    print(f'Std PL Train: {std_PL_train}')
    print(f'----')

    CI_TRAIN_MIN = np.min(mae_CI_train_box)
    CI_TRAIN_MAX = np.max(mae_CI_train_box)
    CI_TRAIN_IQR = np.subtract(*np.percentile(mae_CI_train_box, [75,25]))
    PL_TRAIN_MIN = np.min(mae_PL_train_box)
    PL_TRAIN_MAX = np.max(mae_PL_train_box)
    PL_TRAIN_IQR = np.subtract(*np.percentile(mae_PL_train_box, [75,25]))

    print(f'CI Train MAE min: {CI_TRAIN_MIN}')
    print(f'CI Train MAE max: {CI_TRAIN_MAX}')
    print(f'CI Train MAE IQR: {CI_TRAIN_IQR}')
    print(f'PL Train MAE min: {PL_TRAIN_MIN}')
    print(f'PL Train MAE max: {PL_TRAIN_MAX}')
    print(f'PL Train MAE IQR: {PL_TRAIN_IQR}')

    with open(f'lstm{pipeline}_{experiment}/train_predictions.txt', 'w') as f:
        for i in range(len(y_test)):
            f.write(f'target: {y_train[i]} | pred: {y_pred[i]}\n')
    
    # Predict test set
    y_pred = model.predict(x_test_norm)

    if seperated_output:
        y_pred = np.column_stack(y_pred)

    y_pred[:,0] = y_pred[:,0] * (ci_max - ci_min) + ci_min
    y_pred[:,1] = y_pred[:,1] * (pl_max - pl_min) + pl_min
    
    y_pred[:,0] = y_pred[:,0].clip(ci_min, ci_max)
    y_pred[:,0] = np.round(y_pred[:,0])
    
    y_pred[:,1] = y_pred[:,1].clip(pl_min, pl_max)
    y_pred[:,1] = np.round(y_pred[:,1],1)
    
    mae_CI_test = mean_absolute_error(y_test[:,0], y_pred[:,0])
    mae_PL_test = mean_absolute_error(y_test[:,1], y_pred[:,1])
    
    print("\nTest Results:\n")
    print(f'CI Test MAE: {mae_CI_test}')
    print(f'PL Test MAE: {mae_PL_test}')
    
    std_CI_test = []
    std_PL_test = []
    
    for i in range(len(y_pred)):
    
        CI_loss_test = np.abs(y_test[i,0] - y_pred[i,0])
        PL_loss_test = np.abs(y_test[i,1] - y_pred[i,1])
    
    
        std_CI_test.append(CI_loss_test)
        std_PL_test.append(PL_loss_test)
    
    std_CI_test = np.std(std_CI_test)
    std_PL_test = np.std(std_PL_test)

    for i in range(len(y_pred)):
        mae_CI_test_box.append(abs(y_test[i,0] - y_pred[i,0]))
        mae_PL_test_box.append(abs(y_test[i,1] - y_pred[i,1]))

    print(f'Std CI Test: {std_CI_test}')
    print(f'Std PL Test: {std_PL_test}')
    print(f'----')

    CI_TEST_MIN = np.min(mae_CI_test_box)
    CI_TEST_MAX = np.max(mae_CI_test_box)
    CI_TEST_IQR = np.subtract(*np.percentile(mae_CI_test_box, [75,25]))
    PL_TEST_MIN = np.min(mae_PL_test_box)
    PL_TEST_MAX = np.max(mae_PL_test_box)
    PL_TEST_IQR = np.subtract(*np.percentile(mae_PL_test_box, [75,25]))
    
    print(f'CI Train MAE min: {CI_TEST_MIN}')
    print(f'CI Train MAE max: {CI_TEST_MAX}')
    print(f'CI Train MAE IQR: {CI_TEST_IQR}')
    print(f'PL Train MAE min: {PL_TEST_MIN}')
    print(f'PL Train MAE max: {PL_TEST_MAX}')
    print(f'PL Train MAE IQR: {PL_TEST_IQR}')

    plt.clf()
    plt.boxplot(mae_CI_test_box)
    plt.savefig(f"lstm{pipeline}_{experiment}/mae_CI_test_boxplot_lstm.png")

    plt.clf()
    plt.boxplot(mae_PL_test_box)
    plt.savefig(f"lstm{pipeline}_{experiment}/mae_PL_test_boxplot_lstm.png")

    with open(f'lstm{pipeline}_{experiment}/test_predictions.txt', 'w') as f:
        for i in range(len(y_test)):
            f.write(f'target: {y_test[i]} | pred: {y_pred[i]}\n')

    if not os.path.exists('LSTM-EXPERIMENTS.csv'):
        # If the file doesn't exist, create it and write the header
        with open('LSTM-EXPERIMENTS.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)

    with open('LSTM-EXPERIMENTS.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        print(str(learning_rate))
        csvwriter.writerow(header)

        row = [experiment, pipeline, timesteps, epochs, batch_size, latent_dim, latent_dim, dense_layer, format(learning_rate, '.6f'), scale, seperated_output, str(mae_CI_train), str(std_CI_train), str(mae_PL_train), str(std_PL_train), str(mae_CI_test), str(std_CI_test), str(mae_PL_test), str(std_PL_test), str(CI_TRAIN_MIN), str(CI_TRAIN_MAX), str(CI_TRAIN_IQR), str(PL_TRAIN_MIN), str(PL_TRAIN_MAX), str(PL_TRAIN_IQR), str(CI_TEST_MIN), str(CI_TEST_MAX), str(CI_TEST_IQR), str(PL_TEST_MIN), str(PL_TEST_MAX), str(PL_TEST_IQR)]
        print(row)
        csvwriter.writerow(row)

    with open(f'lstm{pipeline}_{experiment}/original_preds.txt', 'w') as out:
        out.write('FLIGHT ID | ALLFT_CI | ALLFT_PL\n')
        for i in range(len(y_pred_original)):
            out.write(f'{ID_original[i]} | {int(y_pred_original[i,0])} | {round(y_pred_original[i,1])}\n')

    with open(f'lstm{pipeline}_{experiment}/predicted_preds.txt', 'w') as out:
        out.write('FLIGHT ID | PREDICTED_CI | PREDICTED_PL\n')
        for i in range(len(y_pred_predicted)):
            out.write(f'{ID_predicted[i]} | {int(y_pred_predicted[i,0])} | {round(y_pred_predicted[i,1])}\n')

    with open(f'lstm{pipeline}_{experiment}/dynamo_tests.txt', 'w') as out:
        out.write('FLIGHT ID | REAL_CI | REAL_PL | CI_OUT | PL_OUT\n')
        for i in range(len(y_pred)):
            out.write(f'{ID_test[i]} | {int(y_test[i,0])} | {round(y_test[i,1])}| {int(y_pred[i,0])}| {round(y_pred[i,1])}\n')

    #------------------------#
    #---TESTING AND RESULTS--#
    #-------ENDS HERE--------#

if __name__ == "__main__":
    main()
