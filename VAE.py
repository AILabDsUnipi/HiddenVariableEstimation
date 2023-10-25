import pandas as pd 
import os
import platform
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import mean_absolute_error as mae
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, Bidirectional
from keras.layers import Dense,Lambda,TimeDistributed
from keras.layers import concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
from tensorflow import keras
from tensorflow.python.framework.ops import disable_eager_execution
import sys
import csv

#tf.config.run_functions_eagerly(True)

def make_directory(directory_path):
    try:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    except FileExistsError:
        print(f"Error: Directory '{directory_path}' already exists.")
        raise

def standard_normalize(array1, array2, hidden1, hidden2):
    combined = np.concatenate((array1, array2, hidden1, hidden2), axis=0)
    mean = np.mean(combined, axis=(0, 1))
    std = np.std(combined, axis=(0, 1))
    array1_normalized = (array1 - mean) / std
    array2_normalized = (array2 - mean) / std
    hidden1_normalized = (hidden1 - mean) / std
    hidden2_normalized = (hidden2 - mean) / std

    return array1_normalized, array2_normalized, hidden1_normalized, hidden2_normalized

def min_max_normalize(array1, array2, hidden1, hidden2):
    combined = np.concatenate((array1, array2, hidden1, hidden2), axis=0)
    all_min = np.min(combined, axis=(0, 1))
    all_max = np.max(combined, axis=(0, 1))
    array1_normalized = (array1 - all_min) / (all_max - all_min)
    array2_normalized = (array2 - all_min) / (all_max - all_min)
    hidden1_normalized = (hidden1 - all_min) / (all_max - all_min)
    hidden2_normalized = (hidden2 - all_min) / (all_max - all_min)

    return array1_normalized, array2_normalized, hidden1_normalized, hidden2_normalized


def MinMaxNormalizeData(df_feats):
    
    df_feats_copy = df_feats.copy(deep = True)

    feats_min_df = df_feats_copy.min(axis=0)
    feats_max_df = df_feats_copy.max(axis=0)

    df_feats_copy = (df_feats_copy - feats_min_df) / (feats_max_df - feats_min_df)

    return df_feats_copy

def StandardNormalizeData(df_feats):
    
    df_feats_copy = df_feats.copy(deep = True)

    feats_min_df = df_feats_copy.std(axis=0)
    feats_max_df = df_feats_copy.mean(axis=0)

    df_feats_copy = (df_feats_copy - feats_min_df) / (feats_max_df - feats_min_df)

    return df_feats_copy


def makeTrajectories(timesteps, pipeline): 

    end = timesteps
    start = 0
    counter = 0
    
    path = os.getcwd()
    
    #all trajectories used for train
    all_df_list_Train = [] 
    CI_list = []
    PL_list = []
    ID_list = []

    original_feats = []
    ID_list_original = []

    predicted_feats = []
    ID_list_predicted = []

    #Load dataSet --Start--

    if pipeline == 2:
        df_train = pd.read_csv(f"data_trajectories/FDR-{timesteps}-timestep-trajectories6v2phases.csv",dtype='unicode',header=None)
        df_original = pd.read_csv(f"data_trajectories/Original-{timesteps}.csv",dtype='unicode',header=None)
        df_predicted = pd.read_csv(f"data_trajectories/Predicted-Split-{timesteps}.csv",dtype='unicode',header=None)
    if pipeline == 1:
        df_train = pd.read_csv(f"data_trajectories/FDR-{timesteps}-timestep-trajectories11v2phases.csv",dtype='unicode',header=None)

    ID = df_train.iloc[:,0]
    CI = df_train.iloc[:,1]
    PL = df_train.iloc[:,2]

    ID_original = df_original.iloc[:,0]
    ID_predicted = df_predicted.iloc[:,0]

    #Load dataSet --End--
  
    #Keep only Features columns
    df_feats = (df_train.iloc[:,3:]).astype(float)
    df_feats_original = (df_original.iloc[:,1:]).astype(float)
    df_feats_predicted = (df_predicted.iloc[:,1:]).astype(float)

    # Min-Max Scaler
    df_norm = MinMaxNormalizeData(df_feats)
    df_original_norm = MinMaxNormalizeData(df_feats_original)
    df_predicted_norm = MinMaxNormalizeData(df_feats_predicted)    
    
    #Split dataSet into Trajectories of timestep row-length
    #trj number = 

    while(counter != df_norm.shape[0]/timesteps):

        tmp = df_norm.iloc[start:end,:].to_numpy()
        all_df_list_Train.append(tmp)
        
        tmpCI = CI.iat[start]
        CI_list.append(tmpCI)
        
        tmpPL = PL.iat[start]
        PL_list.append(tmpPL)

        tmpID = ID.iat[start]
        ID_list.append(tmpID)
        
  
        start += timesteps
        end += timesteps
        counter += 1

    end = timesteps
    start = 0
    counter = 0

    while(counter != df_original_norm.shape[0]/timesteps):

        tmp = df_original_norm.iloc[start:end,:].to_numpy()
        original_feats.append(tmp)

        tmpID = ID_original.iat[start]
        ID_list_original.append(tmpID)
        
  
        start += timesteps
        end += timesteps
        counter += 1

    end = timesteps
    start = 0
    counter = 0

    while(counter != df_predicted_norm.shape[0]/timesteps):

        tmp = df_predicted_norm.iloc[start:end,:].to_numpy()
        predicted_feats.append(tmp)

        tmpID = ID_predicted.iat[start]
        ID_list_predicted.append(tmpID)
        
  
        start += timesteps
        end += timesteps
        counter += 1

    finalTrainDt = np.array(all_df_list_Train)
    final_original = np.array(original_feats)
    final_predicted = np.array(predicted_feats)

    return finalTrainDt, ID_list, CI_list, PL_list, original_feats, ID_list_original, predicted_feats, ID_list_predicted

def create_lstm_vae(input_dim, timesteps, batch_size, intermediate_dim, latent_dim, epsilon_std, learning_rate):

    # input layer expects input will be batches of steps*features-dimensional vectors.
    x = Input(batch_shape=(None, timesteps, input_dim,), name="Input_layer")
    
    # LSTM encoding
    h = Bidirectional(LSTM(intermediate_dim, activation = 'relu', name = "Encoder_lstm_layer"))(x)

    # VAE Z layer / sample latent layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon
        #return z_mean + K.exp(z_log_sigma / 2) * epsilon
    
    # sampling output
    z = Lambda(sampling)([z_mean, z_log_sigma])
    
    
    # decoded LSTM layer
    h_decoded = RepeatVector(timesteps)(z)
    decoder_h = LSTM(intermediate_dim, activation='relu' ,return_sequences = True, name = "decoder_lstm_layer")(h_decoded)
    decoder_mean = TimeDistributed(Dense(input_dim, activation="sigmoid"))(decoder_h)


    # end-to-end autoencoder
    vae = Model(x, decoder_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    concat = concatenate([z_mean, h])

    l = Model(x, h)
    c = Model(x, concat)

    
    def vae_loss(x, decoder_mean):
        xent_loss = metrics.mse(x, decoder_mean)
        kl_loss = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss
     

    vae.compile(optimizer = Adam(learning_rate = learning_rate), loss = vae_loss, metrics=['accuracy'])
    vae.summary()
   
    return vae, encoder, l, c

def train_model(vae, x_train, epochs, batch_size):

    x_tensor = tf.convert_to_tensor(x_train)

    vae.fit(x_tensor, x_tensor, epochs = epochs, batch_size = batch_size, shuffle=(True))       
       

def visualize(x_train, preds, timesteps, features, pipeline, experiment):
    
    print("[plotting...]")
    
    rdm = random.randint(0, (len(x_train) - 1))
    arr=np.array([[x_train[rdm]]])
    arr=arr.reshape(timesteps,features)
    plt.title("Original Trajectory")
    plt.xlabel("Timesteps")
    plt.ylabel("Features Value")
    plt.plot(arr)
    plt.savefig(f"vae{pipeline}_{experiment}/original-{features}.png")
    plt.clf()
    #plt.show()
    
    arr2=np.array([[preds[rdm]]])
    arr2=arr2.reshape(timesteps,features)
    plt.title("Reconstructed Trajectory")
    plt.xlabel("Timesteps")
    plt.ylabel("Features Value")
    plt.plot(arr2)
    plt.savefig(f"vae{pipeline}_{experiment}/reconstructed-{features}.png")
    #plt.show()

def evaluation(x_train, preds, timesteps, features, train, pipeline, experiment):
    # Mean absolute error function of train and predicted matrixes and boxplot
    mae_sum = []

    for i in range(len(x_train)):
        mae_sum.append(mae(x_train[i], preds[i]))

    mae_mean = np.mean(np.array(mae_sum))
    mae_std = np.std(np.array(mae_sum))

    print(f'MEAN OF MAE: {mae_mean}')
    print(f'STD OF MAE: {mae_std}')

    plt.clf()
    plt.boxplot(mae_sum)
    plt.savefig(f"vae{pipeline}_{experiment}/{train}_mae_boxplot-{features}.png")   

    return mae_mean, mae_std

def main():
          
    #initialization of model parameters

    experiment = random.randint(0, 1000)
    pipeline = 2
    if pipeline == 1:
        features = 11
    else:
        features = 6
    epochs = 100
    batch_size = 256
    input_dim = features 
    intermediate_dim = 64
    latent_dim = 32
    timesteps = 30
    epsilon_std = 1.0
    learning_rate = 0.001

    seed = 42

    header = ['Experiment', 'Pipeline', 'Timesteps', 'Epochs', 'batch_size', 'LSTM', 'latent', 'learning_rate', 'VAE TRAIN MAE', 'VAE TRAIN STD', 'VAE TEST MAE', 'VAE TEST STD']

    make_directory(f'vae{pipeline}_{experiment}')

    train_file = open('data_trajectories/train_ids.txt')
    test_file = open('data_trajectories/test_ids.txt')

    train_ids = []
    test_ids = []

    for line in train_file:
        train_ids.append(line.split("_")[1].strip())

    for line in test_file:
        test_ids.append(line.strip())

    #Dataframe after pre process and MinMaxScaler
    Data, ID, CI, PL, Data_original, ID_original, Data_predicted, ID_predicted = makeTrajectories(timesteps, pipeline)

    trainData = []
    testData = []
    allData = []


    #random.seed(seed)
    #rdm = random.sample(range(Data.shape[0]), trajectory_num)
    #rdm = sorted(rdm)
    
    CI_train = []
    PL_train = []
    ID_train = []
    
    CI_test = []
    PL_test = []
    ID_test = []

    CI_all = []
    PL_all = []
    ID_all = []

    
    for i in range(Data.shape[0]):
        if ID[i] in train_ids:
            trainData.append(Data[i])
            CI_train.append(CI[i])
            PL_train.append(PL[i])
            ID_train.append(ID[i])

        
        if ID[i] in test_ids:
            
            testData.append(Data[i])
            CI_test.append(CI[i])
            PL_test.append(PL[i])
            ID_test.append(ID[i])

        allData.append(Data[i])
        CI_all.append(CI[i])
        PL_all.append(PL[i])
        ID_all.append(ID[i])

   
    trainData = np.asarray(trainData, dtype = np.float32)
    testData = np.asarray(testData, dtype = np.float32)
    allData = np.asarray(allData, dtype = np.float32)

    Data_predicted = np.asarray(Data_predicted, dtype = np.float32)
    Data_original = np.asarray(Data_original, dtype = np.float32)    

    print(f'CI_train shape {len(CI_train)}')
    print(f'PL_train shape {len(PL_train)}') 

    print(f'CI_test shape {len(CI_test)}')
    print(f'PL_test shape {len(PL_test)}') 

    print(f'All Data shape: {Data.shape}')

    print(f'Train dataset shape: {trainData.shape}')
    print(f'Test dataset shape: {testData.shape}')

    print(f'Hidden Original shape: {Data_predicted.shape}')
    print(f'Hidden Predicted shape: {Data_original.shape}')   
    
    # Create and Train model
    vae, encoder, l, c = create_lstm_vae(input_dim, timesteps, batch_size, intermediate_dim, latent_dim, epsilon_std, learning_rate)
    

    train_model(vae, allData, epochs, batch_size)
    
    vae.save(f'vae{pipeline}_{experiment}/vae')
    encoder.save(f'vae{pipeline}_{experiment}/encoder')

    # Predict train set
    predsDataTrain = vae.predict(trainData)
    print(f"Train preds: {predsDataTrain.shape}")
    
    # Predict test set
    predsDataTest = vae.predict(testData)
    predsOriginal = vae.predict(Data_original)
    predsPredicted = vae.predict(Data_predicted)
    print(f"Test preds: {predsDataTest.shape}")

    print("------------------------")

    # latent for train
    latent_train = encoder.predict(trainData)
    print(f"latent train shape: {latent_train.shape}") 
    
    # latent for test
    latent_test = encoder.predict(testData)

    latent_hidden_original = encoder.predict(Data_original)
    latent_hidden_predicted = encoder.predict(Data_predicted)

    latent_train_norm_standard, latent_test_norm_standard, latent_original_norm_standard, latent_predicted_norm_standard = standard_normalize(latent_train, latent_test, latent_hidden_original, latent_hidden_predicted)
    latent_train_norm_mm, latent_test_norm_mm, latent_original_norm_mm, latent_predicted_norm_mm = min_max_normalize(latent_train, latent_test, latent_hidden_original, latent_hidden_predicted)

    print(f"latent test shape: {latent_test.shape}") 

    # Prepeare files for regression
    df_train_mm = pd.DataFrame(latent_train_norm_mm)
    df_test_mm = pd.DataFrame(latent_test_norm_mm)
    df_original_mm = pd.DataFrame(latent_original_norm_mm)
    df_predicted_mm = pd.DataFrame(latent_predicted_norm_mm)

    # Train file
    df_train_mm.insert(latent_train.shape[1], "CI", CI_train, True)
    df_train_mm.insert(latent_train.shape[1]+1, "PL", PL_train, True)
    df_train_mm.insert(latent_train.shape[1]+2, "ID", ID_train, True)
    df_train_mm.to_csv(f"vae{pipeline}_{experiment}/{timesteps}-timesteps-{latent_train.shape[1]}-latent-train-{features}-mm.csv")
    
    # Test file
    df_test_mm.insert(latent_test.shape[1], "CI", CI_test, True)
    df_test_mm.insert(latent_test.shape[1]+1, "PL", PL_test, True)
    df_test_mm.insert(latent_test.shape[1]+2, "ID", ID_test, True)
    df_test_mm.to_csv(f'vae{pipeline}_{experiment}/{timesteps}-timesteps-{latent_test.shape[1]}-latent-test-{features}-mm.csv')
   
    # Hidden File

    df_original_mm.insert(latent_hidden_original.shape[1], "ID", ID_original, True)
    df_original_mm.to_csv(f'vae{pipeline}_{experiment}/{timesteps}-timesteps-{latent_hidden_original.shape[1]}-latent-original-{features}-mm.csv')

    df_predicted_mm.insert(latent_hidden_predicted.shape[1], "ID", ID_predicted, True)
    df_predicted_mm.to_csv(f'vae{pipeline}_{experiment}/{timesteps}-timesteps-{latent_hidden_original.shape[1]}-latent-predicted-{features}-mm.csv')

    # Prepeare files for regression
    df_train_standard = pd.DataFrame(latent_train_norm_standard)
    df_test_standard = pd.DataFrame(latent_test_norm_standard)
    df_original_standard = pd.DataFrame(latent_original_norm_standard)
    df_predicted_standard = pd.DataFrame(latent_predicted_norm_standard)

    # Train file
    df_train_standard.insert(latent_train.shape[1], "CI", CI_train, True)
    df_train_standard.insert(latent_train.shape[1]+1, "PL", PL_train, True)
    df_train_standard.insert(latent_train.shape[1]+2, "ID", ID_train, True)
    df_train_standard.to_csv(f"vae{pipeline}_{experiment}/{timesteps}-timesteps-{latent_train.shape[1]}-latent-train-{features}-standard.csv")
    
    # Test file
    df_test_standard.insert(latent_test.shape[1], "CI", CI_test, True)
    df_test_standard.insert(latent_test.shape[1]+1, "PL", PL_test, True)
    df_test_standard.insert(latent_test.shape[1]+2, "ID", ID_test, True)
    df_test_standard.to_csv(f'vae{pipeline}_{experiment}/{timesteps}-timesteps-{latent_test.shape[1]}-latent-test-{features}-standard.csv')

    # Hidden File

    df_original_standard.insert(latent_hidden_original.shape[1], "ID", ID_original, True)
    df_original_standard.to_csv(f'vae{pipeline}_{experiment}/{timesteps}-timesteps-{latent_hidden_original.shape[1]}-latent-original-{features}-standard.csv')

    df_predicted_standard.insert(latent_hidden_predicted.shape[1], "ID", ID_predicted, True)
    df_predicted_standard.to_csv(f'vae{pipeline}_{experiment}/{timesteps}-timesteps-{latent_hidden_original.shape[1]}-latent-predicted-{features}-standard.csv')

    # Visualization of train set
    visualize(testData, predsDataTest, timesteps, features, pipeline, experiment)

    visualize(Data_original, predsOriginal, timesteps, features, 'Original', experiment)
    
    visualize(Data_predicted, predsPredicted, timesteps, features, 'Original', experiment)

    # Evaluation of Training
    print('\nTraining Evaluation\n')
   
    train_mae, train_std = evaluation(trainData, predsDataTrain, timesteps, features, 'train', pipeline, experiment)
    
    # Evaluation of Testing
    print('\nTest Evaluation\n')
  
    test_mae, test_std = evaluation(testData, predsDataTest, timesteps, features, 'test', pipeline, experiment)
    
    if not os.path.exists('VAE-EXPERIMENTS.csv'):
        # If the file doesn't exist, create it and write the header
        with open('VAE-EXPERIMENTS.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

    with open('VAE-EXPERIMENTS.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        header = ['Experiment', 'Pipeline', 'Timesteps', 'Epochs', 'batch_size', 'LSTM', 'latent', 'learning_rate', 'epsilon_std' 'VAE TRAIN MAE', 'VAE TRAIN STD', 'VAE TEST MAE', 'VAE TEST STD']
        row = [experiment, pipeline, timesteps, epochs, batch_size, intermediate_dim, latent_dim, format(learning_rate, '.6f'), epsilon_std, train_mae, train_std, test_mae, test_std]
        print(row)
        csvwriter.writerow(row)

if __name__ == "__main__":
    main()
