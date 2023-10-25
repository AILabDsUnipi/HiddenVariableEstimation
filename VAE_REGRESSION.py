import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from torch.utils.data import DataLoader

import random
import numpy as np
import os
import glob
import csv

import sys
import torch as T
import torch.nn as nn
from torch.autograd import Variable 

from torch.utils.data import Dataset, TensorDataset


def make_directory(directory_path):
    try:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    except FileExistsError:
        print(f"Error: Directory '{directory_path}' already exists.")
        raise


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)



drop = None
activation = 'linear'

class NN(nn.Module):
    def __init__(self,input_size,output_size, num_n1, num_n2 = 0):
        super(NN,self).__init__()
        self.fc_linear1 = nn.Linear(input_size,num_n1)
        self.relu = nn.ReLU()
        self.num_n1 = num_n1
        self.num_n2 = num_n2
        if(drop != None):
            self.dropout = nn.Dropout(p=drop)
        if(activation == 'sigmoid'):
            self.sig = nn.Sigmoid()

        if(self.num_n2 != 0):
            self.fc_linear2 = nn.Linear(num_n1,num_n2)
            self.relu2 = nn.ReLU()
            self.fc_linear3 = nn.Linear(num_n2,2)
        else:
            self.fc_linear2 = nn.Linear(num_n1,2)


    def forward(self,x):
        if(self.num_n2 != 0):
            out = self.fc_linear1(x) #Forward propogation 
            out = self.relu(out)
            if drop != None:
                out = self.dropout(out)
            out = self.fc_linear2(out)
            out = self.relu2(out)
            out = self.fc_linear3(out)
            if activation == 'sigmoid':
                out = self.sig(out)
        else:
            out = self.fc_linear1(x) #Forward propogation 
            out = self.relu(out)
            if drop != None:
                out = self.dropout(out)
            out = self.fc_linear2(out)
            if activation == 'sigmoid':
                out = self.sig(out)

        return out

path = os.getcwd()

def MinMaxScaler(df):
    
    copy_df = df.copy(deep = True)
    
    for i in range(len(copy_df.axes[1])):
        feat = copy_df.iloc[0:,i]
        diff = feat.max() - feat.min()
        feat += (-1)*(feat.min())
        if diff == 0:
            feat = 0.0
        else:
            feat = feat/diff
        
        copy_df.iloc[0:,i] = feat

    return copy_df

def MinMaxScalerY(df):

    copy_df = df.copy(deep = True)

    CI = copy_df.iloc[0:,0]
    PL = copy_df.iloc[0:,1]
    
    max_ci = CI.max()
    max_pl = PL.max() 
    
    min_ci = CI.min()
    min_pl = PL.min()
    
    max_minus_min1 = max_ci - min_ci
    max_minus_min2 = max_pl - min_pl

    CI +=(-1)*min_ci
    CI = CI/max_minus_min1
    
    PL +=(-1)*min_pl
    PL = PL/max_minus_min2
    
    copy_df.iloc[0:,0] = CI
    copy_df.iloc[0:,1] = PL

    return copy_df, max_ci, min_ci, max_pl, min_pl

def PrepareTrainData(path, scale):
    
    # Get Train Data

    if path.split("_")[0][-1] == '1':


        if scale == 'minmax':

            df = pd.read_csv(f"{path}-train-11-mm.csv",dtype='float32',header=0)
        
        else:

            df = pd.read_csv(f"{path}-train-11-standard.csv",dtype='float32',header=0)
    else:

        if scale == 'minmax':

            df = pd.read_csv(f"{path}-train-6-mm.csv",dtype='float32',header=0)

        else:

            df = pd.read_csv(f"{path}-train-6-standard.csv",dtype='float32',header=0)
    
    cols = df.shape[1]

    x_train_df = df.iloc[0:,1:(cols-3)].astype('float32')

    y_train_df = df.iloc[0:,(cols-3):cols-1].astype('float32') 

    y_train_df_norm,max_ci,min_ci,max_pl,min_pl = MinMaxScalerY(y_train_df)

    x_train_describe = x_train_df.describe()

    x_train = x_train_df.to_numpy().astype(np.float32)
    y_train = y_train_df.to_numpy().astype(np.float32)

    y_train_norm = y_train_df_norm.to_numpy().astype(np.float32)

    
    return x_train, y_train, y_train_norm, max_ci, min_ci, max_pl, min_pl

def PrepareTestData(path, scale):
    
    if path.split("_")[0][-1] == '1':

        if scale == 'minmax':

            df = pd.read_csv(f"{path}-test-11-mm.csv",dtype='float32',header=0)

        else:

            df = pd.read_csv(f"{path}-test-11-standard.csv",dtype='float32',header=0)
    else:

        if scale == 'minmax':

            df = pd.read_csv(f"{path}-test-6-mm.csv",dtype='float32',header=0)
            df_original = pd.read_csv(f"{path}-original-6-mm.csv",dtype='float32',header=0)
            df_predicted = pd.read_csv(f"{path}-predicted-6-mm.csv",dtype='float32',header=0)

        else:

            df = pd.read_csv(f"{path}-test-6-standard.csv",dtype='float32',header=0)
            df_original = pd.read_csv(f"{path}-original-6-standard.csv",dtype='float32',header=0)
            df_predicted = pd.read_csv(f"{path}-predicted-6-standard.csv",dtype='float32',header=0)

    cols = df.shape[1]
    cols_hidden = df_original.shape[1]

    x_test_df = df.iloc[0:,1:(cols-3)].astype('float32')

    y_test_df = df.iloc[0:,(cols-3):cols-1].astype('float32') 

    x_original_df = df_original.iloc[0:,1:(cols_hidden-1)].astype('float32')

    x_predicted_df = df_predicted.iloc[0:,1:(cols_hidden-1)].astype('float32')

    test_ids_df = df.iloc[0:,-1].astype(int)

    original_ids_df = df_original.iloc[0:,-1].astype(int)

    predicted_ids_df = df_predicted.iloc[0:,-1].astype(int)

    y_test_norm_df, max_ci_test, min_ci_test, max_pl_test, min_pl_test = MinMaxScalerY(y_test_df)

    y_test_norm = y_test_norm_df.to_numpy().astype(np.float32)

    x_original = x_original_df.to_numpy().astype(np.float32)
    x_predicted = x_predicted_df.to_numpy().astype(np.float32)
    x_test = x_test_df.to_numpy().astype(np.float32)
    y_test = y_test_df.to_numpy().astype(np.float32)
    original_ids = original_ids_df.to_numpy().astype(int)
    predicted_ids = predicted_ids_df.to_numpy().astype(int)
    test_ids = test_ids_df.to_numpy().astype(int)

    return x_test, y_test, test_ids, x_original, x_predicted, original_ids, predicted_ids, y_test_norm, max_ci_test, min_ci_test, max_pl_test, min_pl_test
    
def main():

    #Regression parameters, we have lists in order to run multiple tests if needed in one run
    split_list = [2,5,10]
    leaf_list = [2,5,10]
    depth_list = [6,7]
    scale_list = ['standard', 'minmax']
    lr_list = [0.1]
    e_num_list = [1000]
    latent_features = '30-timesteps-32-latent'

    for sp in split_list:
        for l in leaf_list:
            for d in depth_list:
                for sc in scale_list:
                    #parameters
                    random_number = random.randint(0, 10000)
                    model_type = 'gbm'
                    experiment = 'vae2_4'
                    path = f'{experiment}/{latent_features}'
                    scale = sc

                    make_directory(f'{experiment}/regr_{random_number}')

                    header = ['experiment', 'split', 'leaf', 'depth', 'lr', 'e_num', 'scale', 'mae_CI_train', 'std_CI_train', 'mae_PL_train', 'std_PL_train', 'mae_CI_test', 'std_CI_test', 'mae_PL_test', 'std_PL_test', 'CI_TRAIN_MIN', 'CI_TRAIN_MAX', 'CI_TRAIN_IQR', 'PL_TRAIN_MIN', 'PL_TRAIN_MAX', 'PL_TRAIN_IQR', 'CI_TEST_MIN', 'CI_TEST_MAX', 'CI_TEST_IQR', 'PL_TEST_MIN', 'PL_TEST_MAX', 'PL_TEST_IQR']
                    
                    split = sp
                    leaf = l
                    depth = d
                    lr = lr_list[0]
                    e_num = e_num_list[0]
                    verbose = True
                    max_features = None

                    params_gbm = {
                        'min_samples_split': split,
                        'min_samples_leaf':leaf,
                        'n_estimators' : e_num,
                        'max_depth': depth,
                        'learning_rate': lr,
                        'verbose': verbose,
                        'max_features': max_features
                    }
                    
                    gs = MultiOutputRegressor(ensemble.GradientBoostingRegressor(**params_gbm), n_jobs = -1) 

                    x_train, y_train, y_train_norm, max_ci_train, min_ci_train, max_pl_train, min_pl_train = PrepareTrainData(path, scale)
                    x_test, y_test, test_ids, x_original, x_predicted, original_ids, predicted_ids, y_test_norm, max_ci_test, min_ci_test, max_pl_test, min_pl_test = PrepareTestData(path, scale)

                    ci_min = min(min_ci_train, min_ci_test)
                    pl_min = min(min_pl_train, min_pl_test)
                    ci_max = max(max_ci_train, max_ci_test)
                    pl_max = max(max_pl_train, max_pl_test)

                    gs.fit(x_train, y_train_norm)

                    mae_CI_test_box = []
                    mae_CI_train_box = []
                    mae_PL_train_box = []
                    mae_PL_test_box = []

                    y_pred = gs.predict(x_train)
                    y_pred_original = gs.predict(x_original)
                    y_pred_predicted = gs.predict(x_predicted)

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
                    plt.savefig(f"{experiment}/regr_{random_number}/mae_CI_train_boxplot.png")

                    plt.clf()
                    plt.boxplot(mae_PL_train_box)
                    plt.savefig(f"{experiment}/regr_{random_number}/mae_PL_train_boxplot.png")

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

                    with open(f'{experiment}/regr_{random_number}/train_predictions.txt', 'w') as f:
                        for i in range(len(y_test)):
                            f.write(f'target: {y_train[i]} | pred: {y_pred[i]}\n')
                    
                    # Predict test set
                    y_pred = gs.predict(x_test)

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
                    plt.savefig(f"{experiment}/regr_{random_number}/mae_CI_test_boxplot.png")

                    plt.clf()
                    plt.boxplot(mae_PL_test_box)
                    plt.savefig(f"{experiment}/regr_{random_number}/mae_PL_test_boxplot.png")

                    with open(f'{experiment}/regr_{random_number}//test_predictions.txt', 'w') as f:
                        for i in range(len(y_test)):
                            f.write(f'target: {y_test[i]} | pred: {y_pred[i]}\n')

                    if not os.path.exists(f'{experiment}/REGRESSION-EXPERIMENTS.csv'):
                        # If the file doesn't exist, create it and write the header
                        with open(f'{experiment}/REGRESSION-EXPERIMENTS.csv', 'w', newline='') as csvfile:
                            csvwriter = csv.writer(csvfile)
                            csvwriter.writerow(header)

                    with open(f'{experiment}/REGRESSION-EXPERIMENTS.csv', 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)

                        row = [random_number, split, leaf, depth, lr, e_num, scale, str(mae_CI_train), str(std_CI_train), str(mae_PL_train), str(std_PL_train), str(mae_CI_test), str(std_CI_test), str(mae_PL_test), str(std_PL_test), str(CI_TRAIN_MIN), str(CI_TRAIN_MAX), str(CI_TRAIN_IQR), str(PL_TRAIN_MIN), str(PL_TRAIN_MAX), str(PL_TRAIN_IQR), str(CI_TEST_MIN), str(CI_TEST_MAX), str(CI_TEST_IQR), str(PL_TEST_MIN), str(PL_TEST_MAX), str(PL_TEST_IQR)]
                        csvwriter.writerow(row)

                    with open(f'{experiment}/regr_{random_number}/original_preds.txt', 'w') as out:
                        out.write('FLIGHT ID | ALLFT_CI | ALLFT_PL\n')
                        for i in range(len(y_pred_original)):
                            out.write(f'{original_ids[i]} | {int(y_pred_original[i,0])} | {round(y_pred_original[i,1])}\n')

                    with open(f'{experiment}/regr_{random_number}/predicted_preds.txt', 'w') as out:
                        out.write('FLIGHT ID | PREDICTED_CI | PREDICTED_PL\n')
                        for i in range(len(y_pred_predicted)):
                            out.write(f'{predicted_ids[i]} | {int(y_pred_predicted[i,0])} | {round(y_pred_predicted[i,1])}\n')

                    with open(f'{experiment}/regr_{random_number}/dynamo_tests.txt', 'w') as out:
                        out.write('FLIGHT ID | REAL_CI | REAL_PL | CI_OUT | PL_OUT\n')
                        for i in range(len(y_pred)):
                            out.write(f'{test_ids[i]} | {int(y_test[i,0])} | {round(y_test[i,1])}| {int(y_pred[i,0])}| {round(y_pred[i,1])}\n')
    
if __name__ == "__main__":
    main()
    



