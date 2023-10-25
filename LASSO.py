from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import copy
import numpy as np
from numpy import random
from csv import writer
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
pd.set_option('display.max_columns', None)
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

#==================================================================
#=======================PARAMETERS & MODE==========================
#==================================================================

#parameters
a = 0.0000001
mi = 100000

#mode
#pipeline 2 (51 features)
mode = 'dsciqrvh-all'

#pipeline 1 (69 features)
#mode = 'dsciqr-pip1'

if mode.split("-")[0] == 'dsciqrvh':
	path = "data/FDR_dsc_iqr_dot.csv"
	path_pred_ps = 'data/PredictedSplit_dsc_iqr_dot.csv'
	path_pred_o  ='data/ALLFT_dsc_iqr_dot.csv'

if mode.split("-")[-1] == 'pip1':
	if mode.split("-")[0] == 'dsciqr':
		path = "data/stats-v6-pip1-iqr.csv"

boxplot_data = {"CI_train": [], "PL_train": [], "CI_test": [], "PL_test": []}

# Read the flight id list
id_list_ = open("train_id.txt").readlines()
id_list = []
for x in id_list_:
	y = x.replace("\n", "")
	y = y.split("_")[1]
	id_list.append(y)

test_list_ = open("test_ids.txt").readlines()
test_list = []
for x in test_list_:
	y = x.replace("\n", "")
	test_list.append(y)

# Open the csv with the features and drop NaN values
df = pd.read_csv(path)
df = df.dropna().reset_index(drop=True)
print(df['ID'])
df['ID'] = df['ID'].astype(str)

boxplot_data = {"CI_train": [], "PL_train": [], "CI_test": [], "PL_test": []}

# Read the flight id list
id_list_ = open("train_id.txt").readlines()
id_list = []
for x in id_list_:
	y = x.replace("\n", "")
	y = y.split("_")[1]
	id_list.append(y)

test_list_ = open("test_ids.txt").readlines()
test_list = []
for x in test_list_:
	y = x.replace("\n", "")
	test_list.append(y)

# Open the csv with the features and drop NaN values
df = pd.read_csv(path)
df = df.dropna().reset_index(drop=True)
df['ID'] = df['ID'].astype(str)

# Min-Max CI-LW (Known)
max_ci = df.loc[:, "CI"].max()
min_ci = df.loc[:, "CI"].min()
max_lw = df.loc[:, "LW"].max()
min_lw = df.loc[:, "LW"].min()

# Split train-test set based on the flight id list
df_x_train = df.loc[df['ID'].isin(id_list)].drop(['CI', 'LW'], axis=1).reset_index(drop=True)
df_x_test = df.loc[df['ID'].isin(test_list)].drop(['CI', 'LW'], axis=1).reset_index(drop=True)
df_y_train = df.loc[df['ID'].isin(id_list)][['CI', 'LW']].reset_index(drop=True)
df_y_test = df.loc[df['ID'].isin(test_list)][['CI', 'LW']].reset_index(drop=True)
df_id_test = df.loc[df['ID'].isin(test_list)][['ID']].reset_index(drop=True)

# Copies of the DataFrames
df_copy = df.copy().drop(['CI', 'LW', 'ID'], axis=1)
df_hidden_vars = df.copy()[['CI', 'LW']]
df_x_train_norm = df_x_train.copy().drop(['ID'], axis=1)
df_x_test_norm = df_x_test.copy().drop(['ID'], axis=1)
df_y_train_norm = df_y_train.copy(deep=True)
df_y_test_norm = df_y_test.copy(deep=True)	


# Normalization (Min-Max)
df_x_train_norm = (df_x_train_norm - df_copy.min()) / (df_copy.max() - df_copy.min())

df_x_test_norm = (df_x_test_norm - df_copy.min()) / (df_copy.max() - df_copy.min())
df_y_train_norm = (df_y_train_norm - df_hidden_vars.min()) / (df_hidden_vars.max() - df_hidden_vars.min())
df_y_test_norm = (df_y_test_norm - df_hidden_vars.min()) / (df_hidden_vars.max() - df_hidden_vars.min())
if mode.split("-")[-1] != 'pip1':
	df_x_pred_ps_norm = (df_x_pred_ps_norm - df_copy.min()) / (df_copy.max() - df_copy.min())
	df_x_pred_o_norm = (df_x_pred_o_norm - df_copy.min()) / (df_copy.max() - df_copy.min())

# DataFrames to Numpy
x_train_norm = df_x_train_norm.to_numpy().astype(np.float32)
y_train_norm = df_y_train_norm.to_numpy().astype(np.float32)
y_test_norm = df_y_test_norm.to_numpy().astype(np.float32)
x_test_norm = df_x_test_norm.to_numpy().astype(np.float32)
if mode.split("-")[-1] != 'pip1':
	x_pred_o_norm = df_x_pred_o_norm.to_numpy().astype(np.float32)
	x_pred_ps_norm = df_x_pred_ps_norm.to_numpy().astype(np.float32)
y_test = df_y_test.to_numpy().astype(np.float32)
y_train = df_y_train.to_numpy().astype(np.float32)

print(x_train_norm.shape)
print(x_test_norm.shape)

input_size = len(x_train_norm[0])

with open('lasso_results_final.csv', 'a') as f_object:
	writer_object = writer(f_object)
	header = ["Stats file", "Mode", "random_id", "alpha", "max_iter", "CI MAE TRAIN", "CI STD TRAIN", "PL MAE TRAIN", "PL STD TRAIN", "CI MAE TEST", "CI STD TEST", "PL MAE TEST", "PL STD TEST", "time needed"]
	writer_object.writerow(header)

for i in range(1):
	start = timer()
	random_id = random.randint(100000)

	params = {'alpha': a,
	'max_iter': mi}

	gs = MultiOutputRegressor(Lasso(**params), n_jobs = -1)
	gs.fit(x_train_norm, y_train_norm)

	end = timer()

	pickle.dump(gs, open("pretrained_models/"+f"lasso_{random_id}", 'wb'))

	y_pred = gs.predict(x_train_norm)

	y_pred[:,0] = y_pred[:,0] * (max_ci - min_ci) + min_ci
	y_pred[:,1] = y_pred[:,1] * (max_lw - min_lw) + min_lw

	y_pred[:,0] = y_pred[:,0].clip(min_ci, max_ci)
	y_pred[:,0] = np.round(y_pred[:,0])

	y_pred[:,1] = y_pred[:,1].clip(min_lw, max_lw)
	y_pred[:,1] = np.round(y_pred[:,1],1)

	mae_CI_train = mean_absolute_error(y_train[:,0], y_pred[:,0])
	mae_PL_train = mean_absolute_error(y_train[:,1], y_pred[:,1])

	std_CI_train = []
	std_PL_train = []

	for i in range(len(y_pred)):

		CI_loss_train = np.abs(y_train[i,0] - y_pred[i,0])
		PL_loss_train = np.abs(y_train[i,1] - y_pred[i,1])

		boxplot_data["CI_train"].append(CI_loss_train)
		boxplot_data["PL_train"].append(PL_loss_train)

		std_CI_train.append(CI_loss_train)
		std_PL_train.append(PL_loss_train)

	std_CI_train = np.std(std_CI_train)
	std_PL_train = np.std(std_PL_train)

	# TESTING
	y_pred = gs.predict(x_test_norm)

	y_pred[:,0] = y_pred[:,0] * (max_ci - min_ci) + min_ci
	y_pred[:,1] = y_pred[:,1] * (max_lw - min_lw) + min_lw

	y_pred[:,0] = y_pred[:,0].clip(min_ci, max_ci)
	y_pred[:,0] = np.round(y_pred[:,0])

	y_pred[:,1] = y_pred[:,1].clip(min_lw, max_lw)
	y_pred[:,1] = np.round(y_pred[:,1],1)

	mae_CI_test = mean_absolute_error(y_test[:,0], y_pred[:,0])
	mae_PL_test = mean_absolute_error(y_test[:,1], y_pred[:,1])

	std_CI_test = []
	std_PL_test = []

	for i in range(len(y_pred)):

		CI_loss_test = np.abs(y_test[i,0] - y_pred[i,0])
		PL_loss_test = np.abs(y_test[i,1] - y_pred[i,1])

		boxplot_data["CI_test"].append(CI_loss_test)
		boxplot_data["PL_test"].append(PL_loss_test)

		std_CI_test.append(CI_loss_test)
		std_PL_test.append(PL_loss_test)

	std_CI_test = np.std(std_CI_test)
	std_PL_test = np.std(std_PL_test)

	if mode.split("-")[-1] != 'pip1':
		x_pred_o_norm = np.clip(x_pred_o_norm, 0, 1)
		x_pred_ps_norm = np.clip(x_pred_ps_norm, 0, 1)

		with open(f'predictions/lasso-predictions-{mode}-{random_id}.txt', 'w') as f_:
			f_.write("flight_id, ALLFT_CI, ALLFT_LW, PREDICTED_CI, PREDICTED_LW\n")
			original_feature_id = list(zip(x_pred_o_norm, pred_df_names_o))
			predicted_feature_id = list(zip(x_pred_ps_norm, pred_df_names_ps))

			for i in range(len(original_feature_id)):

				(x_original, fid_original) = original_feature_id[i]
				(x_predicted, fid_predicted) = predicted_feature_id[i]

				fid_original = fid_original.split("_")[1]
				fid_predicted = fid_predicted.split("_")[1]

				prediction_original = gs.predict(x_original.reshape(1, -1))
				prediction_original = prediction_original[0]

				prediction_predicted = gs.predict(x_predicted.reshape(1, -1))
				prediction_predicted = prediction_predicted[0]

				prediction_original[0] = (prediction_original[0] * (max_ci - min_ci)) + min_ci
				prediction_original[1] = (prediction_original[1] * (max_lw - min_lw)) + min_lw

				prediction_original[0] = prediction_original[0].clip(min_ci, max_ci)
				prediction_original[0] = np.round(prediction_original[0])

				prediction_original[1] = prediction_original[1].clip(min_lw, max_lw)
				prediction_original[1] = np.round(prediction_original[1],1)


				prediction_predicted[0] = (prediction_predicted[0] * (max_ci - min_ci)) + min_ci
				prediction_predicted[1] = (prediction_predicted[1] * (max_lw - min_lw)) + min_lw

				prediction_predicted[0] = prediction_predicted[0].clip(min_ci, max_ci)
				prediction_predicted[0] = np.round(prediction_predicted[0])

				prediction_predicted[1] = prediction_predicted[1].clip(min_lw, max_lw)
				prediction_predicted[1] = np.round(prediction_predicted[1],1)

				string = str(fid_original)+", "+str(prediction_original[0])+", "+str(prediction_original[1])+", "+str(prediction_predicted[0])+", "+str(prediction_predicted[1])+'\n'
				
				f_.write(string)

	with open(f'predictions/lasso-pred-{mode}-{random_id}.txt', 'w') as f_:
		f_.write("FLIGHT ID, TRUE CI, TRUE PL, PRED CI, PRED PL\n")
		for (x, fid, y) in zip(x_test_norm, df_id_test.values.tolist(), df_y_test.values.tolist()):

			prediction = gs.predict(x.reshape(1,-1))

			prediction[0][0] = prediction[0][0] * (max_ci - min_ci) + min_ci
			prediction[0][1] = prediction[0][1] * (max_lw - min_lw) + min_lw

			prediction[0][0] = prediction[0][0].clip(min_ci, max_ci)
			prediction[0][0] = np.round(prediction[0][0])

			prediction[0][1] = prediction[0][1].clip(min_lw, max_lw)
			prediction[0][1] = np.round(prediction[0][1],1)

			string = fid[0]+", "+str(y[0])+", "+str(y[1])+", "+str(prediction[0][0])+", "+str(prediction[0][1])+"\n"
			f_.write(string)


	with open('lasso_results_final.csv', 'a') as f_object:
		writer_object = writer(f_object)

		row = [path, mode, random_id, a, mi, mae_CI_train, std_CI_train, mae_PL_train, std_PL_train, mae_CI_test, std_CI_test, mae_PL_test, std_PL_test,  "{:.2f}".format(float(end-start)/60.0)]
		writer_object.writerow(row)

		f_object.close()

pickle.dump( boxplot_data, open("boxplot_data/boxplot_data_lasso_"+str(random_id)+".p", "wb" ))