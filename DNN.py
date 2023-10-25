from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import numpy as np
from numpy import random
from csv import writer
import pandas as pd
pd.set_option('display.max_columns', None)
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import torch as T
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random as r

#==================================================================
#=======================PARAMETERS & MODE==========================
#==================================================================

#parameters
wd = 0.00001
drop = None
learning_rate = 0.0001
num_n1 = 100
num_n2 = 50
num_epochs = 50000
activation = 'linear'

#modes
#pipeline 2 (51 features)
mode = 'dsciqrvh-all'

#pipeline 1 (69 features)
#mode = 'dsciqr-pip1'

#==================================================================
#==================================================================
#==================================================================

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
y_test = df_y_test.to_numpy().astype(np.float32)
y_train = df_y_train.to_numpy().astype(np.float32)
if mode.split("-")[-1] != 'pip1':
	x_pred_o_norm = df_x_pred_o_norm.to_numpy().astype(np.float32)
	x_pred_ps_norm = df_x_pred_ps_norm.to_numpy().astype(np.float32)

input_size = len(x_train_norm[0])

with open('dnn_results_final.csv', 'a') as f_object:
	writer_object = writer(f_object)
	header = ["Stats file", "Mode", "random_id", "Output Activation",  "Number of Neurons", "learning_rate", "Epochs", "Dropout"," Weight Decay", "CI MAE TRAIN", "CI STD TRAIN", "PL MAE TRAIN", "PL STD TRAIN", "CI MAE TEST", "CI STD TEST", "PL MAE TEST", "PL STD TEST", "time needed"]
	writer_object.writerow(header)

for i in range(1):

	start = timer()
	random_id = random.randint(100000)

	model = NN(input_size, 2, num_n1, num_n2)
	optimizer = T.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=wd)
	criterion = nn.MSELoss()
	model.train()

	# TRAINING
	for epoch in range(num_epochs):

		inputs = Variable(T.from_numpy(x_train_norm)) #convert numpy array to torch tensor
		targets = Variable(T.from_numpy(y_train_norm)) #convert numpy array to torch tensor
		outputs = model(inputs) #output
		loss = criterion(outputs,targets) #loss function

		optimizer.zero_grad() #gradient
		loss.backward() #backward propogation
		optimizer.step() #1-step optimization(gradeint descent)

		if(epoch % 500 == 0):
			with T.no_grad():
				inputs_test = Variable(T.from_numpy(x_test_norm))
				targets_test = Variable(T.from_numpy(y_test_norm))
				pred = model(inputs_test).numpy()

				pred[:,0] = pred[:,0] * (max_ci - min_ci) + min_ci
				pred[:,1] = pred[:,1] * (max_lw - min_lw) + min_lw

				pred[:,0] = pred[:,0].clip(min_ci, max_ci)
				pred[:,0] = np.round(pred[:,0])

				pred[:,1] = pred[:,1].clip(min_lw, max_lw)
				pred[:,1] = np.round(pred[:,1],1)

				mae_CI = mean_absolute_error(y_test[:,0], pred[:,0])
				mae_PL = mean_absolute_error(y_test[:,1], pred[:,1])

				print("Epoch: "+str(epoch)+", Train Loss: "+str(loss.item())+", Test CI Loss: "+str(mae_CI)+", Test PL Loss: "+str(mae_PL))

			if mode.split("-")[-1] != 'pip1':
				for (x, fid) in zip(x_pred_o_norm, pred_df_names_o):

					fid = fid.split("_")[1]

					x_input = Variable(T.from_numpy(x))

					prediction = model(x_input).detach().numpy()
					prediction[0] = prediction[0] * (max_ci - min_ci) + min_ci
					prediction[1] = prediction[1] * (max_lw - min_lw) + min_lw

					prediction[0] = prediction[0].clip(min_ci, max_ci)
					prediction[0] = np.round(prediction[0])

					prediction[1] = prediction[1].clip(min_lw, max_lw)
					prediction[1] = np.round(prediction[1],1)

					string = str(fid)+", "+str(prediction[0])+", "+str(prediction[1])
					print(string)


	T.save(model.state_dict(), "pretrained_models/"+f"dnn_{random_id}")

	end = timer()

	model.eval()

	train_inputs = Variable(T.from_numpy(x_train_norm))
	y_pred = model(train_inputs).detach().numpy()

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
	train_inputs = Variable(T.from_numpy(x_test_norm))
	y_pred = model(train_inputs).detach().numpy()

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

		with open(f'predictions/dnn-predictions-{mode}-{random_id}.txt', 'w') as f_:
			f_.write("flight_id, ALLFT_CI, ALLFT_LW, PREDICTED_CI, PREDICTED_LW\n")
			original_feature_id = list(zip(x_pred_o_norm, pred_df_names_o))
			predicted_feature_id = list(zip(x_pred_ps_norm, pred_df_names_ps))
			print(len(original_feature_id))
			print(len(predicted_feature_id))
			for i in range(len(original_feature_id)):

				(x_original, fid_original) = original_feature_id[i]
				(x_predicted, fid_predicted) = predicted_feature_id[i]

				fid_original = fid_original.split("_")[1]
				fid_predicted = fid_predicted.split("_")[1]

				x_input = Variable(T.from_numpy(x_original))
				prediction_original = model(x_input).detach().numpy()

				x_input = Variable(T.from_numpy(x_predicted))
				prediction_predicted = model(x_input).detach().numpy()

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


	with open(f'predictions/dnn-pred-{mode}-{random_id}.txt', 'w') as f_:
		f_.write("FLIGHT ID, TRUE CI, TRUE PL, PRED CI, PRED PL\n")

		for (x, fid, y) in zip(x_test_norm, df_id_test.values.tolist(), df_y_test.values.tolist()):

			fid = fid[0]
			inputs = Variable(T.from_numpy(x.reshape(1,-1)))
			prediction = model(inputs).detach().numpy()
			prediction = prediction[0]
			prediction[0] = prediction[0] * (max_ci - min_ci) + min_ci
			prediction[1] = prediction[1] * (max_lw - min_lw) + min_lw

			prediction[0] = prediction[0].clip(min_ci, max_ci)
			prediction[0] = np.round(prediction[0])

			prediction[1] = prediction[1].clip(min_lw, max_lw)
			prediction[1] = np.round(prediction[1],1)

			string = fid+", "+str(y[0])+", "+str(y[1])+", "+str(prediction[0])+", "+str(prediction[1])+"\n"
			f_.write(string)


	with open('dnn_results_final.csv', 'a') as f_object:
		writer_object = writer(f_object)

		tag = f"{num_n1}_{num_n2}"
		row = [path, mode, random_id, activation, tag, learning_rate, num_epochs, drop, wd, mae_CI_train, std_CI_train, mae_PL_train, std_PL_train, mae_CI_test, std_CI_test, mae_PL_test, std_PL_test,  "{:.2f}".format(float(end-start)/60.0)]
		writer_object.writerow(row)

		f_object.close()

pickle.dump( boxplot_data, open("boxplot_data/boxplot_data_dnn_"+str(random_id)+".p", "wb" ))