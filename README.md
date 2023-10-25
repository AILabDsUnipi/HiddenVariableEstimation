
# 0 FILES  


## 0.0 [Training Scripts are:]  

-1 DNN.py (inp:features)  
-2 GBM.py (inp:features)  
-3 KRR.py (inp:features)  
-4 LASSO.py (inp:features)  
-5 SVR.py (inp:features)  
-6 LSTM.py (inp:trajectory)  
-7 VAE.py + VAE_REGRESSION.py (inp:trajectory)  

all 8 scripts have their hyperparameter & mode selection
in the indicated part inside the code.  

## 0.1 [Training scripts' outputs are:]  


for (0,1,2,4,5)  

1. A CSV file that is appended, every run with
information about the completed run. This file
is generated at the root folder.  

2. A file for the boxplot data, which is stored under
'boxplot_data' folder which contains essential data
for boxplot generation later.  

3. A file that stores the trained model's information
of weights and architecture after the training is
complete, which is stored under the 'pretrained_models'
folder.  

4. Two prediction files, the big one is for the evaluation
set of the training set provided/used, the small one is 
for the Hidden Trajectories.  

for (6)  

1. A CSV file that is appended, every run with
information about the completed run. This file
is generated at the root folder.  

2. A folder that is created through the experiment that contains
all the important data that is produced through training and
testing.  
	* Boxplots, train and test predictions.  
	* In case of pipeline 2, where we can also test hidden trajectories
	 Hidden Original and Predicted_Split predictions are saved in .txt files.  


(Every single run is labeled with a different random ID,
between 0 and 10.000 for mapping convenience between models,
boxplot data, statistics and so on and so forth)  

for (7)  

The Variational Autoencoder works similarly to the LSTM method but it contains to scripts that have to run in order for it to work.  

Firstly, we run the VAE.py with the desired parameters in order to train the Variational Autoencoder, a respective folder is made that contains useful information, results and then encoder/decoder models.  

Then, based on the experiement folder produced, we run accordingly the VAE_REGRESSION.py to feed the GBM Regressor with the latent data from the respective VAE   model.  

There are subtle details that cannot be considered shortly in this README but with a fast parse of those files someone can understand easily the pipeline and   successfully run an experiment.  


### -- FILES ASSOCIATED WITH METHODS THAT TAKE FEATURES AS INPUT --  


## 0.2 [makeBoxPlots.py]


Inside this script, you can change the names of the models
that want you to produce boxplots for and the output is 
boxplot pdf-images that provide visual elaboration on the
statistics of the chosen models' performance (Testing MSE Loss)
along with an extra txt file which contains the boxplot statistical
information with numbers.  

## 0.3 [data (folder)]

Folder 'data' contains the configured datasets that our models use
as their input.  


## 0.4 [train & test id .txt files]


Are used in training scripts in order to discriminate trajectories
between train and test.  

### FILES ASSOCIATED WITH METHODS THAT TAKE TRAJECTORIES AS INPUT  

## 0.5 [(makeTrajectories6phases, makeTrajectories11phases, makeTestInputs).py]  

These 3 files are used to construct the input data for the family of methods
that use trajectories as input like (6) LSTM.py. For pipeline 2, pipeline 1 and
original and predicted_split input respectively. Please take in mind that makeTestInputs.py is not made dynamically to work for both original and predicted_split inputs, thus, minor adjustments are needed to work.  

## 0.6 [data_trajectories (folder)]  

Folder 'data_trajectories' contains the configured datasets that the family of trajectory input methods are using.  

# 1 Requirements & Technical Details  

## 1.0 [Module | Version]  

"We ensure compatibility with the following versions of modules"  

Python | 3.8.10  
torch | 1.11.0  
pandas | 1.3.3  
matplotlib | 3.4.3  
sklearn | 0.0.post1  
keras | 2.12.0  
tensorflow | 1.*  

## 1.1 [How to run]  


To run any of [LASSO, DNN, SVR, GBM, KRR].py learning scripts:  
	1. Configure the hyperparameters as you like as per instructed at section [0.0]  
	2. Open a cmd to HV_CODE directory and run "python3 [chosen_method].py"  
 
To run makeBoxPlots.py script which makes the boxplots:  
	1. You need to have atleast 1 pretrained_model of each method (because it produces comparative results)  
	2. Configure the models you choose as per instructed at section [0.2]  
	
## 1.2 [Technical Requirements]  

1. Scripts do not require gpu to run.  
2. All algorithms have low requirements except some extreme tunings at KRR and SVR
	where we found that a minimum of 16gb ram is needed.  
