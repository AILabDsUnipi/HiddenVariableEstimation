import matplotlib.pyplot as plt
import csv
import pandas as pd
import pickle
import numpy as np

plt.rcParams.update({'font.size': 22})

ci_dict = {}
lw_dict = {}

path = '/boxplot_data/'

#==================================================================
#===================LOAD THE DESIRED MODELS========================
#==================================================================

lasso_data = pickle.load( open( path+"XXXX.p", "rb" ) )
dnn_data = pickle.load( open( path+"XXXX.p", "rb" ) )
svr_data = pickle.load( open( path+"XXXX.p", "rb" ) )
krr_data = pickle.load( open( path+"XXXX.p", "rb" ) )
gbm_data = pickle.load( open( path+"XXXX.p", "rb" ) )

ci_dict["GBM"] = gbm_data["CI_test"]
ci_dict["NN"] = dnn_data["CI_test"]
ci_dict["KRR"] = krr_data["CI_test"]
ci_dict["SVR"] = svr_data["CI_test"]
ci_dict["LASSO"] = lasso_data["CI_test"]

lasso_ci_per = np.percentile(ci_dict["LASSO"], 75) - np.percentile(ci_dict["LASSO"], 25)
lasso_ci_range = np.max(ci_dict["LASSO"]) - np.min(ci_dict["LASSO"])
dnn_ci_per = np.percentile(ci_dict["NN"], 75) - np.percentile(ci_dict["NN"], 25)
dnn_ci_range = np.max(ci_dict["NN"]) - np.min(ci_dict["NN"])
svr_ci_per = np.percentile(ci_dict["SVR"], 75) - np.percentile(ci_dict["SVR"], 25)
svr_ci_range = np.max(ci_dict["SVR"]) - np.min(ci_dict["SVR"])
krr_ci_per = np.percentile(ci_dict["KRR"], 75) - np.percentile(ci_dict["KRR"], 25)
krr_ci_range = np.max(ci_dict["KRR"]) - np.min(ci_dict["KRR"])
gbm_ci_per = np.percentile(ci_dict["GBM"], 75) - np.percentile(ci_dict["GBM"], 25)
gbm_ci_range = np.max(ci_dict["GBM"]) - np.min(ci_dict["GBM"])

lw_dict["GBM"] = gbm_data["PL_test"]
lw_dict["NN"] = dnn_data["PL_test"]
lw_dict["KRR"] = krr_data["PL_test"]
lw_dict["SVR"] = svr_data["PL_test"]
lw_dict["LASSO"] = lasso_data["PL_test"]

lasso_lw_per = np.percentile(lw_dict["LASSO"], 75) - np.percentile(lw_dict["LASSO"], 25)
lasso_lw_range = np.max(lw_dict["LASSO"]) - np.min(lw_dict["LASSO"])
dnn_lw_per = np.percentile(lw_dict["NN"], 75) - np.percentile(lw_dict["NN"], 25)
dnn_lw_range = np.max(lw_dict["NN"]) - np.min(lw_dict["NN"])
svr_lw_per = np.percentile(lw_dict["SVR"], 75) - np.percentile(lw_dict["SVR"], 25)
svr_lw_range = np.max(lw_dict["SVR"]) - np.min(lw_dict["SVR"])
krr_lw_per = np.percentile(lw_dict["KRR"], 75) - np.percentile(lw_dict["KRR"], 25)
krr_lw_range = np.max(lw_dict["KRR"]) - np.min(lw_dict["KRR"])
gbm_lw_per = np.percentile(lw_dict["GBM"], 75) - np.percentile(lw_dict["GBM"], 25)
gbm_lw_range = np.max(lw_dict["GBM"]) - np.min(lw_dict["GBM"])



fig, ax = plt.subplots()
ax.boxplot(ci_dict.values())
ax.set_xticklabels(ci_dict.keys())

plt.savefig('CI_MAE_BOXPLOT_COMPARE.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(ci_dict["NN"])

plt.savefig('CI_MAE_BOXPLOT_NN.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(ci_dict["SVR"])

plt.savefig('CI_MAE_BOXPLOT_SVR.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(ci_dict["KRR"])

plt.savefig('CI_MAE_BOXPLOT_KRR.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(ci_dict["GBM"])

plt.savefig('CI_MAE_BOXPLOT_GBM.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(ci_dict["LASSO"])

plt.savefig('CI_MAE_BOXPLOT_LASSO.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(lw_dict["NN"])

plt.savefig('LW_MAE_BOXPLOT_NN.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(lw_dict.values())
ax.set_xticklabels(lw_dict.keys())

plt.savefig('PL_MAE_BOXPLOT_COMPARE.png')


fig, ax = plt.subplots()
ax.boxplot(lw_dict["SVR"])

plt.savefig('LW_MAE_BOXPLOT_SVR.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(lw_dict["KRR"])

plt.savefig('LW_MAE_BOXPLOT_KRR.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(lw_dict["GBM"])

plt.savefig('LW_MAE_BOXPLOT_GBM.png')

plt.clf()

fig, ax = plt.subplots()
ax.boxplot(lw_dict["LASSO"])

plt.savefig('LW_MAE_BOXPLOT_LASSO.png')

plt.clf()


with open('percentiles_ranges.csv', 'w', encoding='UTF8', newline='') as fstar:
	writer = csv.writer(fstar)
	header = ["Method", "CI Percentile 75-25", "CI Range (Max-Min)", "LW Percentile 75-25", "LW Range (Max-Min)"]
	data1 = ["LASSO", lasso_ci_per, lasso_ci_range, lasso_lw_per, lasso_lw_range]
	data2 = ["NN", dnn_ci_per, dnn_ci_range, dnn_lw_per, dnn_lw_range]
	data3 = ["SVR", svr_ci_per, svr_ci_range, svr_lw_per, svr_lw_range]
	data4 = ["KRR", krr_ci_per, krr_ci_range, krr_lw_per, krr_lw_range]
	data5 = ["GBM", gbm_ci_per, gbm_ci_range, gbm_lw_per, gbm_lw_range]

	writer.writerow(header)
	writer.writerow(data1)
	writer.writerow(data2)
	writer.writerow(data3)
	writer.writerow(data4)
	writer.writerow(data5)


