# import argparse
# import platform
import math
# from torch import nn
# from torch.nn import functional as F
# from torch import optim
# import random
# from collections import deque, namedtuple
# import time
# from datetime import datetime
# import gym
# import torch
# import csv
# import pickle
# import plotly
# import plotly.graph_objs as go
# from torch import multiprocessing as mp
# import os
# import time
# import re
# import fnmatch


# import os
# import time
# from math import sqrt
# from torch.distributions import Categorical
import numpy as np
# from gym import spaces
# # import matplotlib.pyplot as plt
# import random
# # from sklearn.model_selection import train_test_split

# ##################################################
# from collections import deque
# # import dqn_dvh_external_network
# import h5sparse
# import h5py
# from scipy.sparse import vstack
# from typing import List
# from scipy.sparse import csr_matrix
# import numpy.linalg as LA
# import time
#################################################

# import logging
import matplotlib.pyplot as plt
# from data_prep_parth_complete_onceagain import loadDoseMatrix,loadMask,ProcessDmat
# plt.switch_backend('agg')

epoch = 45999
pdose = 1
Volume_axis = []
B_Volume_axis = []
R_Volume_axis = []
dose_axis_PTV = []
dose_axis_BLA = []
dose_axis_REC = []
# data_result_path='/data/mainul1/results_CORS/FGSM_AttackNone/scratch6_30StepsNewParamenters3/dataWithPlanscoreRun/'
# data_result_path_Perturbed = '/home/mainul/DQN/PerturbedDVH12/'
# data_result_path_Unperturbed = '/home/mainul/DQN/UnperturbedDVH12/'
# data_result_path_Perturbed = "/data2/mainul/DQNFGSMUTSWall/PerturbedDVH12/"
# data_result_path_Unperturbed = "/data2/mainul/DQNFGSMUTSWall/UnperturbedDVH12/"
data_result_path_Perturbed = "/data2/mainul/DQNFGSMUTSWallPaper/PerturbedDVH12/"
data_result_path_Unperturbed = "/data2/mainul/DQNFGSMUTSWallPaper/UnperturbedDVH12/"

# data_result_path="/home/mainul/DQN/Results/"
# Y = np.load("/home/mainul/DQN/Results/12xDVHXInitial.npy")
# print(Y.shape)
# print(Y)
# The 23 lines are for patientID 1
# Y = np.load(data_result_path+str(1)+'xDVHY' + str(1)  + 'step' + str(0)+'.npy')
Y = np.load(data_result_path_Perturbed+'010perturbed1.npy')
YFull = np.load('/data2/mainul/DataAndGraphDQN/FGSMPaper/PaperRepresantativeUnperturbedFull.npy')

# This block is for DQN=================================
print('Y', Y)
# Y = Y.flatten()
Y = np.reshape(Y, (3, 100))

print('YUnTransposed',Y)
Y = np.transpose(Y)
print('YTransposed',Y)
# # #next block is for acer==========================================================
# print(Y.shape)
# print(Y)
# Y = np.reshape(Y, (100, 3), order='F')
# print(Y)
# #=================================================================================

# Extract columns to retrieve original variables
y_ptv = Y[:, 0]
y_bladder = Y[:, 1]
y_rectum = Y[:, 2]

y_ptvFull = YFull[:,6]
y_bladderFull = YFull[:,7]
y_rectumFull = YFull[:,8]


y_ptvFirst = YFull[:,6][0:96]
y_bladderFirst = YFull[:,7][0:59]
y_rectumFirst = YFull[:,8][0:59]

# y_ptv = y_ptv/max(y_ptv)
# y_bladder = y_bladder/max(y_bladder)
# y_rectum = y_rectum/max(y_rectum)
x_ptvFull = YFull[:,9]
x_bladderFull = YFull[:,10]
x_rectumFull = YFull[:,11]


x_ptvFirst = YFull[:,9][0:96]
x_bladderFirst = YFull[:,10][0:59]
x_rectumFirst = YFull[:,11][0:59]


# print(x_ptvFull)
# print(x_bladderFirst.shape)
# index = np.where(x_bladderFirst>=0.6)
# print(max(x_rectumFull))
# print(max(x_bladderFull))
# print(x_bladderFirst[58])

# Trial
x_ptv = np.linspace(1, 1.15, 100)
x_bladder = np.linspace(0.6, 1.1, 100)
x_rectum = np.linspace(0.6, 1.1, 100)

# x_ptv = np.linspace(0, max(y_ptv), 100)
# x_bladder = np.linspace(0, max(y_bladder), 100)
# x_rectum = np.linspace(0, max(y_rectum), 100)

x_ptv_plot= np.concatenate((x_ptvFirst, x_ptv))
x_bladder_plot= np.concatenate((x_bladderFirst, x_bladder))
x_rectum_plot= np.concatenate((x_rectumFirst, x_rectum))

y_ptv_plot= np.concatenate((y_ptvFirst, y_ptv))
y_bladder_plot= np.concatenate((y_bladderFirst, y_bladder))
y_rectum_plot= np.concatenate((y_rectumFirst, y_rectum))

y_ptv_plot= y_ptv_plot*100
y_bladder_plot= y_bladder_plot*100
y_rectum_plot= y_rectum_plot*100

x_ptv_plot= x_ptv_plot*100
x_bladder_plot= x_bladder_plot*100
x_rectum_plot= x_rectum_plot*100



# Create a plot
plt.figure(figsize=(10, 9))

# Plot the three datasets against the array of points
plt.plot(x_ptv_plot, y_ptv_plot, label='PTV0_p', color='red', linestyle='--')
plt.plot(x_bladder_plot, y_bladder_plot, label='Bla0_p', color='green', linestyle='--' )
plt.plot(x_rectum_plot, y_rectum_plot, label='Rec0_p', color='blue', linestyle='--')

# The next 22 lines are for patientID 0
# Y = np.load(data_result_path+str(0)+'xDVHY' + str(1)  + 'step' + str(0)+'.npy')
Y = np.load(data_result_path_Unperturbed+"010unperturbed0.npy")
# This block is for DQN=================================
Y = Y.flatten()
Y = np.reshape(Y, (3, 100))
print('YUnTransposed',Y)
Y = np.transpose(Y)
print('YTransposed',Y)
# #next block is for acer==========================================================
# Y = np.reshape(Y, (100, 3), order='F')
# print(Y.shape)
# #====================================================

# Y = np.reshape(Y, (100, 3), order='F')

# Extract columns to retrieve original variables
y_ptv = Y[:, 0]
y_bladder = Y[:, 1]
y_rectum = Y[:, 2]

y_ptv_plot= np.concatenate((y_ptvFirst, y_ptv))
y_bladder_plot= np.concatenate((y_bladderFirst, y_bladder))
y_rectum_plot= np.concatenate((y_rectumFirst, y_rectum))

y_ptv_plot= y_ptv_plot*100
y_bladder_plot= y_bladder_plot*100
y_rectum_plot= y_rectum_plot*100
# y_ptv = y_ptv/max(y_ptv)
# y_bladder = y_bladder/max(y_bladder)
# y_rectum = y_rectum/max(y_rectum)

# x_ptv = Y[:, 9]
# x_bladder = Y[:, 10]
# x_rectum = Y[:, 11]


# Create a plot
# plt.figure(figsize=(10, 6))

# Plot the three datasets against the array of points
plt.plot(x_ptv_plot, y_ptv_plot, label='PTV0', color='red', linestyle='-')
plt.plot(x_bladder_plot, y_bladder_plot, label='Bla0', color='green', linestyle='-' )
plt.plot(x_rectum_plot, y_rectum_plot, label='Rec0', color='blue', linestyle='-')

plt.xlim(0,120)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)


# Add titles and labels
# plt.title('Adversarial Perturbation', fontsize = 14)
plt.xlabel('Relative Dose (%)', fontsize = 30)
plt.ylabel('Relative Volume (%)', fontsize = 30)
plt.legend(loc='best', fontsize = 30)

plt.tight_layout()
plt.savefig('/data2/mainul/DataAndGraphDQN/FGSMPaper/AdversarialPerturbationFull.png', dpi = 1200)
plt.show()