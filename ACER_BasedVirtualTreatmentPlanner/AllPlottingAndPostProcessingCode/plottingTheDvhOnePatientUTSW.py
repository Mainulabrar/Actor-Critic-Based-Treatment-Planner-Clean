import argparse
import platform
import math
from torch import nn
from torch.nn import functional as F
from torch import optim
import random
from collections import deque, namedtuple
import time
from datetime import datetime
import gym
import torch
import csv
import pickle
import plotly
import plotly.graph_objs as go
from torch import multiprocessing as mp
import os
import time
import re
import fnmatch


import os
import time
from math import sqrt
from torch.distributions import Categorical
import numpy as np
from gym import spaces
# import matplotlib.pyplot as plt
import random
# from sklearn.model_selection import train_test_split

##################################################
from collections import deque
# import dqn_dvh_external_network
import h5sparse
import h5py
from scipy.sparse import vstack
from typing import List
from scipy.sparse import csr_matrix
import numpy.linalg as LA
import time
import os
#################################################

# import logging
import matplotlib.pyplot as plt
# from data_prep_parth_complete_onceagain import loadDoseMatrix,loadMask,ProcessDmat
# plt.switch_backend('agg')

# data_result_path = "/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/"
# data_result_path ='/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
data_result_path = "/data2/mainul/results_CORS1PaperDiffBeam/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/"
data_result_path = '/data4/mainul/MultiModalAI6Beam/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path='/data4/mainul/MultiModalAIwithoutTPP/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
# data_result_path = "/data2/mainul/results_CORS1PaperDiffBeam7/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/"
graphSavePath = data_result_path+f'FigureSavePath/'
os.makedirs(graphSavePath,exist_ok = True)


def maximum_step(patientid):
    data_path = data_result_path
    pattern_base = "{}xDVHY*step*.npy"
    pattern = pattern_base.format(patientid)

    # Use fnmatch to filter filenames based on the pattern
    files = fnmatch.filter(os.listdir(data_path), pattern)

    # Print the matching filenames
    #print(files)
    captured_numbers = []

    # Iterate through the matching filenames
    for filename in files:
        # Use regular expression to extract the desired part
        match = re.search(r'^(\d+)xDVHY(\d+)step(\d+).npy$', filename)
        
        # Check if a match is found
        if match:
            captured_numbers.append(int(match.group(3)))
        else:
            print("No match found for:", filename)

    # Find the maximum value for this list
    max_value = max(captured_numbers)
    return max_value

# data_result_path = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_path_result_2 = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3NewData1/figuresWithPlanScoresRun/'    

patient_num = ['027']
patient_num = ['008']
# patient_num = ['001']
# Plotting the DVH graphs
# print("Plotting the DVH graphs")

# # for patientid in range(patient_num):
#     # r = lambda: random.randint(0,255)
#     # color1 = '#%02x%02x%02x' % (r(),r(),r())
#     # print("Patient_ID",patientid)
#     # for iter1 in range(1):
#         # try:
#             # if (patientid == 0):                  
#                 # Y0 = np.load(data_result_path+str(patientid)+'xDVHY'  + str(epoch)  + 'iter' + str(iter1)+'.npy')
#                 # plt.plot(Y0[:, 3], Y0[:, 0], label =f'ptv{patientid}', color = color1)
#                 # # plt.plot(Y0[:, 4], Y0[:, 1], label =f'bladder{patientid}', color = color1)
#                 # # plt.plot(Y0[:, 5], Y0[:, 2], label =f'rectum{patientid}', color = color1)
#                 # plt.text(Y0[:, 3][999],Y0[:, 1][999],f'{patientid}',fontsize=0.00005)
#                 # print(f'Figure{episode_length} done')
#             # else:
#                 # Y = np.load(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'iter' + str(iter1)+'.npy')
#                 # plt.plot(Y[:, 3], Y0[:, 0], label =f'ptv{patientid}', color = color1)
#                 # # plt.plot(Y[:, 4], Y0[:, 1], label =f'bladder{patientid}', color = color1)
#                 # # plt.plot(Y[:, 5], Y0[:, 2], label =f'rectum{patientid}', color = color1)
#                 # plt.text(Y[:, 3][999],Y0[:, 1][999],f'{patientid}',fontsize=0.00005)
#                 # print(f'Figure{episode_length} done')
#         # except Exception as e:
#             # print(e)
# # plt.legend(fontsize='5')
# # plt.title(str(patientid)+ 'DVH'+ str(epoch) + 'iter' + str(iter1))
# # plt.savefig(data_result_path2 + str(patientid) + 'DVH' + str(epoch)+ 'iter' + str(iter1) +'PTV'+ '.png',dpi=2000)
# # plt.close()
# epoch = 190001
# arr1 = []
# arr2 = []
# arr3 = []
# arr4 = []
# for patientid in range(patient_num):
#     arr1.append(patientid)
#     print("Patient_ID",patientid)
#     for iter1 in range(1):
#         try:                             
#             Y = np.load(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(iter1)+'.npy')
#             arr2.append(Y[:, 0])
#             arr3.append(Y[:, 1])
#             arr4.append(Y[:, 2])
#         except Exception as e:
#             print(e)
# array2 = np.array([arr1,arr2,arr3,arr4])
# print(array2)
# print("PTV",np.sort(arr2))
# print("Bladder",np.sort(arr3))
# print("Rectum",np.sort(arr4))
# plt.plot(arr1,arr2,label = 'PTV')
# for i in arr2:
#     plt.axhline(y = i,ls = '--', lw = 0.4, color = 'purple')
# plt.xticks(arr1)
# plt.legend(fontsize='5')
# plt.savefig(data_result_path2  + 'Endpoint_comparision'+'PTV' + str(epoch)+ 'iter' + str(iter1) +'.png',dpi=1200)
# plt.close()

# plt.plot(arr1,arr3,label = 'Bladder')
# for i in arr3:
#     plt.axhline(y = i,ls = '--', lw = 0.4, color = 'purple')
# plt.xticks(arr1)
# plt.legend(fontsize='5')
# plt.savefig(data_result_path2  + 'Endpoint_comparision'+'Bladder' + str(epoch)+ 'iter' + str(iter1) +'.png',dpi=1200)
# plt.close()

# plt.plot(arr1,arr4,label = 'Rectum')
# for i in arr4:
#     plt.axhline(y = i,ls = '--', lw = 0.4, color = 'purple')
# plt.xticks(arr1)
# plt.legend(fontsize='5')
# plt.savefig(data_result_path2  + 'Endpoint_comparision'+'Rectum' + str(epoch)+ 'iter' + str(iter1) +'.png',dpi=1200)

# plt.close()
# # for initial DVH
# epoch = 190001
# iter1 = 0
# pdose = 1
# Volume_axis = [] 
# dose_axis = np.linspace(0, pdose * 1.15, 100)
# for patientid in range(patient_num):
#     data_path = '/home/mainul1/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results_CORS/scratch6/data/'
#     Y = np.load(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(iter1)+'.npy')
#     Volume_axis.append(Y[:,2])
# for row in Volume_axis:
#     plt.plot(dose_axis, row, label = 'RECTUM')
# # for patientid in Volume_axis[patientid]:
# #     plt.axhline(y = patientid,ls = '--', lw = 0.4, color = 'purple')
# # plt.xticks(dose_axis)
# plt.legend(fontsize='5')
# plt.savefig(data_path_result_2  + 'DVH'+'rec' + str(epoch)+ 'iter' + str(iter1) +'.png',dpi=1200)
# plt.close()

# # for final DVH
# epoch = 190001
# pdose = 1
# Volume_axis = [] 
# dose_axis = np.linspace(0, pdose * 1.15, 100)
# for patientid in range(patient_num):
#     file_names = os.listdir(data_result_path)
#     Y = np.load(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(maximum_step(patientid))+'.npy')
#     Volume_axis.append(Y[:,0])
# for row in Volume_axis:
#     plt.plot(dose_axis, row, label = 'PTV')
# # for patientid in Volume_axis[patientid]:
# #     plt.axhline(y = patientid,ls = '--', lw = 0.4, color = 'purple')
# plt.xlabel('dose Axis')
# plt.ylabel('Volume Axis')
# plt.legend(fontsize='5')
# plt.savefig(data_path_result_2  + 'DVH'+'PTV' + str(epoch)+ 'iter' + 'max' +'.png',dpi=1200)
# plt.close()

# # for all DVH
# epoch = 190001
# pdose = 1
# Volume_axis = [] 
# dose_axis = np.linspace(0, pdose * 1.15, 100)
# colors = plt.cm.rainbow(np.linspace(0, 1, patient_num))
# # cmap = plt.get_cmap('gnuplot')
# # colors = [cmap(i) for i in np.linspace(0, 1, patient_num)]
# for step in range(22):
#     for patientid in range(patient_num):
#         try:
#             Y = np.load(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(step)+'.npy')
#             Volume_axis.append(Y[:,0])
#         except FileNotFoundError:
#             print(f'file not found for step {step}, skipping')
#             continue
#     # for patientid in Volume_axis[patientid]:
#     #     plt.axhline(y = patientid,ls = '--', lw = 0.4, color = 'purple')
#     for i, row in enumerate(Volume_axis):
#         plt.plot(dose_axis, row, label = 'PTV'+ str(step), color = colors[i])
#     plt.xlabel('dose Axis')
#     plt.ylabel('Volume Axis')
#     # plt.legend(fontsize='5')
#     plt.savefig(data_path_result_2  + 'DVH'+'PTV' + str(epoch)+ 'iter' + str(step) + '.png',dpi=1200)
#     plt.close()
#     Volume_axis = []

# # DVH for each patient each step
# epoch = 190001
# pdose = 1
# Volume_axis = [] 
# dose_axis = np.linspace(0, pdose * 1.15, 100)
# # cmap = plt.get_cmap('gnuplot')
# # colors = [cmap(i) for i in np.linspace(0, 1, patient_num)]
# for patientid in range(patient_num):
#     step_num = maximum_step(patientid)
#     colors = plt.cm.rainbow(np.linspace(0, 1, step_num))
#     for step in range(step_num):
#         try:
#             Y = np.load(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(step)+'.npy')
#             Volume_axis.append(Y[:,0])
#         except FileNotFoundError:
#             print(f'file not found for step {step}, skipping')
#             continue
#     # for patientid in Volume_axis[patientid]:
#     #     plt.axhline(y = patientid,ls = '--', lw = 0.4, color = 'purple')
#     for i, row in enumerate(Volume_axis):
#         plt.plot(dose_axis, row, label = 'PTV'+ str(i), color = colors[i])
#     plt.xlabel('dose Axis')
#     plt.ylabel('Volume Axis')
#     plt.legend(fontsize='5')
#     plt.savefig(data_path_result_2  + 'DVH'+'PTV' + str(epoch)+ 'patientid' + str(patientid) + '.png',dpi=1200)
#     plt.close()
#     Volume_axis = []

handles = []
labels = []

# DVH for each patient only few steps and making it full range
epoch = 120499
pdose = 1
Volume_axis = []
B_Volume_axis = []
R_Volume_axis = []
dose_axis_PTV = []
dose_axis_BLA = []
dose_axis_REC = []
# dose_axis = np.linspace(0, pdose * 1.15, 100)
# cmap = plt.get_cmap('gnuplot')
# colors = [cmap(i) for i in np.linspace(0, 1, patient_num)]
for patientid in patient_num:
    print('patientid', patientid)
    Max_step = maximum_step(patientid)
    Max_step = 3
    # step_num = [0,10, Max_step]
    step_num = [0, Max_step]
    colors = plt.cm.rainbow(np.linspace(0, 1, 3))
    linestyle = [':', '--', '-']
    for step in step_num:
        print('step=',step)
        try:
            Y = np.load(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(step)+'.npy')
            # # plotting for fixed
            # Volume_axis.append(Y[:,0])
            # B_Volume_axis.append(Y[:,1])
            # R_Volume_axis.append(Y[:,2])
            # plotting for maximum range
            Volume_axis.append(Y[:,6])
            B_Volume_axis.append(Y[:,7])
            R_Volume_axis.append(Y[:,8])
            dose_axis_PTV.append(Y[:,9])
            dose_axis_BLA.append(Y[:,10])
            dose_axis_REC.append(Y[:,11])
        except FileNotFoundError:
            print(f'file not found for step {step}, skipping')
            continue
    # for patientid in Volume_axis[patientid]:
    #     plt.axhline(y = patientid,ls = '--', lw = 0.4, color = 'purple')
    Volume_axis = np.array(Volume_axis)*100
    B_Volume_axis =  np.array(B_Volume_axis)*100
    R_Volume_axis =  np.array(R_Volume_axis)*100
    dose_axis_PTV =  np.array(dose_axis_PTV)*100
    dose_axis_BLA =  np.array(dose_axis_BLA)*100
    dose_axis_REC =  np.array(dose_axis_REC)*100

    # plt.rcParams["lines.linewidth"] = 3.5
    # plt.rcParams["axes.linewidth"] = 2.5
    # plt.rcParams["xtick.major.width"] = 2.5  # Major x-tick thickness
    # plt.rcParams["ytick.major.width"] = 2.5

    plt.rcParams["lines.linewidth"] = 5.5
    plt.rcParams["axes.linewidth"] = 4.5
    plt.rcParams["xtick.major.width"] = 4.5  # Major x-tick thickness
    plt.rcParams["ytick.major.width"] = 4.5
    plt.rcParams["xtick.major.size"] = 10  # Length of x-axis major ticks
    plt.rcParams["ytick.major.size"] = 10

    # plt.figure(figsize=(18, 9))
    plt.figure(figsize=(12, 8))
    for i, (row1, row2, row3) in enumerate(zip(dose_axis_PTV, Volume_axis, step_num)):
        line, = plt.plot(row1, row2, linestyle = linestyle[i], label = 'PTV'+ str(row3), color = 'red')
        handles.append(line)  # Store the line handle
        labels.append('PTV'+ str(row3))

    for j, (row1, row2, row3) in enumerate(zip(dose_axis_BLA, B_Volume_axis, step_num)):
        line, = plt.plot(row1, row2, linestyle = linestyle[j], label = 'BLA'+ str(row3), color = 'green')
        handles.append(line)  # Store the line handle
        labels.append('BLA'+ str(row3))

    for k, (row1, row2, row3) in enumerate(zip(dose_axis_REC, R_Volume_axis, step_num)):
        line, = plt.plot(row1, row2, linestyle = linestyle[k], label = 'REC'+ str(row3), color = 'blue')
        handles.append(line)  # Store the line handle
        labels.append('REC'+ str(row3))


    order = [0,3,6,1,4,7,2,5,8]
    ordered_handles = [handles[i] for i in order]
    ordered_labels = [labels[i] for i in order]
    # plt.xlim(0,120)
    # plt.figure(figsize=(10, 9))

    # plt.xticks(fontsize =40)
    # plt.yticks(fontsize = 40)

    plt.xticks(fontsize =45)
    plt.yticks(fontsize = 45)

    # Add titles and labels
    # plt.title('Adversarial Perturbation', fontsize = 14)
    # plt.xlabel('Relative Volume (%)', fontsize = 40)
    # plt.ylabel('Relative Volume (%)', fontsize = 40)

    plt.xlabel('Relative Dose (%)', fontsize = 45)
    plt.ylabel('Relative Volume (%)', fontsize = 45)
    # plt.legend(ordered_handles, ordered_labels, loc='lower left', fontsize = 31)
    plt.tight_layout()
    plt.xlim(0, 145)
    plt.ylim(0, 105)
    # plt.xlabel('Dose Axis')
    # plt.ylabel('Volume Axis')
    # plt.title('DVH for Patient ID '+str(patientid))
    # plt.legend(fontsize='5')
    # plt.savefig(graphSavePath  + 'DVH'+'PTV&OAR' + str(epoch)+ 'patientid' + str(patientid) + '.png',dpi=600)
    plt.savefig(graphSavePath  + 'DVH'+'PTV&OARwOhelp' + str(epoch)+ 'patientid' + str(patientid) + '.png',dpi=600)
    plt.show()
    # plt.show()
    plt.close()

    # trying to see which criteria are passed and which are not passed=======================================
    # prescription_dose = 79.5
    # dose_values_ptv = [87.12/prescription_dose, 1 ]
    # dose_values_ptv_original = [87.12 , prescription_dose]
    # dose_values_bla_exact = [80/prescription_dose, 75/prescription_dose, 70/prescription_dose, 65/prescription_dose]
    # dose_values_bla = [1.01, 0.947, 0.8838, 0.8207]
    # dose_values_bla_original = [80, 75, 70, 65]
    # dose_values_rec_exact = [75/prescription_dose, 70/prescription_dose, 65/prescription_dose, 60/prescription_dose]
    # dose_values_rec = [0.947, 0.8838, 0.8207, 0.7576]
    # dose_values_rec_original = [75, 70, 65, 60]

    # line_style_interest = '-'
    # line_label_interest_ptv = 'PTV'+str(Max_step)
    # line_label_interest_bla = 'BLA'+str(Max_step)
    # line_label_interest_rec = 'REC'+str(Max_step)

    # ax = plt.gca()
    # for l in ax.get_lines():
    #     if l.get_label() == line_label_interest_ptv and l.get_linestyle() == line_style_interest:
    #         Volume_percentage_ptv = np.interp(dose_values_ptv, l.get_xdata(), l.get_ydata())
    #         print(f"{Volume_percentage_ptv*100} percent of PTV volume is getting {dose_values_ptv_original} Gy dose") 
        
    #     elif l.get_label() == line_label_interest_bla and l.get_linestyle() == line_style_interest:
    #         Volume_percentage_bla = np.interp(dose_values_bla, l.get_xdata(), l.get_ydata())
    #         print(f"{Volume_percentage_bla*100} percent of BLA volume is getting {dose_values_bla_original} Gy dose")

    #     elif l.get_label() == line_label_interest_rec and l.get_linestyle() == line_style_interest:
    #         Volume_percentage_rec = np.interp(dose_values_rec, l.get_xdata(), l.get_ydata())
    #         print(f"{Volume_percentage_rec*100} percent of REC volume is getting {dose_values_rec_original} Gy dose")

    # plt.close()

    # Volume_axis = []
    # B_Volume_axis = []
    # R_Volume_axis = []
    # dose_axis_BLA = []
    # dose_axis_PTV = []
    # dose_axis_REC = []