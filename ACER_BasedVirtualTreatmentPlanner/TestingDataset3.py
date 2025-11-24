# -*- coding: utf-8 -*-
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
from contextlib import redirect_stdout
import asyncio
from telegram import Bot


from math import sqrt
from torch.distributions import Categorical
import numpy as np
from gym import spaces
import random

##################################################
from collections import deque
import h5sparse
import h5py
from scipy.sparse import vstack
from typing import List
from scipy.sparse import csr_matrix
import numpy.linalg as LA
import time
#################################################


from Prostate_TORTS.data_prep_TORTS_to_call import loadDoseMatrix,loadMask,ProcessDmat
pdose = 1 # target dose for PTV
maxiter = 40 # maximum iteration number for treatment planing optimization
##################################### ActorCritic Network ###############################

from gym import Env
from gym.spaces import Discrete
import matplotlib.pyplot as plt

INPUT_SIZE = 100  # DVH interval number
patient_list = ['01']

#######################################

class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    # self.state_size = observation_space.shape[0]
    self.state_size = 300
    self.action_size = action_space.n


    self.fc1 = nn.Linear(self.state_size, hidden_size) # (2,32)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size) # (32,32)
    self.fc_actor = nn.Linear(hidden_size, self.action_size) # (32, 4)
    self.fc_critic = nn.Linear(hidden_size, self.action_size) # (32,4)

  def forward(self, x, h):

    # Here, x = state
    x = F.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h

from math import pi
def planIQ_train(MPTV, MBLA, MREC, xVec,pdose,check):
    # score of treatment plan, two kinds of scores:
    # 1: score from standard criterion, 2: score_fined for self-defined in order to emphasize ptv
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)

    scoreall = np.zeros((11,))
    max_limit = 1

    avg_DPTV = DPTV[0]
    # line 2
    score2 = avg_DPTV - 1.1
    if score2 >= 0:
        score2 = 0
    if score2 < 0:
        score2 = 1 
    delta2 = 0.08
    if (avg_DPTV > 1.05):
        score2_fine = (1 / pi * np.arctan(-(avg_DPTV - 1.075) / delta2) + 0.5) * 8
    else:
        score2_fine = 6########################################
    # score2_fine = score2
    scoreall[0] = avg_DPTV


    DBLA1 = DBLA[DBLA >= 1.01]
    avg_DBLA1 = DBLA1.shape[0] / DBLA.shape[0]
    score5 = (avg_DBLA1 - 0.2 )/(-0.05)
    if score5 > max_limit:
        score5 = 1
    if score5 < 0:
        score5 = 0
    delta3 = 0.05
    if avg_DBLA1 < 0.2:
        score3_fine = 1 / pi * np.arctan(-(avg_DBLA1 - 0.175) / delta3) + 0.5
    else:
        score3_fine = 0
    # scoreall[3] = score5
    scoreall[3] = avg_DBLA1

    DBLA2 = DBLA[DBLA >= pdose * 0.947]
    avg_DBLA2 = DBLA2.shape[0] / DBLA.shape[0]
    score6 = (avg_DBLA2 - 0.3 )/(-0.05)
    if score6 > max_limit:
        score6 = 1
    if score6 < 0:
        score6 = 0
    delta4 = 0.05
    if avg_DBLA2 < 0.3:
        score4_fine = 1 / pi * np.arctan(-(avg_DBLA2 - 0.55 / 2) / delta4) + 0.5
    else:
        score4_fine = 0
    # scoreall[4] = avg
    scoreall[4] = avg_DBLA2

    DBLA3 = DBLA[DBLA >= 0.8838]
    avg_DBLA3 = DBLA3.shape[0] / DBLA.shape[0]
    score7 = (avg_DBLA3 - 0.4 )/(-0.05)
    if score7 > max_limit:
        score7 = 1
    if score7 < 0:
        score7 = 0
    delta5 = 0.05
    if avg_DBLA3 < 0.4:
        score5_fine = 1 / pi * np.arctan(-(avg_DBLA3 - 0.75 / 2) / delta5) + 0.5
    else:
        score5_fine = 0
    # scoreall[5] = score7
    scoreall[5] = avg_DBLA3

    DBLA4 = DBLA[DBLA >= 0.8207]
    avg_DBLA4 = DBLA4.shape[0] / DBLA.shape[0]
    score8 = (avg_DBLA4 - 0.55)/(-0.05)
    if score8 > max_limit:
        score8 = 1
    if score8 < 0:
        score8 = 0
    delta6 = 0.05
    if avg_DBLA4 < 0.55:
        score6_fine = 1 / pi * np.arctan(-(avg_DBLA4 - 1.05 / 2) / delta6) + 0.5
    else:
        score6_fine = 0
    # scoreall[6] = score8
    scoreall[6] = avg_DBLA4

    DREC1 = DREC[DREC >= 0.947]
    avg_DREC1 = DREC1.shape[0] / DREC.shape[0]
    score9 = (avg_DREC1 - 0.2)/(-0.05)
    if score9 > max_limit:
        score9 = 1
    if score9 < 0:
        score9 = 0
    delta7 = 0.05
    if avg_DREC1 < 0.2:
        score7_fine = 1 / pi * np.arctan(-(avg_DREC1 - 0.35 / 2) / delta7) + 0.5
    else:
        score7_fine = 0
    # scoreall[7] = score9
    scoreall[7] = avg_DREC1

    DREC2 = DREC[DREC >= 0.8838]
    avg_DREC2 = DREC2.shape[0] / DREC.shape[0]
    score10 = (avg_DREC2 - 0.3)/(-0.05)
    if score10 > max_limit:
        score10 = 1
    if score10 < 0:
        score10 = 0
    delta8 = 0.05
    if avg_DREC2 < 0.3:
        score8_fine = 1 / pi * np.arctan(-(avg_DREC2 - 0.55 / 2) / delta8) + 0.5
    else:
        score8_fine = 0

    # scoreall[8] = score10
    scoreall[8] = avg_DREC2

    DREC3 = DREC[DREC >= 0.8207]
    avg_DREC3 = DREC3.shape[0] / DREC.shape[0]
    score11 = (avg_DREC3 - 0.4)/(-0.05)
    if score11 > max_limit:
        score11 = 1
    if score11 < 0:
        score11 = 0
    delta9 = 0.05
    if avg_DREC3 < 0.4:
        score9_fine = 1 / pi * np.arctan(-(avg_DREC3 - 0.75 / 2) / delta9) + 0.5
    else:
        score9_fine = 0

    # scoreall[9] = score11
    scoreall[9] = avg_DREC3

    DREC4 = DREC[DREC >= 0.7576]
    avg_DREC4 = DREC4.shape[0] / DREC.shape[0]
    score12 = (avg_DREC4 - 0.55)/(-0.05)
    if score12 > max_limit:
        score12 = 1
    if score12 < 0:
        score12 = 0
    delta10 = 0.05
    if avg_DREC4 < 0.55:
        score10_fine = 1 / pi * np.arctan(-(avg_DREC4 - 1.05 / 2) / delta10) + 0.5
    else:
        score10_fine = 0

    # scoreall[10] = score12
    scoreall[10] = avg_DREC4

    score = score2 + score5 + score6 + score7 + score8 + score9 + score10 + score11 + score12
    if check == True:
        print(score2, score5, score6, score7, score8, score9, score10, score11, score12)
    if score2_fine > 0.5:
        score_fine = score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine
    else:
        score_fine = (
                    score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine)

    return score_fine, score, scoreall

def MinimizeDoseOAR_dvh(MPTV, MBLA, MREC,tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec, gamma,pdose,maxiter):
    # treatment planning optimization in DVH-based scheme
    beta=2
    lambdaBLA = lambdaBLA/lambdaPTV
    lambdaREC = lambdaREC/lambdaPTV
    # xVec = np.ones((MPTV.shape[1],))
    DPTV = MPTV.dot(xVec)
    DPTV1 = np.sort(DPTV)
    posi = int(round(0.05 * DPTV1.shape[0]))
    D95 = DPTV1[posi]
#    MPTV95 = MPTV[DPTV>=D95,:]
    factor = pdose / D95
    xVec = xVec * factor
    y = MPTV.dot(xVec)
    MPTVT = MPTV.transpose()
    DPTV = MPTV.dot(xVec)
    for iter in range(maxiter):
        xVec_old = xVec
        DPTV1 = np.sort(DPTV)
        posi = int(round((1 - VPTV) * DPTV1.shape[0]))-1
        if posi < 0:###### what was missing
            posi = 0###### what was missing
        DPTVV = DPTV1[posi]

        DBLA = MBLA.dot(xVec)
        DBLA1 = np.sort(DBLA)
        posi = int(round((1 - VBLA) * DBLA1.shape[0]))-1
        if posi < 0:
            posi = 0
        DBLAV = DBLA1[posi]
        DREC = MREC.dot(xVec)
        DREC1 = np.sort(DREC)
        posi = int(round((1 - VREC) * DREC1.shape[0]))-1
        if posi < 0:
            posi = 0
        DRECV = DREC1[posi]

        MPTVV =  MPTV[DPTV>=DPTVV,:]
        temp= DPTV[DPTV>=DPTVV]
        if np.max(temp) > pdose* tPTV:
            MPTV1 = MPTVV[temp > pdose*tPTV, :]
            targetPTV1 = pdose*tPTV*np.ones((MPTV1.shape[0],))
            MPTV1T = MPTV1.transpose()
            temp1 = MPTV1.dot(xVec)
            temp1 = MPTV1T.dot(temp1)
            temp1 = temp1 * 1/MPTV1.shape[0]
            b1 = MPTV1T.dot(targetPTV1) / MPTV1.shape[0]
        else:
            temp1 = np.zeros((xVec.shape))
            b1 = np.zeros((xVec.shape))
            tempp1 = np.zeros((xVec.shape))
        tempptv = temp

        temp2 = MPTV.dot(xVec)
        temp2 = beta*MPTVT.dot(temp2)/y.shape[0]
        b2 =  beta*MPTVT.dot(y)/y.shape[0]


        MBLAV = MBLA[DBLA>=DBLAV,:]
        temp = DBLA[DBLA>=DBLAV]
        if np.max(temp) > pdose * tBLA:
            MBLA1 = MBLAV[temp > pdose * tBLA, :]
            targetBLA1 = pdose*tBLA*np.ones((MBLA1.shape[0],))
            MBLA1T = MBLA1.transpose()
            temp3 = MBLA1.dot(xVec)
            temp3 = MBLA1T.dot(temp3)
            temp3 = temp3 * lambdaBLA/MBLA1.shape[0]
            b3 = lambdaBLA * MBLA1T.dot(targetBLA1) / max(MBLA1.shape[0], 1)
        else:
            temp3 = np.zeros((xVec.shape))
            b3 = np.zeros((xVec.shape))
            tempp3 = np.zeros((xVec.shape))
        tempbla = temp

        MRECV = MREC[DREC >= DRECV, :]
        temp = DREC[DREC >= DRECV]
        if np.max(temp) > pdose * tREC:
            MREC1 = MRECV[temp > pdose * tREC, :]
            targetREC1 = pdose*tREC*np.ones((MREC1.shape[0],))
            MREC1T = MREC1.transpose()
            temp4 = MREC1.dot(xVec)
            temp4 = MREC1T.dot(temp4)
            temp4 = temp4 * lambdaREC/MREC1.shape[0]
            b4 = lambdaREC * MREC1T.dot(targetREC1) / MREC1.shape[0]
        else:
            temp4 = np.zeros((xVec.shape))
            b4 = np.zeros((xVec.shape))
            tempp4 = np.zeros((xVec.shape))
        temprec = temp

        templhs = temp1+temp2+temp3+temp4
        b = b1+b2+b3+b4-MPTVT.dot(gamma)
        r = b - templhs
        p = r
        rsold = np.inner(r,r)

        if rsold>1e-10:
            for i in range(3):
                if np.max(tempptv) > pdose*tPTV :
                    tempp1 = MPTV1.dot(p)
                    tempp1 = MPTV1T.dot(tempp1)
                    tempp1 = tempp1 * 1 / MPTV1.shape[0]


                tempp2 = MPTV.dot(p)
                tempp2 = beta * MPTVT.dot(tempp2)/y.shape[0]

                if np.max(tempbla) > pdose * tBLA:
                    tempp3 = MBLA1.dot(p)
                    tempp3 = MBLA1T.dot(tempp3)
                    tempp3 = tempp3 * lambdaBLA / MBLA1.shape[0]

                if np.max(temprec) > pdose * tREC:
                    tempp4 = MREC1.dot(p)
                    tempp4 = MREC1T.dot(tempp4)
                    tempp4 = tempp4 * lambdaREC / MREC1.shape[0]


                Ap = tempp1 + tempp2 + tempp3 + tempp4
                pAp = np.inner(p, Ap)
                alpha = rsold / pAp
                xVec = xVec + alpha * p
                xVec[xVec<0]=0
                r = r - alpha * Ap
                rsnew = np.inner(r, r)
                if np.sqrt(rsnew) < 1e-5:
                    break
                p = r + (rsnew / rsold) * p
                rsold = rsnew
        DPTV = MPTV.dot(xVec)
        y = (DPTV * beta/y.shape[0] + gamma) / (beta/y.shape[0])
        Dy = np.sort(y)
        posi = int(round(0.05 * Dy.shape[0]))
        D95 = Dy[posi]
        temp = np.zeros(y.shape)
        temp[y>=D95] = y[y>=D95]
        temp[temp<pdose] = pdose
        y[y>=D95] = temp[y>=D95]
        gamma = gamma + beta * (MPTV.dot(xVec)-y)/y.shape[0]

        if LA.norm(xVec - xVec_old, 2) / LA.norm(xVec_old, 2) < 5e-3:
            break
    DPTV = MPTV.dot(xVec)
    DPTV1 = np.sort(DPTV)
    posi = int(round(0.05 * DPTV1.shape[0]))
    D95 = DPTV1[posi]
    factor = pdose / D95 # thresholidng
    xVec = xVec * factor
    converge = 1
    if iter == maxiter - 1:
        converge = 0

    return xVec, iter

def runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter):
    # run optimization and generate DVH curves
    xVec, iter = MinimizeDoseOAR_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC,lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
#    j = 0
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)

    ## Plot DVH curve for optimized plan
    edge_ptv = np.zeros((INPUT_SIZE + 1,))
    edge_ptv[1:INPUT_SIZE + 1] = np.linspace(pdose, pdose * 1.15, INPUT_SIZE)
    (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
    y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

    edge_bladder = np.zeros((INPUT_SIZE + 1,))
    edge_bladder[1:INPUT_SIZE + 1] = np.linspace(0.6 * pdose, 1.1 * pdose, INPUT_SIZE)
    (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
    y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

    edge_rectum = np.zeros((INPUT_SIZE + 1,))
    edge_rectum[1:INPUT_SIZE + 1] = np.linspace(0.6 * pdose, 1.1 * pdose, INPUT_SIZE)
    (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
    y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

    Y = np.zeros((INPUT_SIZE, 3))
    Y[:, 0] = y_ptv
    Y[:, 1] = y_bladder
    Y[:, 2] = y_rectum

    Y = np.reshape(Y, (100 * 3,), order='F')

    return Y, iter, xVec

import math as m
class TreatmentEnv(Env):
    """A Treatment planning environment for OpenAI gym"""

    def __init__(self):
        self.action_space = Discrete(18)  # Box(low=np.array([0,0.5]), high=np.array([26,1.5]), dtype=np.float32)#Discrete (26)
        self.observation_space = np.zeros([300])

        self.time_limit = 30

    def step(self, action, t, Score_fine, Score, MPTV, MBLA, MREC, MBLA1, MREC1, tPTV, tBLA, tREC,
             lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, pdose, maxiter):

        self.action = action

        paraMax = 100000  # change in validation as well
        paraMin = 0
        paraMax_tPTV = 1.2
        paraMin_tPTV = 1
        paraMax_tOAR = 1
        paraMax_VOAR = 1
        paraMax_VPTV = 0.3


        if action == 0:
            tPTV = min(tPTV * 1.01, paraMax_tPTV)
        elif action == 1:
            tPTV = max(tPTV * 0.91, paraMin_tPTV)
        elif action == 2:
            tBLA = min(tBLA * 1.25, paraMax_tOAR)
        elif action == 3:
            tBLA = tBLA * 0.6
        elif action == 4:
            tREC = min(tREC * 1.25, paraMax_tOAR)
        elif action == 5:
            tREC = tREC * 0.6
        elif action == 6:
            lambdaPTV = lambdaPTV * 1.65
        elif action == 7:
            lambdaPTV = lambdaPTV * 0.6
        elif action == 8:
            lambdaBLA = lambdaBLA * 1.65
        elif action == 9:
            lambdaBLA = lambdaBLA * 0.6
        elif action == 10:
            lambdaREC = lambdaREC * 1.65
        elif action == 11:
            lambdaREC = lambdaREC * 0.6
        elif action == 12:
            VPTV = min(VPTV * 1.25, paraMax_VPTV)
        elif action == 13:
            VPTV = VPTV * 0.8
        elif action == 14:
            VBLA = min(VBLA * 1.25, paraMax_VOAR)
        elif action == 15:
            VBLA = VBLA * 0.8
        elif action == 16:
            VREC = min(VREC * 1.25, paraMax_VOAR)
        elif action == 17:
            VREC = VREC * 0.8

        xVec = np.ones((MPTV.shape[1],))
        gamma = np.zeros((MPTV.shape[0],))
        n_state, _, xVec = \
            runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,
                       gamma, pdose, maxiter)
        Score_fine1, Score1, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)

        extra = 0 if Score1 != 9 else 2

        # Uncomment this part original scoring system
        reward = (Score_fine1 - Score_fine) + (Score1 - Score) * 4
        Done = False
        if Score1 == 9:
            Done = True
        return n_state, reward, Score_fine1, Score1, Done, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec

    def reset(self):
        self.state = self.observation_space
        self.time_limit = 30
        return self.state

    def close(self):
        pass



def test(rank, args, T):
    print("testing starts")

    cuda = False

    save_dir = os.path.join('results_test', args.name)

    can_test = True  # Test flag
    t_start = 1  # Test step counter to check against global counter
    rewards, steps = [], []  # Rewards and steps for plotting
    l = str(len(str(args.T_max)))  # Max num. of digits for logging steps

    test_set = patient_list

    i = 0

    env = TreatmentEnv()
    model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)


    done = True  # Start new episode

    # stores step, reward, avg_steps and time
    results_dict = {'t': [], 'reward': [], 'avg_steps': [], 'time': []}

    # Loading of all MPTV and other files
    pid = patient_list

    data_path = '/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/'
    data_path2 = '/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/'
    data_path_TORTS = '/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/Prostate_TORTS/'

    pid = ['01', '02','03','04','05','06','07','08','09','10', '11', '12','13','14','15','16','17','18','19', '20', '21', '22','23','24','25','26','27','28','29', '30']

    for i in range(len(pid)):
        print("len(pid)", len(pid))
        print("loading patient:",pid[i])
        globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path_TORTS+ 'test_TORTS' + str(pid[i])+ '.hdf5')
        # print("doseMatrix loaded")
        globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()[
            'rectumLabel' + str(i)], \
        globals()['PTVLabel' + str(i)] = loadMask(data_path_TORTS + "test_dose_mask_TORTS" + str(pid[i]) + ".h5", data_path_TORTS + "test_structure_mask_TORTS" + str(pid[i])+".h5",)
        print("PTVLabel loaded")


    while T.value() <= args.T_max:
        if can_test:
            t_start = T.value()  # Reset counter

            epoch = 120499



            model.load_state_dict(torch.load('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/results/results/episode120499.pth'))
            model.eval()

            # ------------- range of parmaeter -----------------
            paraMax = 100000  # change in validation as well
            paraMin = 0.1
            paraMax_tPTV = 1.2
            paraMin_tPTV = 1
            paraMax_tOAR = 1
            paraMax_VOAR = 1
            paraMax_VPTV = 0.3
            # ---------------------------------------------------

            avg_rewards, avg_episode_lengths = [], []

            for patientid in range(len(pid)):
                id1 = patientid
                print("========================================Patient",id1)
                print("done", done)

                pid = ['01', '02','03','04','05','06','07','08','09','10', '11', '12','13','14','15','16','17','18','19', '20', '21', '22','23','24','25','26','27','28','29', '30']
                sampleid = patientid
                doseMatrix = globals()['doseMatrix_' + str(sampleid)]
                targetLabels = globals()['targetLabels_' + str(sampleid)]
                bladderLabel = globals()['bladderLabel' + str(sampleid)]
                rectumLabel = globals()['rectumLabel' + str(sampleid)]
                PTVLabel = globals()['PTVLabel' + str(sampleid)]

                MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, bladderLabel,
                                                             rectumLabel)
                time_per_episode = 0

                while True:

                    if done:

                        hx = torch.zeros(1, args.hidden_size)
                        cx = torch.zeros(1, args.hidden_size)

                        done, episode_length = False, 0
                        reward_sum = 0

                        env = TreatmentEnv()

                        planScore = 8.5
                        # ------------------------ initial paramaters & input --------------------
                        while (planScore >= 8.1):
                            # this is the general block
                            epsilon = 1e-10
                            tPTV = random.uniform(1 + epsilon, 1.2 - epsilon)
                            tBLA = random.uniform(0.3 + epsilon, 1 - epsilon)
                            tREC = random.uniform(0.3 + epsilon, 1 - epsilon)
                            lambdaPTV = random.uniform(0.3 + epsilon, 1 - epsilon)
                            lambdaBLA = random.uniform(0.3 + epsilon, 1 - epsilon)
                            lambdaREC = random.uniform(0.3 + epsilon, 1 - epsilon)
                            VPTV = random.uniform(0.1 + epsilon, 0.3 - epsilon)
                            VBLA = random.uniform(0.3 + epsilon, 1 - epsilon)
                            VREC = random.uniform(0.3 + epsilon, 1 - epsilon)
                            xVec = np.ones((MPTV.shape[1],))
                            gamma = np.zeros((MPTV.shape[0],))
                            
                            print("tPTV:{}  tBLA:{}  tREC:{}  lambdaPTV:{} lambdaBLA:{}   lambdaREC:{}  VPTV:{}  VBLA:{}  VREC:{} ",
                                tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC)
                            # --------------------- solve treatment planning optmization -----------------------------
                            state_test0, iter, xVec = \
                                runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC,
                                           lambdaPTV, lambdaBLA, lambdaREC,
                                           VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter)
                            # --------------------- generate input for NN -----------------------------
                            Dose = doseMatrix.dot(xVec)
                            DPTV = MPTV.dot(xVec)
                            DBLA = MBLA.dot(xVec)
                            DREC = MREC.dot(xVec)
                            DPTV = np.sort(DPTV)
                            DPTV = np.flipud(DPTV)
                            DBLA = np.sort(DBLA)
                            DBLA = np.flipud(DBLA)
                            DREC = np.sort(DREC)
                            DREC = np.flipud(DREC)

                            # For plotting against changing and fixed both edge_ptv
                            edge_ptv = np.zeros((100 + 1,))
                            edge_ptv_max = np.zeros((100 + 1,))                        
                            edge_ptv[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                            edge_ptv_max[1:100 + 1] = np.linspace(0, max(DPTV), 100)
                            x_ptv = np.linspace(0.5 * max(DPTV) / 100, max(DPTV), 100)
                            (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                            (n_ptv_max, b_max) = np.histogram(DPTV, bins=edge_ptv_max)
                            y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)
                            y_ptv_max = 1 - np.cumsum(n_ptv_max / len(DPTV), axis=0)

                            edge_bladder = np.zeros((100 + 1,))
                            edge_bladder_max = np.zeros((100 + 1,))
                            edge_bladder[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                            edge_bladder_max[1:100 + 1] = np.linspace(0, max(DBLA), 100)                            
                            x_bladder = np.linspace(0.5 * max(DBLA) / 100, max(DBLA), 100)
                            (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                            (n_bladder_max, b_max) = np.histogram(DBLA, bins = edge_bladder_max)
                            y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)
                            y_bladder_max = 1 - np.cumsum(n_bladder_max/len(DBLA), axis = 0)

                            edge_rectum = np.zeros((100 + 1,))
                            edge_rectum_max = np.zeros((100 + 1,))
                            edge_rectum[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                            edge_rectum_max[1:100 + 1] = np.linspace(0, max(DREC), 100)                            
                            x_rectum = np.linspace(0.5 * max(DREC) / 100, max(DREC), 100)
                            (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                            (n_rectum_max, b_max) = np.histogram(DREC , bins = edge_rectum_max)
                            y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)
                            y_rectum_max = 1 - np.cumsum(n_rectum_max / len(DREC), axis = 0)

                            Y = np.zeros((100, 12))
                            Y[:, 0] = y_ptv
                            Y[:, 1] = y_bladder
                            Y[:, 2] = y_rectum

                            # X = np.zeros((1000, 3))
                            Y[:, 3] = x_ptv
                            Y[:, 4] = x_bladder
                            Y[:, 5] = x_rectum

                            # storing max range histograms
                            Y[:, 6] = y_ptv_max
                            Y[:, 7] = y_bladder_max
                            Y[:, 8] = y_rectum_max

                            Y[:, 9] = edge_ptv_max[1:100+1]
                            Y[:, 10] = edge_bladder_max[1:100+1]
                            Y[:, 11] = edge_rectum_max[1:100+1]


                            # This is the new save path
                            planscoresSavePath = '/data2/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/planscores3rd/'
                            os.makedirs(planscoresSavePath, exist_ok = True)
                            data_result_path='/data2/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/dataWithPlanscoreRun3rd/'
                            os.makedirs(data_result_path, exist_ok = True)
                            actionSavePath = '/data2/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/actions3rd/'
                            os.makedirs(actionSavePath, exist_ok = True)
                            np.save(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(episode_length),
                                Y)
                            np.save(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(episode_length),
                                Y)
                            # np.save(data_result_path+id+'xDVHYInitial',
                            #         Y)
                            # np.save(data_result_path+id+'xDVHXInitial',
                            # np.save(data_result_path+id+'xDVHXInitial',
                            #         X)
                            # np.save(data_result_path+id+'xVecInitial', xVec)
                            # np.save(data_result_path+id+'DoseInitial', Dose)
                            np.save(data_result_path+str(patientid)+'Dose' + str(epoch)  + 'step' + str(episode_length), Dose)
                            np.save(data_result_path+str(patientid)+'doseMatrix' + str(epoch)  + 'step' + str(episode_length), doseMatrix)
                            np.save(data_result_path+str(patientid)+'bladderLabel' + str(epoch)  + 'step' + str(episode_length), bladderLabel)
                            np.save(data_result_path+str(patientid)+'rectumLabel' + str(epoch)  + 'step' + str(episode_length), rectumLabel)
                            np.save(data_result_path+str(patientid)+'PTVLabel' + str(epoch)  + 'step' + str(episode_length), PTVLabel)

                            MAX_STEP = args.max_episode_length + 1
                            tPTV_all = np.zeros((MAX_STEP + 1))
                            tBLA_all = np.zeros((MAX_STEP + 1))
                            tREC_all = np.zeros((MAX_STEP + 1))
                            lambdaPTV_all = np.zeros((MAX_STEP + 1))
                            lambdaBLA_all = np.zeros((MAX_STEP + 1))
                            lambdaREC_all = np.zeros((MAX_STEP + 1))
                            VPTV_all = np.zeros((MAX_STEP + 1))
                            VBLA_all = np.zeros((MAX_STEP + 1))
                            VREC_all = np.zeros((MAX_STEP + 1))
                            planScore_all = np.zeros((MAX_STEP + 1))
                            planScore_fine_all = np.zeros((MAX_STEP + 1))
                            # print("Testing score")
                            planScore_fine, planScore, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)
                            # logging.info('---------------------- initialization ------------------------------')

                            tPTV_all[0] = tPTV
                            tBLA_all[0] = tBLA
                            tREC_all[0] = tREC
                            lambdaPTV_all[0] = lambdaPTV
                            lambdaBLA_all[0] = lambdaBLA
                            lambdaREC_all[0] = lambdaREC
                            VPTV_all[0] = VPTV
                            VBLA_all[0] = VBLA
                            VREC_all[0] = VREC
                            planScore_all[0] = planScore
                            planScore_fine_all[0] = planScore_fine
                            print(planScore)
                            np.save(planscoresSavePath+str(patientid)+ 'planscoreInsideIf',planScore)
                            # ----------------two ways of NN schemes (with/without rules) --------------------------
                            state_test = state_test0
                            if cuda:
                                state_test = torch.from_numpy(state_test).float()
                                state_test = state_test.to(device)

                            state_test = state_to_tensor(state_test)
                            start_time = time.time()
                        # ------------------------ initial paramaters & input --------------------
                        
                    start_time_old = start_time
                    start_time = time.time()
                    Total_code_time = start_time - start_time_old
                    print('Total Code Time', Total_code_time)
                    # Calculate policy
                    with torch.no_grad():
                        policy, _, _, (hx, cx) = model(state_test, (hx, cx))


                    action = torch.multinomial(policy, 1)[0, 0]

                    print('action', action)
                    end_time = time.time()
                    delta_time = end_time - start_time
                    print('Time for prediction', delta_time)
                    time_per_episode = time_per_episode+Total_code_time
                    print('Time Per Episode', time_per_episode )
                    
                    t = 1

                    if action == 0:
                        tPTV = min(tPTV * 1.01, paraMax_tPTV)
                    elif action == 1:
                        tPTV = max(tPTV * 0.91, paraMin_tPTV)
                    elif action == 2:
                        tBLA = min(tBLA * 1.25, paraMax_tOAR)
                    elif action == 3:
                        tBLA = tBLA * 0.6
                    elif action == 4:
                        tREC = min(tREC * 1.25, paraMax_tOAR)
                    elif action == 5:
                        tREC = tREC * 0.6
                    elif action == 6:
                        lambdaPTV = lambdaPTV * 1.65
                    elif action == 7:
                        lambdaPTV = lambdaPTV * 0.6
                    elif action == 8:
                        lambdaBLA = lambdaBLA * 1.65
                    elif action == 9:
                        lambdaBLA = lambdaBLA * 0.6
                    elif action == 10:
                        lambdaREC = lambdaREC * 1.65
                    elif action == 11:
                        lambdaREC = lambdaREC * 0.6
                    elif action == 12:
                        VPTV = min(VPTV * 1.25, paraMax_VPTV)
                    elif action == 13:
                        VPTV = VPTV * 0.8
                    elif action == 14:
                        VBLA = min(VBLA * 1.25, paraMax_VOAR)
                    elif action == 15:
                        VBLA = VBLA * 0.8
                    elif action == 16:
                        VREC = min(VREC * 1.25, paraMax_VOAR)
                    elif action == 17:
                        VREC = VREC * 0.8

                    print("tPTV:{}  tBLA:{}  tREC:{}  lambdaPTV:{} lambdaBLA:{}   lambdaREC:{}  VPTV:{}  VBLA:{}  VREC:{} ".format(
                            tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC))

                    # --------------------- solve treatment planning optmization -----------------------------
                    xVec = np.ones((MPTV.shape[1],))
                    gamma = np.zeros((MPTV.shape[0],))
                    n_state_test, iter, xVec = \
                        runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA,
                                   VREC, xVec,
                                   gamma, pdose, maxiter)
                    planScore_fine, planScore, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)
                    print("\n")
                    print("Test:{},{} ,{}, Individual_scores[{}]".format(

                                                                                                       epoch,
                                                                                                       planScore,
                                                                                                       planScore_fine,
                                                                                                       scoreall))
                    state_test = n_state_test
                    if cuda:
                        state_test = torch.from_numpy(state_test).float()
                        state_test = state_test.to(device)

                    state_test = state_to_tensor(state_test)

                    j = episode_length

                    tPTV_all[j + 1] = tPTV
                    tBLA_all[j + 1] = tBLA
                    tREC_all[j + 1] = tREC
                    lambdaPTV_all[j  + 1] = lambdaPTV
                    lambdaBLA_all[j + 1] = lambdaBLA
                    lambdaREC_all[j + 1] = lambdaREC
                    VPTV_all[j + 1] = VPTV
                    VBLA_all[j + 1] = VBLA
                    VREC_all[j + 1] = VREC
                    planScore_all[j + 1] = planScore
                    planScore_fine_all[j + 1] = planScore_fine
                    print(planScore)

                    Dose = doseMatrix.dot(xVec)
                    # np.save(data_result_path+id+'xVec' + str(episode)  + 'step' + str(i + 1), xVec)
                    # np.save(data_result_path+id+'xDose' + str(episode)  + 'step' + str(i + 1), Dose)
                    np.save(data_result_path+str(patientid)+'Dose' + str(epoch)  + 'step' + str(episode_length + 1), Dose)

                    DPTV = MPTV.dot(xVec)
                    DBLA = MBLA.dot(xVec)
                    DREC = MREC.dot(xVec)
                    DPTV = np.sort(DPTV)
                    DPTV = np.flipud(DPTV)
                    DBLA = np.sort(DBLA)
                    DBLA = np.flipud(DBLA)
                    DREC = np.sort(DREC)
                    DREC = np.flipud(DREC)

                    # For plotting against changing and fixed both edge_ptv
                    edge_ptv = np.zeros((100 + 1,))
                    edge_ptv_max = np.zeros((100 + 1,))                        
                    edge_ptv[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                    edge_ptv_max[1:100 + 1] = np.linspace(0, max(DPTV), 100)
                    x_ptv = np.linspace(0.5 * max(DPTV) / 100, max(DPTV), 100)
                    (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                    (n_ptv_max, b_max) = np.histogram(DPTV, bins=edge_ptv_max)
                    y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)
                    y_ptv_max = 1 - np.cumsum(n_ptv_max / len(DPTV), axis=0)

                    edge_bladder = np.zeros((100 + 1,))
                    edge_bladder_max = np.zeros((100 + 1,))
                    edge_bladder[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                    edge_bladder_max[1:100 + 1] = np.linspace(0, max(DBLA), 100)                            
                    x_bladder = np.linspace(0.5 * max(DBLA) / 100, max(DBLA), 100)
                    (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                    (n_bladder_max, b_max) = np.histogram(DBLA, bins = edge_bladder_max)
                    y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)
                    y_bladder_max = 1 - np.cumsum(n_bladder_max/len(DBLA), axis = 0)

                    edge_rectum = np.zeros((100 + 1,))
                    edge_rectum_max = np.zeros((100 + 1,))
                    edge_rectum[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
                    edge_rectum_max[1:100 + 1] = np.linspace(0, max(DREC), 100)                            
                    x_rectum = np.linspace(0.5 * max(DREC) / 100, max(DREC), 100)
                    (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                    (n_rectum_max, b_max) = np.histogram(DREC , bins = edge_rectum_max)
                    y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)
                    y_rectum_max = 1 - np.cumsum(n_rectum_max / len(DREC), axis = 0)

                    Y = np.zeros((100, 12))
                    Y[:, 0] = y_ptv
                    Y[:, 1] = y_bladder
                    Y[:, 2] = y_rectum

                    # X = np.zeros((1000, 3))
                    Y[:, 3] = x_ptv
                    Y[:, 4] = x_bladder
                    Y[:, 5] = x_rectum

                    # storing max range histograms                    
                    Y[:, 6] = y_ptv_max
                    Y[:, 7] = y_bladder_max
                    Y[:, 8] = y_rectum_max

                    Y[:, 9] = edge_ptv_max[1:100+1]
                    Y[:, 10] = edge_bladder_max[1:100+1]
                    Y[:, 11] = edge_rectum_max[1:100+1]

                    np.save(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(episode_length + 1),
                            Y)
 
                    check_model2 = model.state_dict()


                    done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length

                    episode_length += 1  # Increase episode counter

                    np.save(planscoresSavePath+str(patientid)+'planscoreBeforeWhileBreaking'+str(episode_length),planScore)
                    np.save(actionSavePath+str(patientid)+ 'actionBeforeWhileBreaking'+ str(episode_length), action)
                    if planScore == 9:
                        done = True
                        break

                    if done:
                        break


                tpp_parameters = np.zeros((MAX_STEP + 1,11))
                tpp_parameters[:, 0] = np.array(tPTV_all)
                tpp_parameters[:, 1] = np.array(tBLA_all)
                tpp_parameters[:, 2] = np.array(tREC_all)
                tpp_parameters[:, 3] = np.array(lambdaPTV_all)
                tpp_parameters[:, 4] = np.array(lambdaBLA_all)
                tpp_parameters[:, 5] = np.array(lambdaREC_all)
                tpp_parameters[:, 6] = np.array(VPTV_all)
                tpp_parameters[:, 7] = np.array(VBLA_all)
                tpp_parameters[:, 8] = np.array(VREC_all)
                tpp_parameters[:, 9] = np.array(planScore_all)
                tpp_parameters[:, 10] = np.array(planScore_fine_all)
                # np.save(data_result_path + id + 'tpptuning' + str(epoch),
                #         tpp_parameters)
                name1 = data_result_path + str(patientid) + 'tpptuning' + str(epoch)
                np.savez(name1+'.npz',l1 = tPTV_all, l2 =tBLA_all,  l3 = tREC_all, l4 =lambdaPTV_all, l5 =lambdaBLA_all, l6 =lambdaREC_all, l7 = VPTV_all, l8= VBLA_all, l9 = VREC_all, l10 = planScore_all, l11 = planScore_fine_all)
                np.save(planscoresSavePath+str(patientid)+'planscoreBeforeForBreaking',planScore)


            # Dumping the results in pickle format
            with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
                pickle.dump(results_dict, f)

            if args.evaluate:
                return

            steps.append(t_start)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))  # Save model params

            can_test = False  # Finish testing

        else:
            # print("t_start", t_start)
            # print("diff:", T.value() - t_start)
            if T.value() - t_start >= args.evaluation_interval:
                can_test = True

        time.sleep(0.001)  # Check if available to test every millisecond

    # Dumping the results in pickle format

    with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
        pickle.dump(results_dict, f)
    env.close()


# Original arguments to use
parser = argparse.ArgumentParser(description='ACER')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--num-processes', type=int, default=3, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=250000, metavar='STEPS', help='Number of training steps')
# parser.add_argument('--T-max', type=int, default=100, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
# parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
# parser.add_argument('--max-episode-length', type=int, default=20, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--max-episode-length', type=int, default=29, metavar='LENGTH', help='Maximum episode length')
# parser.add_argument('--max-episode-length', type=int, default=30, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--on-policy', action='store_true', help='Use pure on-policy training (A3C)')
parser.add_argument('--memory-capacity', type=int, default=100000, metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
parser.add_argument('--replay-start', type=int, default=20000, metavar='EPISODES', help='Number of transitions to save before starting off-policy training')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--trace-max', type=float, default=10, metavar='c', help='Importance weight truncation (max) value')
parser.add_argument('--trust-region', action='store_true', help='Use trust region')
parser.add_argument('--trust-region-decay', type=float, default=0.99, metavar='α', help='Average model weight decay rate')
parser.add_argument('--trust-region-threshold', type=float, default=1, metavar='δ', help='Trust region threshold value')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--lr', type=float, default=0.0007, metavar='η', help='Learning rate')
parser.add_argument('--lr-decay', action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--batch-size', type=int, default=16, metavar='SIZE', help='Off-policy batch size')
parser.add_argument('--entropy-weight', type=float, default=0.0001, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE', help='Gradient L2 normalisation')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=500, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
# parser.add_argument('--evaluation-episodes', type=int, default=30, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--name', type=str, default='results', help='Save folder')
parser.add_argument('--env', type=str, default='CartPole-v1',help='environment name')


# The following block is for running in the background
TELEGRAM_BOT_TOKEN = "7949104193:AAFdqdcHtvsIVyqrw42pOLqdHIn_vv3fkaw"  # Your Telegram bot token
CHAT_ID = "5625785220"  # Your Telegram chat ID


bot = Bot(token=TELEGRAM_BOT_TOKEN)

async def send_notification():
    await bot.send_message(chat_id=CHAT_ID, text="The Process patient_test_TROTS is Finished")

async def send_notification_error():
    await bot.send_message(chat_id=CHAT_ID, text="The Process patient_test_TROTS is Finished with error")



# # The following block is for running in the background================================================
if __name__ == '__main__':
  with open('patient_test_TROTS3rd.txt', 'w') as f:
    with redirect_stdout(f):
        try:
          # BLAS setup
          os.environ['OMP_NUM_THREADS'] = '1'
          os.environ['MKL_NUM_THREADS'] = '1'

          # Setup
          args = parser.parse_args()
          # Creating directories.
          save_dir = os.path.join('results_test', args.name)
          if not os.path.exists(save_dir):
            os.makedirs(save_dir)
          print(' ' * 26 + 'Options')

          # Saving parameters
          with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
            for k, v in vars(args).items():
              print(' ' * 26 + k + ': ' + str(v))
              f.write(k + ' : ' + str(v) + '\n')

          mp.set_start_method(platform.python_version()[0] == '3' and 'spawn' or 'fork')  # Force true spawning (not forking) if available
          torch.manual_seed(args.seed)
          T = Counter()  # Global shared counter
          gym.logger.set_level(gym.logger.ERROR)  # Disable Gym warnings

          env = TreatmentEnv()

          shared_model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)

          env.close()

          fields = ['t', 'rewards', 'avg_steps', 'time']
          with open(os.path.join(save_dir, 'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)


          # Start validation agent
          processes = []
          p = mp.Process(target=test, args=(0, args, T))
          p.start()
          p.join()
          print("args.evaluate",args.evaluate)
          asyncio.run(send_notification())

        except Exception as e:
        # Handle or ignore the error here
          asyncio.run(send_notification_error())
          pass

