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
from math import sqrt
from torch.distributions import Categorical
import numpy as np
from gym import spaces
import random
from collections import deque
import h5sparse
import h5py
from scipy.sparse import vstack
from typing import List
from scipy.sparse import csr_matrix
import numpy.linalg as LA
import time
from lib_dvh.data_prep import loadDoseMatrix,loadMask,ProcessDmat
from gym import Env
from gym.spaces import Discrete


pdose = 1 # target dose for PTV
maxiter = 40 # maximum iteration number for treatment planing optimization
INPUT_SIZE = 100  # DVH interval number
patient_list = ['007','011']
start_time = time.time()

class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
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
    ind = round(0.03 / 0.015) - 1

    max_limit = 1

    avg_DPTV = (DPTV[ind] + DPTV[ind + 1] + DPTV[ind - 1]) / 3
    score2 =  (avg_DPTV - 1.1)/(-0.03)
    if score2 > max_limit:
        score2 = 1
    if score2 < 0:
        score2 = 0
    delta2 = 0.08
    if (avg_DPTV > 1.05):
        score2_fine = (1 / pi * np.arctan(-(avg_DPTV - 1.075) / delta2) + 0.5) * 8
    else:
        score2_fine = 6########################################

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
    DPTV = MPTV.dot(xVec)
    DPTV1 = np.sort(DPTV)
    posi = int(round(0.05 * DPTV1.shape[0]))
    D95 = DPTV1[posi]
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

        # How many times it will loop per epoch
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
            runOpt_dvh(MPTV, MBLA1, MREC1, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,
                       gamma, pdose, maxiter)
        Score_fine1, Score1, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)

        extra = 0 if Score1 != 9 else 2

        # Uncomment this part original scoring system
        reward = (Score_fine1 - Score_fine) + (Score1 - Score) * 4
        Done = False
        if Score1 == 9:
            Done = True
        return n_state, reward, Score_fine1, Score1, scoreall, Done, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec

    def reset(self):
        self.state = self.observation_space
        self.time_limit = 30
        return self.state

    def close(self):
        pass


class SharedRMSprop(optim.RMSprop):
  def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
    super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False)

    # State initialisation (must be done before step, else will not be shared between threads)
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = p.data.new().resize_(1).zero_()
        state['square_avg'] = p.data.new().resize_as_(p.data).zero_()

  def share_memory(self):
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'].share_memory_()
        state['square_avg'].share_memory_()

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        state = self.state[p]

        square_avg = state['square_avg']
        alpha = group['alpha']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], p.data)

        # g = αg + (1 - α)Δθ^2
        # square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        # θ ← θ - ηΔθ/√(g + ε)
        avg = square_avg.sqrt().add_(group['eps'])
        # p.data.addcdiv_(-group['lr'], grad, avg)
        p.data.addcdiv_(grad, avg, value=-group['lr'])

    return loss


class EpisodicReplayMemory():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity // max_episode_length
    self.memory = deque(maxlen=self.num_episodes)
    self.trajectory = []

  def append(self, state, action, reward, policy):
    self.trajectory.append(Transition(state, action, reward, policy))  # Save s_i, a_i, r_i+1, µ(·|s_i)
    # Terminal states are saved with actions as None, so switch to next episode
    if action is None:
      self.memory.append(self.trajectory)
      self.trajectory = []
  # Samples random trajectory
  def sample(self, maxlen=0):
    mem = self.memory[random.randrange(len(self.memory))]
    T = len(mem)
    # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
    if maxlen > 0 and T > maxlen + 1:
      t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
      return mem[t:t + maxlen + 1]
    else:
      return mem

  # Samples batch of trajectories, truncating them to the same length
  def sample_batch(self, batch_size, maxlen=0):
    batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
    minimum_size = min(len(trajectory) for trajectory in batch)
    batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
    return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

  def length(self):
    # Return number of epsiodes saved in memory
    return len(self.memory)

  def __len__(self):
    return sum(len(episode) for episode in self.memory)


# Converts a state from the OpenAI Gym (a numpy array) to a batch tensor
def state_to_tensor(state):
  return torch.from_numpy(state).float().unsqueeze(0)


# Knuth's algorithm for generating Poisson samples
def _poisson(lmbd):
  L, k, p = math.exp(-lmbd), 0, 1
  while p > L:
    k += 1
    p *= random.uniform(0, 1)
  return max(k - 1, 0)


# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
  for param, shared_param in zip(model.parameters(), shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad


# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
  for param_group in optimiser.param_groups:
    param_group['lr'] = lr


# Updates networks
def _update_networks(args, T, model, shared_model, shared_average_model, loss, optimiser):

  optimiser.zero_grad()

  loss.backward()
  # Gradient L2 normalisation
  nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)

  # Transfer gradients to shared model and update
  _transfer_grads_to_shared_model(model, shared_model)
  optimiser.step()
  if args.lr_decay:
    # Linearly decay learning rate
    _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))

  # Update shared_average_model
  for shared_param, shared_average_param in zip(shared_model.parameters(), shared_average_model.parameters()):
    shared_average_param = args.trust_region_decay * shared_average_param + (1 - args.trust_region_decay) * shared_param


# Computes an "efficient trust region" loss (policy head only) based on an existing loss and two distributions
def _trust_region_loss(model, distribution, ref_distribution, loss, threshold, g, k):
  kl = - (ref_distribution * (distribution.log() - ref_distribution.log())).sum(1).mean(0)

  # Compute dot products of gradients
  k_dot_g = (k * g).sum(1).mean(0)
  k_dot_k = (k ** 2).sum(1).mean(0)
  # Compute trust region update
  if k_dot_k.item() > 0:
    trust_factor = ((k_dot_g - threshold) / k_dot_k).clamp(min=0).detach()
  else:
    trust_factor = torch.zeros(1)
  # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
  trust_loss = loss + trust_factor * kl

  return trust_loss


# Trains model
def _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret,
           average_policies, old_policies=None):
  off_policy = old_policies is not None
  action_size = policies[0].size(1)
  policy_loss, value_loss = 0, 0

  # Calculate n-step returns in forward view, stepping backwards from the last state
  t = len(rewards)
  for i in reversed(range(t)):
    # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
    if off_policy:
      rho = policies[i].detach() / old_policies[i]
    else:
      rho = torch.ones(1, action_size)

    # Qret ← r_i + γQret
    Qret = rewards[i] + args.discount * Qret
    # Advantage A ← Qret - V(s_i; θ)
    A = Qret - Vs[i]

    # Log policy log(π(a_i|s_i; θ))
    log_prob = policies[i].gather(1, actions[i]).log()
    # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
    single_step_policy_loss = -(rho.gather(1, actions[i]).clamp(max=args.trace_max) * log_prob * A.detach()).mean(
      0)  # Average over batch
    # Off-policy bias correction
    if off_policy:
      # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
      bias_weight = (1 - args.trace_max / rho).clamp(min=0) * policies[i]
      single_step_policy_loss -= (
                bias_weight * policies[i].log() * (Qs[i].detach() - Vs[i].expand_as(Qs[i]).detach())).sum(1).mean(0)
    if args.trust_region:
      # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
      k = -average_policies[i].gather(1, actions[i]) / (policies[i].gather(1, actions[i]) + 1e-10)
      if off_policy:
        g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A / (policies[i] + 1e-10).gather(1, actions[i]) \
             + (bias_weight * (Qs[i] - Vs[i].expand_as(Qs[i])) / (policies[i] + 1e-10)).sum(1)).detach()
      else:
        g = (rho.gather(1, actions[i]).clamp(max=args.trace_max) * A / (policies[i] + 1e-10).gather(1, actions[
          i])).detach()
      # Policy update dθ ← dθ + ∂θ/∂θ∙z*
      policy_loss += _trust_region_loss(model, policies[i].gather(1, actions[i]) + 1e-10,
                                        average_policies[i].gather(1, actions[i]) + 1e-10, single_step_policy_loss,
                                        args.trust_region_threshold, g, k)
    else:
      # Policy update dθ ← dθ + ∂θ/∂θ∙g
      policy_loss += single_step_policy_loss

    # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
    policy_loss -= args.entropy_weight * -(policies[i].log() * policies[i]).sum(1).mean(
      0)  # Sum over probabilities, average over batch

    # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
    Q = Qs[i].gather(1, actions[i])
    value_loss += ((Qret - Q) ** 2 / 2).mean(0)  # Least squares loss

    # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
    truncated_rho = rho.gather(1, actions[i]).clamp(max=1)
    # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()

  # Update networks
  _update_networks(args, T, model, shared_model, shared_average_model, policy_loss + value_loss, optimiser)


def train2(rank, args, T, shared_model, shared_average_model, optimiser):
  torch.manual_seed(args.seed + rank)

  env = TreatmentEnv()
  # env.seed(args.seed + rank)
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  # print("model", model)
  model.train()

  if not args.on_policy:
    # Normalise memory capacity by number of training processes
    memory = EpisodicReplayMemory(args.memory_capacity // args.num_processes, args.max_episode_length)

  test_set = ['01']

  #   '------------------------------------------ validation ----------------------------------------------------')

  pid = ['001']

  data_path = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/'
  data_path2 = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/'

  data_result_pathTr = '/data2/BlaFixed/Train/'
  os.makedirs(data_result_pathTr, exist_ok = True)
  save_dir = '/data2/BlaFixed/Train/parameters/'
  os.makedirs(save_dir, exist_ok = True)
  data_result_pathTrInputs = '/data2/BlaFixed/Train/Inputs/'
  os.makedirs(data_result_pathTrInputs, exist_ok = True)

  for i in range(len(pid)):
    print("len(pid)", len(pid))
    globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path + str(pid[i]) + '.hdf5')
    print("doseMatrix loaded")
    globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()[
      'rectumLabel' + str(i)], \
    globals()['PTVLabel' + str(i)] = loadMask(data_path2 + str(pid[i]) + '.h5')
    print("PTVLabel loaded")
    print(globals()['doseMatrix_' + str(i)].shape)

  # Comment this when you have more than one test cases
  testcase = 0
  doseMatrix = globals()['doseMatrix_' + str(testcase)]
  targetLabels = globals()['targetLabels_' + str(testcase)]
  bladderLabel = globals()['bladderLabel' + str(testcase)]
  rectumLabel = globals()['rectumLabel' + str(testcase)]
  PTVLabel = globals()['PTVLabel' + str(testcase)]
  MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel)


  t = 1  # Thread step counter
  done = True  # Start new episode


  while T.value() <= args.T_max:
    while True:
      # Sync with shared model at least every t_max steps
      # print("loading model started")
      epoch = T.value()
      model.load_state_dict(shared_model.state_dict())
      if epoch % 250 < 15 or epoch % 250 > 235:
        torch.save(model.state_dict(),
                       os.path.join(save_dir,'episode' + str(epoch) + '.pth'))
      # print("model loaded")
      # Get starting timestep
      t_start = t
      epoch = T.value()

      # Reset or pass on hidden state
      if done:
        # print("inside if done:")
        hx, avg_hx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
        # print("hx created")
        cx, avg_cx = torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size)
        # print("cx created")

        reward_sum_total = 0
        qvalue_sum = 0
        num_q = 0
        loss_perepoch = []
        doseMatrix = []
        targetLabels = []
        bladderLabel = []
        rectumLabel = []
        PTVLabel = []

        epsilon = 1e-10
        tPTV = random.uniform(1+epsilon,1.2-epsilon)
        tBLA = random.uniform(0+epsilon,1-epsilon)
        tREC = random.uniform(0+epsilon,1-epsilon)
        lambdaPTV = random.uniform(0+epsilon,30-epsilon)
        lambdaBLA = random.uniform(0+epsilon,30-epsilon)
        lambdaREC = random.uniform(0+epsilon,30-epsilon)
        VPTV = random.uniform(0+epsilon,0.3-epsilon)
        VBLA = random.uniform(0+epsilon,1-epsilon)
        VREC = random.uniform(0+epsilon,1-epsilon)
        step_count = 0
        # --------------------- solve treatment planning optmization -----------------------------
        xVec = np.ones((MPTV.shape[1],))
        gamma = np.zeros((MPTV.shape[0],))
        state, _, xVec = \
          runOpt_dvh(MPTV, MBLA1, MREC1, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC,
                     VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter)

        ################ Uncomment: For traning results #####################
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

        state = state_to_tensor(state)
        # print("Training score")
        Score_fine, Score, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)

        tPTV_all[0] = tPTV
        tBLA_all[0] = tBLA
        tREC_all[0] = tREC
        lambdaPTV_all[0] = lambdaPTV
        lambdaBLA_all[0] = lambdaBLA
        lambdaREC_all[0] = lambdaREC
        VPTV_all[0] = VPTV
        VBLA_all[0] = VBLA
        VREC_all[0] = VREC
        planScore_all[0] = Score
        planScore_fine_all[0] = Score_fine

        # print("Initial score:",Score)
        # np.save(data_result_pathTr+'patient'+str(testcase)+'step'+str(0)+'epoch'+str(epoch)+ 'scoreall', scoreall)
        done, episode_length = False, 0
      else:
        # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
        hx = hx.detach()
        cx = cx.detach()

      # Lists of outputs for training
      policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []

      while not done and t - t_start < args.t_max:
        policy, Q, V, (hx, cx) = model(state, (hx, cx))
        policySave = policy.detach().numpy()
        hxSave = hx.detach().numpy()
        cxSave = cx.detach().numpy()
        average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))
        average_policySave = average_policy.detach().numpy()
        avg_hxSave = avg_hx.detach().numpy()
        avg_cxSave = avg_cx.detach().numpy()
        # Sample action
        action = torch.multinomial(policy, 1)[0, 0]
        actionSave = action.item()
        # print("action",action)
        j = 1 #some random number as it is not used

        next_state, reward, Score_fine, Score, scoreall, done, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec = env.step(
          action.item(), j, Score_fine, Score, MPTV, MBLA, MREC, MBLA1, MREC1, tPTV, tBLA, tREC, lambdaPTV,
          lambdaBLA, lambdaREC, VPTV, VBLA, VREC, pdose, maxiter)

        next_state = state_to_tensor(next_state)
        reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards

        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        print("Train:{}, {},{}\n".format(rank, T.value(), episode_length))
        episode_length += 1  # Increase episode counter

        tPTV_all[episode_length] = tPTV
        tBLA_all[episode_length] = tBLA
        tREC_all[episode_length] = tREC
        lambdaPTV_all[episode_length] = lambdaPTV
        lambdaBLA_all[episode_length] = lambdaBLA
        lambdaREC_all[episode_length] = lambdaREC
        VPTV_all[episode_length] = VPTV
        VBLA_all[episode_length] = VBLA
        VREC_all[episode_length] = VREC
        planScore_all[episode_length] = Score
        planScore_fine_all[episode_length] = Score_fine

        if not args.on_policy:
          memory.append(state, action, reward, policy.detach())  # Save just tensors
        [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies),
                                           (policy, Q, V, torch.LongTensor([[action]]), torch.Tensor([[reward]]), average_policy))]

        # Increment counters
        t += 1

        T.increment()

        # Update state
        state = next_state

      # Break graph for last values calculated (used for targets, not directly as model outputs)
      if done:
        # Qret = 0 for terminal s
        Qret = torch.zeros(1, 1)

        if epoch % 250 < 15 or epoch % 250 > 235:
          name1 = data_result_pathTr + str(testcase) + 'tpptuning' + str(epoch)
          np.savez(name1+'.npz',l1 = tPTV_all, l2 =tBLA_all,  l3 = tREC_all, l4 =lambdaPTV_all, l5 =lambdaBLA_all, l6 =lambdaREC_all, l7 = VPTV_all, l8= VBLA_all, l9 = VREC_all, l10 = planScore_all, l11 = planScore_fine_all)

        if not args.on_policy:
          # Save terminal state for offline training
          memory.append(state, None, None, None)
          # print("memory.appended")
      else:
        # Qret = V(s_i; θ) for non-terminal s
        _, _, Qret, _ = model(state, (hx, cx))
        Qret = Qret.detach()

      _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret, average_policies)
      # print("on policy training ended")

      # Finish on-policy episode
      if done:
        break

    # Train the network off-policy when enough experience has been collected
    if not args.on_policy and len(memory) >= args.replay_start:
      # Sample a number of off-policy episodes based on the replay ratio
      for epochOffine in range(_poisson(args.replay_ratio)):
        # Act and train off-policy for a batch of (truncated) episode
        trajectories = memory.sample_batch(args.batch_size, maxlen=args.t_max)

        # Reset hidden state
        hx, avg_hx = torch.zeros(args.batch_size, args.hidden_size), torch.zeros(args.batch_size, args.hidden_size)
        cx, avg_cx = torch.zeros(args.batch_size, args.hidden_size), torch.zeros(args.batch_size, args.hidden_size)

        # Lists of outputs for training
        policies, Qs, Vs, actions, rewards, old_policies, average_policies = [], [], [], [], [], [], []

        # Loop over trajectories (bar last timestep)
        for i in range(len(trajectories) - 1):
          # Unpack first half of transition
          state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i]), 0)
          action = torch.LongTensor([trajectory.action for trajectory in trajectories[i]]).unsqueeze(1)
          reward = torch.Tensor([trajectory.reward for trajectory in trajectories[i]]).unsqueeze(1)
          old_policy = torch.cat(tuple(trajectory.policy for trajectory in trajectories[i]), 0)

          stateSave = state.numpy()

          # Calculate policy and values
          policy, Q, V, (hx, cx) = model(state, (hx, cx))
          average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(state, (avg_hx, avg_cx))
          policySave = policy.detach().numpy()
          actionSave = action.detach().numpy()
          hxSave = hx.detach().numpy()
          cxSave = cx.detach().numpy()
          average_policySave = average_policy.detach().numpy()
          avg_hxSave = avg_hx.detach().numpy()
          avg_cxSave = avg_cx.detach().numpy()

          # Save outputs for offline training
          [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies, old_policies),
                                             (policy, Q, V, action, reward, average_policy, old_policy))]

          # Unpack second half of transition
          next_state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i + 1]), 0)
          done = torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1)

        # Do forward pass for all transitions
        _, _, Qret, _ = model(next_state, (hx, cx))
        # Qret = 0 for terminal s, V(s_i; θ) otherwise
        Qret = ((1 - done) * Qret).detach()

        # Train the network off-policy
        _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs,
               actions, rewards, Qret, average_policies, old_policies=old_policies)

      print("replay ratio run")

    done = True

  env.close()

import matplotlib.pyplot as plt
def test(rank, args, T, shared_model):
    print("testing starts")
    torch.manual_seed(args.seed + rank)

    cuda = False

    # save_dir = os.path.join('results', args.name)
    save_dir = os.path.join('/data2/','NetworkSavePath', args.name)
    os.makedirs(save_dir, exist_ok =True)

    can_test = True  # Test flag
    t_start = 1  # Test step counter to check against global counter
    rewards, steps = [], []  # Rewards and steps for plotting
    l = str(len(str(args.T_max)))  # Max num. of digits for logging steps

    ###############################################################################################

    test_set = patient_list
    # logging.info(
    #     '------------------------------------------ validation ----------------------------------------------------')
    i = 0

    env = TreatmentEnv()
    # env.seed(args.seed + rank)
    model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
    model.eval()

    done = True  # Start new episode

    # stores step, reward, avg_steps and time
    results_dict = {'t': [], 'reward': [], 'avg_steps': [], 'time': []}

    # Loading of all MPTV and other files
    pid = patient_list
    # data_path = './lib_dvh/f_dijs/0'
    data_path = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/'
    data_path2 = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/'

    for i in range(len(pid)):
        print("len(pid)", len(pid))
        print("loading patient:",pid[i])
        globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path + str(pid[i]) + '.hdf5')
        # print("doseMatrix loaded")
        globals()['targetLabels_' + str(i)], globals()['bladderLabel' + str(i)], globals()[
            'rectumLabel' + str(i)], \
        globals()['PTVLabel' + str(i)] = loadMask(data_path2 + str(pid[i]) + '.h5')
        print("PTVLabel loaded")

    while T.value() <= args.T_max:
        if can_test:
            t_start = T.value()  # Reset counter

            if T.value() == 0:
                epoch = 0
            else:
                # epoch = T.value() - 1
                epoch = T.value()

            model.load_state_dict(shared_model.state_dict())
            model.eval()
            torch.save(model.state_dict(),
                       os.path.join(save_dir,'episode' + str(epoch) + '.pth'))  # Save model params

            # ------------- range of parmaeter -----------------
            paraMax = 100000  # change in validation as well
            paraMin = 0
            paraMax_tPTV = 1.2
            paraMin_tPTV = 1
            paraMax_tOAR = 1
            paraMax_VOAR = 1
            paraMax_VPTV = 0.3
            # ---------------------------------------------------

            # Evaluate over several episodes and average results
            avg_rewards, avg_episode_lengths = [], []
            for i in range(args.evaluation_episodes):
                # print("i",i)
                sampleid = i
                id = test_set[sampleid]
                doseMatrix = globals()['doseMatrix_' + str(sampleid)]
                targetLabels = globals()['targetLabels_' + str(sampleid)]
                bladderLabel = globals()['bladderLabel' + str(sampleid)]
                rectumLabel = globals()['rectumLabel' + str(sampleid)]
                PTVLabel = globals()['PTVLabel' + str(sampleid)]

                MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, bladderLabel,
                                                             rectumLabel)

                while True:
                    # Reset or pass on hidden state
                    if done:

                        hx = torch.zeros(1, args.hidden_size)
                        cx = torch.zeros(1, args.hidden_size)

                        done, episode_length = False, 0
                        reward_sum = 0

                        env = TreatmentEnv()
                        # ------------------------ initial paramaters & input --------------------
                        tPTV = 1
                        tBLA = 1
                        tREC = 1
                        lambdaPTV = 1
                        lambdaBLA = 1
                        lambdaREC = 1
                        VPTV = 0.1
                        VBLA = 1
                        VREC = 1
                        xVec = np.ones((MPTV.shape[1],))
                        gamma = np.zeros((MPTV.shape[0],))
                        # --------------------- solve treatment planning optmization -----------------------------
                        state_test0, iter, xVec = \
                            runOpt_dvh(MPTV, MBLA1, MREC1, tPTV, tBLA, tREC,
                                       lambdaPTV, lambdaBLA, lambdaREC,
                                       VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter)
                        # --------------------- generate input for NN -----------------------------
                        DPTV = MPTV.dot(xVec)
                        DBLA = MBLA.dot(xVec)
                        DREC = MREC.dot(xVec)
                        DPTV = np.sort(DPTV)
                        DPTV = np.flipud(DPTV)
                        DBLA = np.sort(DBLA)
                        DBLA = np.flipud(DBLA)
                        DREC = np.sort(DREC)
                        DREC = np.flipud(DREC)
                        edge_ptv = np.zeros((1000 + 1,))
                        edge_ptv[1:1000 + 1] = np.linspace(0, max(DPTV), 1000)
                        x_ptv = np.linspace(0.5 * max(DPTV) / 1000, max(DPTV), 1000)
                        (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                        y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

                        edge_bladder = np.zeros((1000 + 1,))
                        edge_bladder[1:1000 + 1] = np.linspace(0, max(DBLA), 1000)
                        x_bladder = np.linspace(0.5 * max(DBLA) / 1000, max(DBLA), 1000)
                        (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                        y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

                        edge_rectum = np.zeros((1000 + 1,))
                        edge_rectum[1:1000 + 1] = np.linspace(0, max(DREC), 1000)
                        x_rectum = np.linspace(0.5 * max(DREC) / 1000, max(DREC), 1000)
                        (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                        y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

                        Y = np.zeros((1000, 3))
                        Y[:, 0] = y_ptv
                        Y[:, 1] = y_bladder
                        Y[:, 2] = y_rectum

                        X = np.zeros((1000, 3))
                        X[:, 0] = x_ptv
                        X[:, 1] = x_bladder
                        X[:, 2] = x_rectum

                        # data_result_path='./data/data/Results/GPU2AC/general/37/'
                        data_result_path = '/data2/BlaFixed/validation/'
                        os.makedirs(data_result_path, exist_ok = True)

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
                        # ----------------two ways of NN schemes (with/without rules) --------------------------
                        state_test = state_test0
                        if cuda:
                            state_test = torch.from_numpy(state_test).float()
                            state_test = state_test.to(device)

                        state_test = state_to_tensor(state_test)
                        # ------------------------ initial paramaters & input --------------------
                        tPTV = 1
                        tBLA = 1
                        tREC = 1
                        lambdaPTV = 1
                        lambdaBLA = 1
                        lambdaREC = 1
                        VPTV = 0.1
                        VBLA = 1
                        VREC = 1

                        np.save(data_result_path+'patient'+str(id)+'step'+str(episode_length)+'epoch'+str(epoch)+ 'scoreall', scoreall)
                        np.save(data_result_path+'patient'+str(id)+'step'+str(episode_length)+'epoch'+str(epoch) +'xDVHY', state_test0)

                    # Calculate policy
                    with torch.no_grad():
                        policy, _, _, (hx, cx) = model(state_test, (hx, cx))

                    policySave = policy.numpy()
                    hxSave = hx.numpy()
                    cxSave = cx.numpy()

                    np.save(data_result_path+'patient'+str(id)+'step'+str(episode_length)+'epoch'+str(epoch) +'Policy', policySave)
                    np.save(data_result_path+'patient'+str(id)+'step'+str(episode_length+1)+'epoch'+str(epoch) +'hx', hxSave)
                    np.save(data_result_path+'patient'+str(id)+'step'+str(episode_length+1)+'epoch'+str(epoch) +'cx', cxSave)

                    # Choose action greedily
                    action = torch.multinomial(policy, 1)[0, 0]

                    actionSave = action.item()
                    np.save(data_result_path+'patient'+str(id)+'step'+str(episode_length)+'epoch'+str(epoch) +'action', actionSave)


                    t = 1
                    ###################################################################
                    ####################################################################

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

                    # --------------------- solve treatment planning optmization -----------------------------
                    xVec = np.ones((MPTV.shape[1],))
                    gamma = np.zeros((MPTV.shape[0],))

                    n_state_test, iter, xVec = \
                        runOpt_dvh(MPTV, MBLA1, MREC1, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA,
                                   VREC, xVec,
                                   gamma, pdose, maxiter)

                    planScore_fine, planScore, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)
                    print("Test:{},{} ,{}".format(

                                                                                                       epoch,
                                                                                                       planScore,
                                                                                                       planScore_fine))

                    np.save(data_result_path+'patient'+str(id)+'step'+str(episode_length+1)+'epoch'+str(epoch)+ 'scoreall', scoreall)
                    np.save(data_result_path+'patient'+str(id)+'step'+str(episode_length+1)+'epoch'+str(epoch) +'xDVHY', n_state_test)

                    state_test = n_state_test
                    if cuda:
                        state_test = torch.from_numpy(state_test).float()
                        state_test = state_test.to(device)

                    state_test = state_to_tensor(state_test)

                    j = episode_length

                    tPTV_all[j + 1] = tPTV
                    # print("tPTV_all", tPTV_all)
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

                    DPTV = MPTV.dot(xVec)
                    DBLA = MBLA.dot(xVec)
                    DREC = MREC.dot(xVec)
                    DPTV = np.sort(DPTV)
                    DPTV = np.flipud(DPTV)
                    DBLA = np.sort(DBLA)
                    DBLA = np.flipud(DBLA)
                    DREC = np.sort(DREC)
                    DREC = np.flipud(DREC)
                    edge_ptv = np.zeros((1000 + 1,))
                    edge_ptv[1:1000 + 1] = np.linspace(0, max(DPTV), 1000)
                    x_ptv = np.linspace(0.5 * max(DPTV) / 1000, max(DPTV), 1000)
                    (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                    y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

                    edge_bladder = np.zeros((1000 + 1,))
                    edge_bladder[1:1000 + 1] = np.linspace(0, max(DBLA), 1000)
                    x_bladder = np.linspace(0.5 * max(DBLA) / 1000, max(DBLA), 1000)
                    (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                    y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

                    edge_rectum = np.zeros((1000 + 1,))
                    edge_rectum[1:1000 + 1] = np.linspace(0, max(DREC), 1000)
                    x_rectum = np.linspace(0.5 * max(DREC) / 1000, max(DREC), 1000)
                    (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                    y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

                    Y = np.zeros((1000, 6))
                    Y[:, 0] = y_ptv
                    Y[:, 1] = y_bladder
                    Y[:, 2] = y_rectum
                    # X = np.zeros((1000, 3))
                    Y[:, 3] = x_ptv
                    Y[:, 4] = x_bladder
                    Y[:, 5] = x_rectum

                    check_model2 = model.state_dict()

                    done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length

                    episode_length += 1  # Increase episode counter

                    if planScore == 9:
                        done = True
                        break

                    # Log and reset statistics at the end of every episode
                    if done:
                       break

                name1 = data_result_path + id + 'tpptuning' + str(epoch)
                np.savez(name1+'.npz',l1 = tPTV_all, l2 =tBLA_all,  l3 = tREC_all, l4 =lambdaPTV_all, l5 =lambdaBLA_all, l6 =lambdaREC_all, l7 = VPTV_all, l8= VBLA_all, l9 = VREC_all, l10 = planScore_all, l11 = planScore_fine_all)

            # Dumping the results in pickle format
            with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
                pickle.dump(results_dict, f)

            if args.evaluate:
                return

            steps.append(t_start)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))  # Save model params

            can_test = False  # Finish testing

        else:
            if T.value() - t_start >= args.evaluation_interval:
                can_test = True

        time.sleep(0.001)  # Check if available to test every millisecond

    # Dumping the results in pickle format

    with open(os.path.join(save_dir, 'results.pck'), 'wb') as f:
        pickle.dump(results_dict, f)
    env.close()

# Global counter
class Counter():
  def __init__(self):
    self.val = mp.Value('i', 0)
    self.lock = mp.Lock()

  def increment(self):
    with self.lock:
      self.val.value += 1

  def value(self):
    with self.lock:
      return self.val.value




#
# Plots min, max and mean + standard deviation bars of a population over time
def plot_line(xs, ys_population, save_dir):
  max_colour = 'rgb(0, 132, 180)'
  mean_colour = 'rgb(0, 172, 237)'
  std_colour = 'rgba(29, 202, 255, 0.2)'

  ys = torch.tensor(ys_population)
  ys_min = ys.min(1)[0].squeeze()
  ys_max = ys.max(1)[0].squeeze()
  ys_mean = ys.mean(1).squeeze()
  ys_std = ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = go.Scatter(x=xs, y=ys_max.numpy(), mode='lines', line=dict(color=max_colour, dash='dash'), name='Max')
  trace_upper = go.Scatter(x=xs, y=ys_upper.numpy(), mode='lines', marker=dict(color="#444"), line=dict(width=0), name='+1 Std. Dev.', showlegend=False)
  trace_mean = go.Scatter(x=xs, y=ys_mean.numpy(), mode='lines', line=dict(color=mean_colour), name='Mean')
  trace_lower = go.Scatter(x=xs, y=ys_lower.numpy(), mode='lines', marker=dict(color="#444"), line=dict(width=0), fill='tonexty', fillcolor=std_colour, name='-1 Std. Dev.', showlegend=False)
  trace_min = go.Scatter(x=xs, y=ys_min.numpy(), mode='lines', line=dict(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_mean, trace_upper, trace_lower, trace_min, trace_max],
    'layout': dict(title='Rewards',
                   xaxis={'title': 'Step'},
                   yaxis={'title': 'Average Reward'})
  }, filename=os.path.join(save_dir, 'rewards.html'), auto_open=False)


# Original arguments to use
parser = argparse.ArgumentParser(description='ACER')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--num-processes', type=int, default=3, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=250000, metavar='STEPS', help='Number of training steps')
# parser.add_argument('--T-max', type=int, default=100, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
# parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--max-episode-length', type=int, default=20, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--on-policy', action='store_true', help='Use pure on-policy training (A3C)')
parser.add_argument('--memory-capacity', type=int, default=100000, metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
parser.add_argument('--replay-start', type=int, default=20000, metavar='EPISODES', help='Number of transitions to save before starting off-policy training')
# parser.add_argument('--replay-start', type=int, default=100, metavar='EPISODES', help='Number of transitions to save before starting off-policy training')
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
parser.add_argument('--evaluation-episodes', type=int, default=2, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--name', type=str, default='results', help='Save folder')
parser.add_argument('--env', type=str, default='CartPole-v1',help='environment name')


if __name__ == '__main__':
  # BLAS setup
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'

  # Setup
  args = parser.parse_args()
  # Creating directories.
  save_dir = os.path.join('results', args.name)
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
  shared_model.share_memory()
  if args.model and os.path.isfile(args.model):
    # Load pretrained weights
    shared_model.load_state_dict(torch.load(args.model))
  # Create average network
  shared_average_model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  shared_average_model.load_state_dict(shared_model.state_dict())
  shared_average_model.share_memory()
  for param in shared_average_model.parameters():
    param.requires_grad = False
  # Create optimiser for shared network parameters with shared statistics
  optimiser = SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha=args.rmsprop_decay)
  optimiser.share_memory()
  env.close()

  fields = ['t', 'rewards', 'avg_steps', 'time']
  with open(os.path.join(save_dir, 'test_results.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(fields)


  # Start validation agent
  processes = []
  p = mp.Process(target=test, args=(0, args, T, shared_model))
  p.start()
  processes.append(p)
  print("args.evaluate",args.evaluate)

  if not args.evaluate:
    # Start training agents
    for rank in range(1, args.num_processes + 1):
      p = mp.Process(target=train2, args=(
      rank, args, T, shared_model, shared_average_model, optimiser))
      p.start()
      print('Process ' + str(rank) + ' started')
      processes.append(p)
  # #
  # Clean up
  for p in processes:
    p.join()

  print("Program completed in:",time.time() - start_time)
