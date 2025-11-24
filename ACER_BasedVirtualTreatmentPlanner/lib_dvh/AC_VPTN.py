import numpy as np
from gym import Env
from gym.spaces import Discrete
import matplotlib.pyplot as plt
import math as m

# from lib_dvh.AC_networks import ActorCriticNetwork
from lib_dvh.data_prep import ProcessDmat
from lib_dvh.TP_DVH_algo import runOpt_dvh
from lib_dvh.score_calcu import planIQ_train
# from lib_dvh.action import action_original
# from lib_dvh.action import action_new
# from .myconfig import *

INPUT_SIZE = 100  # DVH interval number


class TreatmentEnv(Env):
    """A Treatment planning environment for OpenAI gym"""

    # metedata = {'render.modes':['human']}
    # Set up the dimensions and type of space that the action and observation space is
    def __init__(self):
        self.action_space = Discrete(18)  # Box(low=np.array([0,0.5]), high=np.array([26,1.5]), dtype=np.float32)#Discrete (26)
        self.observation_space = np.zeros([300])

        # How many times it will loop per epoch
        self.time_limit = 30

    def step(self, action, t, Score_fine, Score, MPTV, MBLA, MREC, MBLA1, MREC1, tPTV, tBLA, tREC,
             lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, pdose, maxiter):
        # The environment that the agent will work in includes both the Treatment planning system and the Reward function

        # Uncomment this part for action_original
        # xVec = np.ones((MPTV.shape[1],))
        # gamma = np.zeros((MPTV.shape[0],))
        # _, _, xVec, _ = \
        #     runOpt_dvh(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,
        #                gamma, pdose, maxiter)
        # state = np.reshape(state, [INPUT_SIZE * 3])
        # xVec = np.ones((MPTV.shape[1],))
        # Score_fine, Score, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)
        self.action = action
        # print("action:", action)
        # Uncomment this part for original action and tab the code after else and before DPTV. Also, change actionnum in myconfig file
        # if action % 3 == 1:
        #     n_state = state
        #     # print('This is still the same')
        #     reward = 0
        #     action_factor = 1
        # else:
        paraMax = 100000  # change in validation as well
        paraMin = 0
        paraMax_tPTV = 1.2
        paraMin_tPTV = 1
        paraMax_tOAR = 1
        paraMax_VOAR = 1
        paraMax_VPTV = 0.3

        increment_PTV = 1.1
        decrement_PTV = 0.9
        PTV_e_increment = 0.1
        increment = 1.1
        decrement = 0.9
        e_increment = 0.1

        if action == 0:
            tPTV = tPTV * increment_PTV if tPTV < paraMax_tPTV else paraMax_tPTV

        elif action == 1:
            tPTV = tPTV * decrement_PTV if tPTV > paraMin_tPTV else paraMin_tPTV

        elif action == 2:
            tBLA = tBLA * increment if tBLA < paraMax_tOAR else paraMax_tOAR

        elif action == 3:
            tBLA = tBLA * decrement if tBLA > paraMin else paraMin

        elif action == 4:
            tREC = tREC * increment if tREC < paraMax_tOAR else paraMax_tOAR

        elif action == 5:
            tREC = tREC * decrement if tREC > paraMin else paraMin

        elif action == 6:
            lambdaPTV = lambdaPTV * m.exp(PTV_e_increment) if lambdaPTV < paraMax else paraMax

        elif action == 7:
            lambdaPTV = lambdaPTV * m.exp(-PTV_e_increment) if lambdaPTV > paraMin else paraMin

        elif action == 8:
            lambdaBLA = lambdaBLA * m.exp(e_increment) if lambdaBLA < paraMax else paraMax

        elif action == 9:
            print("action 9 chosen")
            lambdaBLA = lambdaBLA * m.exp(-e_increment) if lambdaBLA > paraMin else paraMin

        elif action == 10:
            lambdaREC = lambdaREC * m.exp(e_increment) if lambdaREC < paraMax else paraMax

        elif action == 11:
            lambdaREC = lambdaREC * m.exp(-e_increment) if lambdaREC > paraMin else paraMin

        elif action == 12:
            VPTV = VPTV * increment_PTV if VPTV < paraMax_VPTV else paraMax_VPTV

        elif action == 13:
            VPTV = VPTV * decrement_PTV if VPTV > paraMin else paraMin

        elif action == 14:
            VBLA = VBLA * increment if VBLA < paraMax_VOAR else paraMax_VOAR

        elif action == 15:
            VBLA = VBLA * decrement if VBLA > paraMin else paraMin

        elif action == 16:
            VREC = VREC * increment if VREC < paraMax_VOAR else paraMax_VOAR

        elif action == 17:
            VREC = VREC * decrement if VREC > paraMin else paraMin

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

        # increment_PTV = 1.01
        # decrement_PTV = 0.99
        # PTV_e_increment = 0.1
        # increment = 1.04
        # decrement = 0.96
        # e_increment = 0.3
        #
        # if action == 0:
        #     tPTV = tPTV * increment_PTV if tPTV < paraMax_tPTV else paraMax_tPTV
        #
        # elif action == 1:
        #     tPTV = tPTV * decrement_PTV if tPTV > paraMin_tPTV else paraMin_tPTV
        #
        # elif action == 2:
        #     tBLA = tBLA * increment if tBLA < paraMax_tOAR else paraMax_tOAR
        #
        # elif action == 3:
        #     tBLA = tBLA * decrement if tBLA > paraMin else paraMin
        #
        # elif action == 4:
        #     tREC = tREC * increment if tREC < paraMax_tOAR else paraMax_tOAR
        #
        # elif action == 5:
        #     tREC = tREC * decrement if tREC > paraMin else paraMin
        #
        # elif action == 6:
        #     lambdaPTV = lambdaPTV * m.exp(PTV_e_increment) if lambdaPTV < paraMax else paraMax
        #
        # elif action == 7:
        #     lambdaPTV = lambdaPTV * m.exp(-PTV_e_increment) if lambdaPTV > paraMin else paraMin
        #
        # elif action == 8:
        #     lambdaBLA = lambdaBLA * m.exp(e_increment) if lambdaBLA < paraMax else paraMax
        #
        # elif action == 9:
        #     lambdaBLA = lambdaBLA * m.exp(-e_increment) if lambdaBLA > paraMin else paraMin
        #
        # elif action == 10:
        #     lambdaREC = lambdaREC * m.exp(e_increment) if lambdaREC < paraMax else paraMax
        #
        # elif action == 11:
        #     lambdaREC = lambdaREC * m.exp(-e_increment) if lambdaREC > paraMin else paraMin
        #
        # elif action == 12:
        #     VPTV = VPTV * increment_PTV if VPTV < paraMax_VPTV else paraMax_VPTV
        #
        # elif action == 13:
        #     VPTV = VPTV * decrement_PTV if VPTV > paraMin else paraMin
        #
        # elif action == 14:
        #     VBLA = VBLA * increment if VBLA < paraMax_VOAR else paraMax_VOAR
        #
        # elif action == 15:
        #     VBLA = VBLA * decrement if VBLA > paraMin else paraMin
        #
        # elif action == 16:
        #     VREC = VREC * increment if VREC < paraMax_VOAR else paraMax_VOAR
        #
        # elif action == 17:
        #     VREC = VREC * decrement if VREC > paraMin else paraMin

    #################################################################################

    # increment_PTV = 1.01
    # increment_PTV_1 = 1.005
    # decrement_PTV = 0.99
    # decrement_PTV_1 = 0.995
    # PTV_e_increment = 0.1
    # PTV_e_increment_1 = 0.05
    # increment = 1.01
    # increment_1 = 1.005
    # decrement = 0.99
    # decrement_1 = 0.995
    # e_increment = 0.1
    # e_increment_1 = 0.05
    #
    # if action == 0:
    #     tPTV = tPTV * increment_PTV if tPTV < paraMax_tPTV else paraMax_tPTV
    #
    # elif action == 1:
    #     tPTV = tPTV * increment_PTV_1 if tPTV < paraMax_tPTV else paraMax_tPTV
    #
    # elif action == 2:
    #     tPTV = tPTV * decrement_PTV if tPTV > paraMin_tPTV else paraMin_tPTV
    #
    # elif action == 3:
    #     tPTV = tPTV * decrement_PTV_1 if tPTV > paraMin_tPTV else paraMin_tPTV
    #
    # elif action == 4:
    #     tBLA = tBLA * increment if tBLA < paraMax_tOAR else paraMax_tOAR
    #
    # elif action == 5:
    #     tBLA = tBLA * increment_1 if tBLA < paraMax_tOAR else paraMax_tOAR
    #
    # elif action == 6:
    #     tBLA = tBLA * decrement if tBLA > paraMin else paraMin
    #
    # elif action == 7:
    #     tBLA = tBLA * decrement_1 if tBLA > paraMin else paraMin
    #
    # elif action == 8:
    #     tREC = tREC * increment if tREC < paraMax_tOAR else paraMax_tOAR
    #
    # elif action == 9:
    #     tREC = tREC * increment_1 if tREC < paraMax_tOAR else paraMax_tOAR
    #
    # elif action == 10:
    #     tREC = tREC * decrement if tREC > paraMin else paraMin
    #
    # elif action == 11:
    #     tREC = tREC * decrement_1 if tREC > paraMin else paraMin
    #
    # elif action == 12:
    #     lambdaPTV = lambdaPTV * m.exp(PTV_e_increment) if lambdaPTV < paraMax else paraMax
    #
    # elif action == 13:
    #     lambdaPTV = lambdaPTV * m.exp(PTV_e_increment_1) if lambdaPTV < paraMax else paraMax
    #
    # elif action == 14:
    #     lambdaPTV = lambdaPTV * m.exp(-PTV_e_increment) if lambdaPTV > paraMin else paraMin
    #
    # elif action == 15:
    #     lambdaPTV = lambdaPTV * m.exp(-PTV_e_increment_1) if lambdaPTV > paraMin else paraMin
    #
    # elif action == 16:
    #     lambdaBLA = lambdaBLA * m.exp(e_increment) if lambdaBLA < paraMax else paraMax
    #
    # elif action == 17:
    #     lambdaBLA = lambdaBLA * m.exp(e_increment_1) if lambdaBLA < paraMax else paraMax
    #
    # elif action == 18:
    #     lambdaBLA = lambdaBLA * m.exp(-e_increment) if lambdaBLA > paraMin else paraMin
    #
    # elif action == 19:
    #     lambdaBLA = lambdaBLA * m.exp(-e_increment_1) if lambdaBLA > paraMin else paraMin
    #
    # elif action == 20:
    #     lambdaREC = lambdaREC * m.exp(e_increment) if lambdaREC < paraMax else paraMax
    #
    # elif action == 21:
    #     lambdaREC = lambdaREC * m.exp(e_increment_1) if lambdaREC < paraMax else paraMax
    #
    # elif action == 22:
    #     lambdaREC = lambdaREC * m.exp(-e_increment) if lambdaREC > paraMin else paraMin
    #
    # elif action == 23:
    #     lambdaREC = lambdaREC * m.exp(-e_increment_1) if lambdaREC > paraMin else paraMin
    #
    # elif action == 24:
    #     VPTV = VPTV * increment_PTV if VPTV < paraMax_VPTV else paraMax_VPTV
    #
    # elif action == 25:
    #     VPTV = VPTV * decrement_PTV if VPTV > paraMin else paraMin
    #
    # elif action == 26:
    #     VBLA = VBLA * increment if VBLA < paraMax_VOAR else paraMax_VOAR
    #
    # elif action == 27:
    #     VBLA = VBLA * decrement if VBLA > paraMin else paraMin
    #
    # elif action == 28:
    #     VREC = VREC * increment if VREC < paraMax_VOAR else paraMax_VOAR
    #
    # elif action == 29:
    #     VREC = VREC * decrement if VREC > paraMin else paraMin



