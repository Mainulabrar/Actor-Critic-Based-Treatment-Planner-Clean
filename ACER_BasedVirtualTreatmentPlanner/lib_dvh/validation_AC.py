# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:55:19 2019

aim: validate & test DQN

@author: writed by Chenyang Shen, modified by Chao Wang
"""
import torch
import numpy as np
import logging
from lib_dvh.AC_networks import ActorCriticNetwork
import matplotlib.pyplot as plt
from .score_calcu import planIQ_train
import sys

sys.path.insert(1, '/data/data/GPU3AC')
from .AC_VPTN import TreatmentEnv
from .action import action_original
from .action import action_new
import math as m
import os
from lib_dvh.data_prep import loadDoseMatrix, loadMask, ProcessDmat
from .myconfig import *

device = iscuda()
if device != "cpu":
    cuda = True

# ------------- range of parmaeter -----------------
paraMax = 100000  # change in validation as well
paraMin = 0
paraMax_tPTV = 1.2
paraMin_tPTV = 1
paraMax_tOAR = 1
paraMax_VOAR = 1
paraMax_VPTV = 0.3
# ---------------------------------------------------


# default values
# INPUT_SIZE = 100
# MAX_STEP = 30

INPUT_SIZE = 300
MAX_STEP_PER_LOOP = maxstep()
MAX_STEP = MAX_STEP_PER_LOOP
# episode = MAX_STEP - 1


logging.basicConfig(filename='result_training_withoutrule.log', level=logging.INFO,
                    format='%(message)s')


def bot_play(runOpt_dvh, MPTV, MBLA, MREC, MBLA1, MREC1, episode, flagg, pdose, maxiter, lr) -> None:
    # test_set=['12','17']#['01','07','08','09','10','11','12','13','14','15','16','17']
    test_set = ['01']
    logging.info(
        '------------------------------------------ validation ----------------------------------------------------')
    # for sampleid in range(test_num):
    # config=get_config()
    # pid=('12','17')
    pid = ['01']
    i = 0
    # data_path = '/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/0'

    data_path = './lib_dvh/f_dijs/0'
    # /data/data/dose_deposition3/plostate_dijs/f_masks/0'
    globals()['doseMatrix_' + str(i)] = loadDoseMatrix(data_path + str(pid[i]) + '.hdf5')

    policy = ActorCriticNetwork(INPUT_SIZE, actionnum())

    for sampleid in range(1):
        id = test_set[sampleid]
        doseMatrix = globals()['doseMatrix_' + str(sampleid)]
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

        # data_result_path='./data/data/Results/GPU2AC/general/3/'
        # np.save(data_result_path+id+'xDVHYInitial',
        #         Y)
        # np.save(data_result_path+id+'xDVHXInitial',
        #         X)
        # np.save(data_result_path+id+'xVecInitial', xVec)
        # np.save(data_result_path+id+'DoseInitial', Dose)
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
        planScore_fine, planScore, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, False)
        logging.info('---------------------- initialization ------------------------------')
        logging.info("Iteration_num: {}  PlanScore: {}  PlanScore_fine: {}".format(iter, planScore, planScore_fine))

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
        # agent=Agent(env,lr,INPUT_SIZE,episode)
        # agent.load_models(episode)
        policy.load_state_dict(torch.load('./preTrained/Grid_' + str(episode) + '.pth'))
        policy.eval()
        # -------------- NN ----------------------------------
        for i in range(MAX_STEP):
            action = policy(state_test, boolean=False)

            n_state_test, reward, Score_fine, Score, done, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec = env.step(
                action, t, Score_fine, Score, MPTV, MBLA, MREC, MBLA1, MREC1, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA,
                lambdaREC, VPTV, VBLA, VREC, pdose, maxiter)


            print("Step:{} Iteration_num:{} PlanScore:{} PlanScore_fine:{}".format(i, iter, planScore, planScore_fine))
            # state_test = tf.convert_to_tensor([n_state_test],dtype=tf.float32)
            state_test = n_state_test
            if cuda:
                state_test = torch.from_numpy(state_test).float()
                state_test = state_test.to(device)

            # collect the result in each iteration
            tPTV_all[i + 1] = tPTV
            tBLA_all[i + 1] = tBLA
            tREC_all[i + 1] = tREC
            lambdaPTV_all[i + 1] = lambdaPTV
            lambdaBLA_all[i + 1] = lambdaBLA
            lambdaREC_all[i + 1] = lambdaREC
            VPTV_all[i + 1] = VPTV
            VBLA_all[i + 1] = VBLA
            VREC_all[i + 1] = VREC
            planScore_all[i + 1] = planScore
            planScore_fine_all[i + 1] = planScore_fine

            # if paraidx == 0:
            #     logging.info("Step: {}  Iteration: {}  Action: {}  tPTV: {} case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
            #                                                                                     round(tPTV,2), case, round(planScore_fine,3), round(planScore,3)))
            # if paraidx == 1:
            #     logging.info("Step: {}  Iteration: {}  Action: {}  tBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
            #                                                                                    round(tBLA,2), case, round(planScore_fine,3), round(planScore,3)))
            # if paraidx == 2:
            #     logging.info("Step: {}  Iteration: {}  Action: {}  tREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
            #                                                                                     round(tREC,2), case, round(planScore_fine,3), round(planScore,3)))
            # if paraidx == 3:
            #     logging.info(
            #         "Step: {}  Iteration: {}  Action: {}  lambdaPTV: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter,
            #                  action, round(lambdaPTV,2),
            #                                                                                                                case,
            #                                                                                                                round(planScore_fine,3),
            #                                                                                                       round(planScore,3)))

            # if paraidx == 4:
            #     logging.info("Step: {}  Iteration: {}  Action: {}  lambdaBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1,
            #                  iter, action,
            #                                                                                     round(lambdaBLA,2), case, round(planScore_fine,3),
            #                                                                                     round(planScore,3)))
            # if paraidx == 5:
            #     logging.info("Step: {}  Iteration: {}  Action: {}  lambdaREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter,
            #                  action,
            #                                                                                     round(lambdaREC,2), case, round(planScore_fine,3),
            #                                                                                     round(planScore,3)))
            # if paraidx == 6:
            #     logging.info("Step: {}  Iteration: {}  Action: {}  VPTV: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
            #                                                                                     round(VPTV,4), case, round(planScore_fine,3), round(planScore,3)))
            # if paraidx == 7:
            #     logging.info("Step: {}  Iteration: {}  Action: {}  VBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
            #                                                                                    round(VBLA,4), case, round(planScore_fine,3), round(planScore,3)))
            # if paraidx == 8:
            #     logging.info("Step: {}  Iteration: {}  Action: {}  VREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
            #                                                                                     round(VREC,4), case, round(planScore_fine,3), round(planScore,3)))

            Dose = doseMatrix.dot(xVec)
            # np.save(data_result_path+id+'xVec' + str(episode)  + 'step' + str(i + 1), xVec)
            # np.save(data_result_path+id+'xDose' + str(episode)  + 'step' + str(i + 1), Dose)

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

            # np.save(data_result_path+id+'xDVHY' + str(episode)  + 'step' + str(i + 1),
            #         Y)
            # np.save(data_result_path+id+'xDVHX' + str(episode) + 'step' + str(i + 1),
            #         X)
            data_result_path2 = './data/data/Results/figuresPATp/'
            plt.plot(x_ptv, y_ptv)
            plt.plot(x_bladder, y_bladder)
            plt.plot(x_rectum, y_rectum)
            plt.legend(('ptv', 'bladder', 'rectum'))
            plt.title('DVH' + str(episode) + 'step' + str(i + 1))
            plt.savefig(data_result_path2 + id + 'DVH' + str(episode) + 'step' + str(i + 1) + '.png')
            plt.show(block=False)

            plt.close()

            if planScore == 9:
                print("parameter tuning is done at step: {} ".format(i + 1))
                break

        plt.plot(tPTV_all)
        plt.plot(tBLA_all)
        plt.plot(tREC_all)
        plt.plot(lambdaPTV_all)
        plt.plot(lambdaBLA_all)
        plt.plot(lambdaREC_all)
        plt.plot(VPTV_all)
        plt.plot(VBLA_all)
        plt.plot(VREC_all)

        plt.legend(('tPTV', 'tBLA', 'tREC', 'lambdaPTV', 'lambdaBLA', 'lambdaREC', 'VPTV', 'VBLA', 'VREC'))
        plt.title('TPP tuning steps')
        plt.savefig('./data/data/Results/figuresPATp/tpptuning' + str(episode) + '.png')
        plt.show(block=False)

        plt.close()

        # np.save(
        #     data_result_path+id+'tPTV' + str(episode),
        #     tPTV_all)
        # np.save(
        #     data_result_path +id+'tBLA' + str(episode),
        #     tBLA_all)
        # np.save(
        #     data_result_path+id+'tREC' + str(episode),
        #     tREC_all)
        # np.save(
        #     data_result_path+id+'lambdaPTV' + str(episode),
        #     lambdaPTV_all)
        # np.save(
        #     data_result_path+id+'lambdaBLA' + str(episode),
        #     lambdaBLA_all)
        # np.save(
        #     data_result_path+id+'lambdaREC' + str(episode),
        #     lambdaREC_all)
        # np.save(
        #     data_result_path+id+'VPTV' + str(episode),
        #     VPTV_all)
        # np.save(
        #     data_result_path +id+'VBLA' + str(episode),
        #     VBLA_all)
        # np.save(
        #     data_result_path+id+'VREC' + str(episode),
        #     VREC_all)
        # np.save(
        #     data_result_path+id+'planScore' + str(episode),
        #     planScore_all)
        # np.save(
        #     data_result_path+id+'planScore_fine' + str(episode),
        #     planScore_fine_all)
