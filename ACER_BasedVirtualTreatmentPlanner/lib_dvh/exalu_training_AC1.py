import numpy as np
import matplotlib.pyplot as plt
from lib_dvh.AC_networks import ActorCriticNetwork
from .score_calcu import planIQ_train
import math as m
import sys
sys.path.insert(1,'/data/data/GPU2AC')
from .AC_VPTN import TreatmentEnv
from .action import action_original
from .action import action_new
import torch
from .myconfig import *
device = iscuda()
if device != "cpu":
	cuda = True

INPUT_SIZE = 300
MAX_STEP_PER_LOOP = maxstep()
MAX_STEP = MAX_STEP_PER_LOOP


# ------------- range of parmaeter -----------------
paraMax = 100000 # change in validation as well
paraMin = 0
paraMax_tPTV = 1.2
paraMin_tPTV = 1
paraMax_tOAR = 1
paraMax_VOAR = 1
paraMax_VPTV = 0.3
# ---------------------------------------------------

from lib_dvh.data_prep import loadDoseMatrix, loadMask, ProcessDmat

def exalu_training_AC1(runOpt_dvh,MPTV, MBLA, MREC,MBLA1, MREC1,episode,flagg,pdose,maxiter,lr):
	policy = ActorCriticNetwork(INPUT_SIZE, actionnum())
	test_set = ['01']

	# for sampleid in range(2): #This was because he had only two test cases
	for sampleid in range(1):
		id = test_set[sampleid]

		# env = TreatmentEnv(config,doseMatrix,targetLabels,bladderLabel,rectumLabel,pdose,maxiter)
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
			runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC,
                       lambdaPTV, lambdaBLA, lambdaREC,
                       VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
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
		planScore_fine, planScore,scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec,pdose,False)

		print("Iteration_num: {}  PlanScore: {}  PlanScore_fine: {}".format(iter, planScore, planScore_fine))

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
        #----------------two ways of NN schemes (with/without rules) --------------------------
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
		policy.load_state_dict(torch.load('./preTrained/Grid_'+str(episode)+'.pth'))
		policy.eval()
		# -------------- NN ----------------------------------
		for i in range(MAX_STEP):
			action = policy(state_test, boolean=True)

			increment_PTV = 1.01
			decrement_PTV = 0.99
			PTV_e_increment = 0.1
			increment = 1.04
			decrement = 0.96
			e_increment = 0.3

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

			# print("tPTV:{}  tBLA:{}  tREC:{}  lambdaPTV:{} lambdaBLA:{}   lambdaREC:{}  VPTV:{}  VBLA:{}  VREC:{} " ,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC,VPTV, VBLA, VREC)
			xVec = np.ones((MPTV.shape[1],))
			gamma = np.zeros((MPTV.shape[0],))
			n_state, iter, xVec = \
					runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC,VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
			if cuda:
				n_state = torch.from_numpy(n_state).float()
				n_state = n_state.to(device)
			planScore_fine, planScore,scoreall =planIQ_train(MPTV,MBLA1,MREC1,xVec,pdose,False)
			print("Step: {} Interation_num: {} PlanScore: {} PlanScore_fine:{}".format(i,iter,planScore,planScore_fine))



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

			if planScore ==9:
				print("parameters tuning is done at step: {}".format(i+1))
				break

		plt.plot(planScore_all)
		data_result_path = './data/data/Results/GPU2AC/planScoreless/'
		np.save(data_result_path + id + 'planScore' + str(episode), planScore_all)
		np.save(data_result_path + id + 'planScore_fine' + str(episode), planScore_fine_all)
		plt.show(block=False)
		# plt.pause(5)


	path='./data/data/Results/GPU2AC/planScoreless/'
	# test_set=['12','17']
	test_set = ['01']
	# x=np.arange(31) #Since MAX_STEP = 30
	x = np.arange(MAX_STEP+1)
	tem = 0
	tem_inital = 0
	# for case in range(2):
	for case in range(1):
		tem_score =  np.load(path+test_set[case]+'planScore'+str(episode)+'.npy')
		# print("tem_score",tem_score)
		# print("x",x)
		plt.plot(x,tem_score,label='$Case{case}$'.format(case=case))
		tem = max(tem_score) + tem
		tem_inital = tem_inital + tem_score[0]
	plt.legend(loc='best')
	plt.title("episode: {} initial score: {}, mean score: {}".format(episode, round(tem_inital / 2, 2), round(tem / 2, 2)))
	plt.savefig('./data/data/Results/GPU2AC/figuresless/' + str(episode) + '.png')
	plt.show()
	plt.close()



					



