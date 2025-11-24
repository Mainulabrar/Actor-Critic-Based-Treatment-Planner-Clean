import numpy as np
import matplotlib.pyplot as plt
import re
import os

ScoreArrayInitial = []
ScoreArrayFinal = []
# planscoresSavePath = '/data2/mainul/results_CORS1/scratch6_30StepsNewParamenters3NewData1/planscores/'
# planscoresSavePath = '/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/planscores/'
# planscoresSavePath = '/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/planscores/'
# planscoresSavePath = '/data2/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/planscores3rd/'
# planscoresSavePath = '/data/mainul1/results_CORS/scratch6_20StepsNewParamenters3NewData4/planscores/'
planscoresSavePath = '/data2/mainul/results_CORS1random3C/scratch6_30StepsNewParamenters3NewData1/planscores/'
# planscoresSavePath = '/data2/mainul/results_CORS1random6C092/scratch6_30StepsNewParamenters3NewData1/planscores/'
# patient_list = ['001', '008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '092', '093', '095', '097', '098']


# patient_list = ['008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '092', '093', '095', '097', '098']
# 098planscoreInsideIf.npy
patient_list = np.arange(5)
# This block is for the initial array=================================
# for i in range(len(patient_list)):
# 	pattern = rf"^{patient_list[i]}planscoreInsideIf\.npy$"
# 	for filename in os.listdir(planscoresSavePath):

# 		if re.match(pattern, filename):
# 			full_file_path = os.path.join(planscoresSavePath, filename)
# 			print(full_file_path)
# 			InitScore = np.load(full_file_path)
# 			print(InitScore)
# 			ScoreArrayInitial.append(InitScore)

# print(np.array(ScoreArrayInitial))
# # np.save('/data2/mainul/DataAndGraph/AllInitialScoreACERrandom6C', ScoreArrayInitial)
# # np.save('/data2/mainul/DataAndGraph/AllInitialScoreACERTROTS2nd', ScoreArrayInitial)
# # np.save('/data2/mainul/DataAndGraph/AllInitialScoreACERrandom3C1', ScoreArrayInitial)
# np.save('/data2/mainul/DataAndGraph/AllInitialScoreACERrandom3C092', ScoreArrayInitial)
# ====================================================================

for i in range(len(patient_list)):
	pattern = rf"^{patient_list[i]}.*\.npy$"
	IndScores = []
	for filename in os.listdir(planscoresSavePath):

		if re.match(pattern, filename):
			full_file_path = os.path.join(planscoresSavePath, filename)
			# print(full_file_path)
			Score = np.load(full_file_path)
			# print(Score)
			IndScores.append(Score)
	IndScores = np.array(IndScores)
	BestScore = np.max(IndScores)
	# print(BestScore)
	ScoreArrayFinal.append(BestScore)

ScoreArrayFinal = np.array(ScoreArrayFinal)

print(ScoreArrayFinal)
# np.save('/data2/mainul/DataAndGraph/AllFinalScoreACERrandom3C092', ScoreArrayFinal)
# np.save('/data2/mainul/DataAndGraph/AllFinalScoreACERrandom6C', ScoreArrayFinal)
# np.save('/data2/mainul/DataAndGraph/AllFinalScoreACERTROTS2nd', ScoreArrayFinal)
# np.save('/data2/mainul/DataAndGraph/AllFinalScoreACERrandom3C1', ScoreArrayFinal)
np.save('/data2/mainul/DataAndGraph/AllFinalScoreACERrandom3C', ScoreArrayFinal)