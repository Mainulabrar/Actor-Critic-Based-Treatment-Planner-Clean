import numpy as np

InitScore = []
MaxScore = []
StepNumber = []
# data_result_path = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3diffbeam7/dataWithPlanscoreRun/'
# data_result_path = '/data2/mainul/results_CORS1PaperDiffBeam7/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_result_path = '/data2/mainul/results_CORS1PaperDiffBeam/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_result_path = '/data4/mainul/MultiModalAI6Beam/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
# data_result_path = '/data4/mainul/MultiModalAI2/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path = '/data4/mainul/MultiModalAIRealPro/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'

patient_list = [ '008',  '010',     '020',    '027',   '031',   '039', '042',  '046',  '061',  '070',    '084',  '087',  '092',  '095',  '098']
patient_list = [ '008',  '010',     '020',    '027',   '031',   '039', '042',  '046',  '061',  '070']



for i in patient_list:
	NpzFile = np.load(data_result_path+f'{i}tpptuning120499.npz')
	print(NpzFile['l10'][0])
	# print(NpzFile['l10'])
	print(np.max(NpzFile['l10']))
	InitScore.append(NpzFile['l10'][0])
	MaxScore.append(np.max(NpzFile['l10']))
	StepNumber.append(np.size(np.nonzero(NpzFile['l10'])[0])-1)
	# print(np.nonzero(NpzFile['l10']))

np.savetxt(data_result_path+'AllInitScore', InitScore)
np.savetxt(data_result_path+'AllMaxScore', MaxScore)

print('InitScore', InitScore)
print('MaxScore', MaxScore)
print('StepNumber', StepNumber)
print('Mean Init', np.mean(InitScore))
# print('Std Init source', ((np.std(InitScore))**2)*(len(InitScore)))
print('Std Init', np.std(InitScore))
print('Mean Max', np.mean(MaxScore))
# print('Std Max Source', ((np.std(MaxScore))**2)*(len(MaxScore)))
print('Std Max ', np.std(MaxScore))
# print(MaxScore)
print('Mean Step', np.mean(StepNumber))
# print('Std Step source', ((np.std(StepNumber))**2)*(len(StepNumber)))
print('Std Step source', np.std(StepNumber))
