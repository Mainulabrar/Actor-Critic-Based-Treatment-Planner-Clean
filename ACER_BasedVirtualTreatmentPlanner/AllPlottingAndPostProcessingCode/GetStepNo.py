import numpy as np

InitScore = []
MaxScore = []
StepNumber = []
# data_result_path = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3diffbeam7/dataWithPlanscoreRun/'
# data_result_path = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3diffbeam/dataWithPlanscoreRun/'
data_result_path = '/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_result_path = "/data2/mainul/results_CORS1random6C92Included/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/"
# data_result_path = '/data4/mainul/MultiModalAI1/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path = '/data4/mainul/MultiModalAI1/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
# data_result_path = '/data4/mainul/MultiModalAI6Beam/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
# data_result_path = '/data4/mainul/MultiModalAIwithoutTPP/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
# patient_list = [ '008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '093', '095', '097', '098']
patient_list = [ '008', '009', '010', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '092', '093', '095', '097', '098']
# patient_list = [ '008',  '010',     '020',    '027',   '031',   '039', '042',  '046',  '061',  '070',    '084',  '087',  '092',  '095',  '098']
# patient_listShort = [ '008', '009', '010', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039',  '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '092', '093', '095', '097', '098']
# patient_list = patient_listShort

# patient_list = np.arange(147)
# print('patient_list',len(patient_list))

epoch = 120499
for i in patient_list:
	NpzFile = np.load(data_result_path+f'{i}tpptuning{epoch}.npz')
	if NpzFile['l10'][0] == 9.0:
		continue
	# print(NpzFile['l10'][0])
	# print(np.max(NpzFile['l10']))
	InitScore.append(NpzFile['l10'][0])
	MaxScore.append(np.max(NpzFile['l10']))
	StepNumber.append(np.size(np.nonzero(NpzFile['l10'])[0])-1)

# np.savetxt(data_result_path+'AllInitScore', InitScore)
# np.savetxt(data_result_path+'AllMaxScore', MaxScore)
print('patient_list', len(InitScore))
print('Init Score', InitScore)
print('MaxScore', MaxScore)
print('Mean Init', np.mean(InitScore))
print('Std Init Score', np.std(InitScore))
print('Mean Max', np.mean(MaxScore))
print('Std Max Score', np.std(MaxScore))
# print(MaxScore)
print('Mean Step', np.mean(StepNumber))
print('Std Step source', np.std(StepNumber))
print('Steps', StepNumber)

# Array = [22, 22, 24, 25, 30, 21, 29, 21, 17, 13, 30, 30, 19, 30, 30, 20, 24, 17, 19, 21, 23, 23, 16, 30, 12, 17, 12, 19, 20, 22]
# print('Length', len(Array))
# print('newMean', np.mean(Array))
# print('newStd', np.std(Array))


