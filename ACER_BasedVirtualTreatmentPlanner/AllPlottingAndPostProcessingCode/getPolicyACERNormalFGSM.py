import numpy as np
import re
import os

test_set = ['001', '008', '009', '010', '011', '014', '015', '017', '018', '020', '022', '023', '025', '027', '030', '031', '036', '037', '039', '042', '045', '046', '054', '061', '065', '068', '070', '073', '074', '077', '080', '081', '084', '091', '093', '095', '097', '098']
UnperturbedPath = "/data2/mainul1/results_CORS/FGSM_Attack0.1crossEntropyUTSW/scratch6_30StepsNewParamenters3/dataWithPlanscoreRun/"
unperturbedArray = np.zeros((len(test_set), 18))
ResultSavePath = "/data2/mainul/DataAndGraph/"
os.makedirs(ResultSavePath, exist_ok = True)

for i in range(len(test_set)):
	pattern = rf"^0Policy{i}step0.npy$"
	for filename in os.listdir(UnperturbedPath):
		if re.match(pattern, filename):
			full_file_path = os.path.join(UnperturbedPath, filename)
			unperturbedArray[i,:] = np.load(full_file_path)

print(unperturbedArray)

np.save(ResultSavePath+'unperturbedACERPaper1', unperturbedArray)


PerturbedPath = "/data2/mainul1/results_CORS/FGSM_Attack0.1crossEntropyUTSW/scratch6_30StepsNewParamenters3/dataWithPlanscoreRun/"

perturbedArray = np.zeros((len(test_set), 18))
for i in range(len(test_set)):
	pattern = rf"^1Policy{i}step0.npy$"
	for filename in os.listdir(PerturbedPath):
		if re.match(pattern, filename):
			full_file_path = os.path.join(PerturbedPath, filename)
			perturbedArray[i,:] = np.load(full_file_path)

print(perturbedArray)

np.save(ResultSavePath+'perturbedValueACERPaper1', perturbedArray)
