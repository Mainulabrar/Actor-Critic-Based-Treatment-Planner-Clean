import numpy as np
import re
import os

test_set = ['001', '008', '009', '010', '011', '014', '015', '017', '018', '020', '022', '023', '025', '027', '030', '031', '036', '037', '039', '042', '045', '046', '054', '061', '065', '068', '070', '073', '074', '077', '080', '081', '084', '091', '093', '095', '097', '098']
UnperturbedPath = "/data2/mainul/DQNFGSMUTSWallPaper1/UnperturbedDVH12/"
unperturbedArray = np.zeros((len(test_set), 27))
ResultSavePath = "/data2/mainul/DataAndGraphDQN/FGSMPaper/"
os.makedirs(ResultSavePath, exist_ok = True)

for i in range(len(test_set)):
	pattern = rf"{test_set[i]}unperturbedValueArrayFlattened0.npy$"
	for filename in os.listdir(UnperturbedPath):
		if re.match(pattern, filename):
			full_file_path = os.path.join(UnperturbedPath, filename)
			unperturbedArray[i,:] = np.load(full_file_path)

print(unperturbedArray)

np.save(ResultSavePath+'unperturbedValueArray1', unperturbedArray)


PerturbedPath = "/data2/mainul/DQNFGSMUTSWallPaper1/PerturbedDVH12/"

perturbedArray = np.zeros((len(test_set), 27))
for i in range(len(test_set)):
	pattern = rf"{test_set[i]}perturbedValueArrayFlattened1.npy$"
	for filename in os.listdir(PerturbedPath):
		if re.match(pattern, filename):
			full_file_path = os.path.join(PerturbedPath, filename)
			perturbedArray[i,:] = np.load(full_file_path)

print(perturbedArray)

np.save(ResultSavePath+'perturbedValueArray1', perturbedArray)
