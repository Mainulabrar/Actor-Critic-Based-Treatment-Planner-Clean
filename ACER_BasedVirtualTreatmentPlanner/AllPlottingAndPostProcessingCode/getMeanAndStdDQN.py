import numpy as np
import os
import re

# patient_list = ['001', '008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '092', '093', '095', '097', '098']
patient_list = ['001', '008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '093', '095', '097', '098']

data_path = '/data2/mainul/DQNYinResults180Paper/'

Max_scores =[]
for i in range(len(patient_list)):  
	pattern = rf'^{patient_list[i]}planScore\d+\.npy$'
	array = []
	for file in os.listdir('/data2/mainul/DQNYinResults65Paper/'):
		if re.search(pattern, file):
			full_file_path = data_path+file
			print(full_file_path)
			array.append(np.load(full_file_path))
	Max_scores.append(max(array))
	
print(Max_scores)
print(np.mean(Max_scores))
