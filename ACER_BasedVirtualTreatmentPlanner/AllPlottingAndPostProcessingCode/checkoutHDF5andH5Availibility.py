import numpy as np
import re
import os

hdf5 = []
h5 = []
for file in os.listdir('/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/'):
	print(file)
	pattern = r"^(.*)\.hdf5$"
	match = re.match(pattern, file)
	if match:
		hdf5.append(match.group(1))

for file in os.listdir('/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/'):
	pattern = r"^(.*)\.h5$"
	match = re.match(pattern, file)
	if match:
		h5.append(match.group(1))

print('hdf5', hdf5)
print('h5', h5)
Common = list(set(hdf5)& set(h5))

print('Common', np.sort(Common))
