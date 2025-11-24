import numpy as np
import os
import matplotlib.pyplot as plt
import re
import fnmatch

# patientid = 29
epoch = 120499

def maximum_step(patientid):
<<<<<<< HEAD
    data_path = '/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData3/dataWithPlanscoreRun/'
=======
    data_path = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3NewData4/dataWithPlanscoreRun/'
>>>>>>> c73116994303ed8bd1ac543821f1824ccc5ecea9
    pattern_base = "{}xDVHY*step*.npy"
    pattern = pattern_base.format(patientid)

    # Use fnmatch to filter filenames based on the pattern
    files = fnmatch.filter(os.listdir(data_path), pattern)

    # Print the matching filenames
    #print(files)
    captured_numbers = []

    # Iterate through the matching filenames
    for filename in files:
        # Use regular expression to extract the desired part
        match = re.search(r'^(\d+)xDVHY(\d+)step(\d+).npy$', filename)
        
        # Check if a match is found
        if match:
            captured_numbers.append(int(match.group(3)))
        else:
            print("No match found for:", filename)

    # Find the maximum value for this list
    max_value = max(captured_numbers)
    return max_value

<<<<<<< HEAD
data_result_path = '/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData3/dataWithPlanscoreRun/'
data_result_path2 = '/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData3/plansScoresSummary/'
=======
data_result_path = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3NewData4/dataWithPlanscoreRun/'
data_result_path2 = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3NewData4/plansScoresSummary/'
>>>>>>> c73116994303ed8bd1ac543821f1824ccc5ecea9
patientNum = 30

# have to fix the stepping thing
for patientid in range(patientNum):
	print('patientid=', patientid)
	name1 = data_result_path + str(patientid) + 'tpptuning' + str(epoch)
	loaded_data = np.load(name1+'.npz')
	# print('npz.shape=', loaded_data.shape)
	maxStep = maximum_step(patientid)
	newMaxStep = maxStep + 1
	print('maxStep=', maxStep)
	steps = np.arange(0, newMaxStep, 1)
	print('steps.shape=',steps.shape)

	tPTV_all_loaded = loaded_data['l1']
	tPTV_all_loaded = tPTV_all_loaded[:newMaxStep]
	plt.plot(steps, tPTV_all_loaded, label = 'tPTV')

	tBLA_all_loaded = loaded_data['l2']
	tBLA_all_loaded = tBLA_all_loaded[:newMaxStep]
	plt.plot(steps, tBLA_all_loaded, label = 'tBLA')

	tREC_all_loaded = loaded_data['l3']
	tREC_all_loaded = tREC_all_loaded[:newMaxStep]
	plt.plot(steps, tREC_all_loaded, label = 'tREC')

	lambdaPTV_all_loaded = loaded_data['l4']
	lambdaPTV_all_loaded = lambdaPTV_all_loaded[:newMaxStep]
	plt.plot(steps, lambdaPTV_all_loaded, label = 'lambdaPTV')

	lambdaBLA_all_loaded = loaded_data['l5']
	lambdaBLA_all_loaded = lambdaBLA_all_loaded[:newMaxStep]
	plt.plot(steps, lambdaBLA_all_loaded, label = 'lambdaBLA')

	lambdaREC_all_loaded = loaded_data['l6']
	lambdaREC_all_loaded = lambdaREC_all_loaded[:newMaxStep]
	plt.plot(steps, lambdaREC_all_loaded, label = 'lambdaREC')

	VPTV_all_loaded = loaded_data['l7']
	VPTV_all_loaded = VPTV_all_loaded[:newMaxStep]
	plt.plot(steps, VPTV_all_loaded, label = 'VPTV')

	VBLA_all_loaded = loaded_data['l8']
	VBLA_all_loaded = VBLA_all_loaded[:newMaxStep]
	plt.plot(steps, VBLA_all_loaded, label = 'VBLA')

	VREC_all_loaded = loaded_data['l9']
	VREC_all_loaded = VREC_all_loaded[:newMaxStep]
	plt.plot(steps, VREC_all_loaded, label = 'VREC')

	plt.legend()
	plt.xlabel('TPP Tuning Steps')
	plt.ylabel('Weights')
	# Repeat for other arrays as needed
	plt.savefig(data_result_path2+str(patientid)+'tpptuning'+str(epoch)+'.png', dpi =1200)
	plt.close()
