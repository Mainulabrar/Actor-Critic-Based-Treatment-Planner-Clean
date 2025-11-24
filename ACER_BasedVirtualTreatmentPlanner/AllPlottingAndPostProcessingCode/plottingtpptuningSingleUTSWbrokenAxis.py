import numpy as np
import os
import matplotlib.pyplot as plt
import re
import fnmatch

# patientid = 29
epoch = 120499
data_result_path = '/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_result_path = '/data4/mainul/MultiModalAI6Beam/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path = '/data4/mainul/MultiModalAI1/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'

graphSavePath ='/data4/Proposal/'

def maximum_step(patientid):

    data_path = data_result_path

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

# <<<<<<< HEAD
# data_result_path = '/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData3/dataWithPlanscoreRun/'
# data_result_path2 = '/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData3/plansScoresSummary/'
# =======
# data_result_path = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3NewData4/dataWithPlanscoreRun/'
# data_result_path2 = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3NewData4/plansScoresSummary/'
# >>>>>>> c73116994303ed8bd1ac543821f1824ccc5ecea9
patientNum = ['008']

plt.rcParams["lines.linewidth"] = 3.5
plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["xtick.major.width"] = 2.5  # Major x-tick thickness
plt.rcParams["ytick.major.width"] = 2.5

plt.figure(figsize=(18, 9))
# have to fix the stepping thing
for patientid in patientNum:
	print('patientid=', patientid)
	name1 = data_result_path + str(patientid) + 'tpptuning' + str(epoch)
	loaded_data = np.load(name1+'.npz')

	for j in range(9):
		print(loaded_data[f'l{j+1}'])
	# print('npz.shape=', loaded_data.shape)
	maxStep = len(np.nonzero(loaded_data['l9'])[0])
	newMaxStep = maxStep
	print('maxStep=', maxStep)
	steps = np.arange(0, newMaxStep, 1)
	print('steps.shape=',steps.shape)


	plt.subplot(211)	
	lambdaPTV_all_loaded = loaded_data['l4']
	lambdaPTV_all_loaded = lambdaPTV_all_loaded[:newMaxStep]
	plt.plot(steps, lambdaPTV_all_loaded, '-b', label = r'$\mathrm{\lambda_{PTV}}$')

	# plt.ylim(15, 40)

	print('lambdaPTV_all_loaded', lambdaPTV_all_loaded)
	# remove bottom spine + all xticks from top plot
	plt.gca().spines['bottom'].set_visible(False)
	plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

	# --- Bottom subplot ---
	plt.subplot(212, sharex=plt.gcf().axes[0])

	tPTV_all_loaded = loaded_data['l1']
	tPTV_all_loaded = tPTV_all_loaded[:newMaxStep]
	plt.plot(steps, tPTV_all_loaded, '-b', label = r'$\mathrm{t_{PTV}}$')

	tBLA_all_loaded = loaded_data['l2']
	tBLA_all_loaded = tBLA_all_loaded[:newMaxStep]
	plt.plot(steps, tBLA_all_loaded, '-b', label = r'$\mathrm{t_{BLA}}$')

	tREC_all_loaded = loaded_data['l3']
	tREC_all_loaded = tREC_all_loaded[:newMaxStep]
	plt.plot(steps, tREC_all_loaded, '-b', label = r'$\mathrm{t_{REC}}$')

	lambdaBLA_all_loaded = loaded_data['l5']
	lambdaBLA_all_loaded = lambdaBLA_all_loaded[:newMaxStep]
	plt.plot(steps, lambdaBLA_all_loaded, '-b', label = r'$\mathrm{\lambda_{BLA}}$')

	lambdaREC_all_loaded = loaded_data['l6']
	lambdaREC_all_loaded = lambdaREC_all_loaded[:newMaxStep]
	plt.plot(steps, lambdaREC_all_loaded, '-b', label = r'$\mathrm{\lambda_{REC}}$')

	VPTV_all_loaded = loaded_data['l7']
	VPTV_all_loaded = VPTV_all_loaded[:newMaxStep]
	plt.plot(steps, VPTV_all_loaded, '-b', label = r'$\mathrm{V_{PTV}}$')

	VBLA_all_loaded = loaded_data['l8']
	VBLA_all_loaded = VBLA_all_loaded[:newMaxStep]
	plt.plot(steps, VBLA_all_loaded, '-b', label = r'$\mathrm{V_{BLA}}$')

	VREC_all_loaded = loaded_data['l9']
	VREC_all_loaded = VREC_all_loaded[:newMaxStep]
	plt.plot(steps, VREC_all_loaded, '-b', label = r'$\mathrm{V_{REC}}$')

	plt.ylim(0, 2)
	# plt.xlim(0,16)
	plt.xlim(0, len(np.nonzero(loaded_data['l9'])[0])+1)
	plt.xticks(fontsize = 40)
	plt.yticks(fontsize = 40)
	plt.legend(loc = 'upper left', fontsize = 40, ncol = 3)
	plt.xlabel('TPP Tuning Steps', fontsize = 40)
	plt.ylabel('Weights', fontsize = 40)
	# remove top spine from bottom plot (so it blends)
	plt.gca().spines['top'].set_visible(False)
	plt.tight_layout()

	# Repeat for other arrays as needed
	# plt.savefig(graphSavePath+str(patientid)+'tpptuning'+str(epoch)+'.png', dpi =600)
	plt.show()
	plt.close()
