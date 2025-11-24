import matplotlib.pyplot as plt
import numpy as np
import os
import time
import re
import fnmatch


# planscoreSummaryPath = '/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/plansScoresSummary/'
# planscoreSavePath = '/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/planscores/'

# planscoreSummaryPath = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3NewData1/plansScoresSummary/'
# planscoreSavePath = '/data/mainul1/results_CORS/scratch6_30StepsNewParamenters3NewData1/planscores/'
# planscoreSavePath = "/data2/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData1/planscores/"
planscoreSavePath = "/data2/mainul1/results_CORS1/scratch6_30StepsNewParamenters3/planscores/"
# planscoreSavePath = "/data2/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData3/planscores/"

data_result_path = "/data2/mainul/DataAndGraph/"



def maximum_step(patientid):
    data_path = planscoreSavePath
    pattern_base = "{}planscoreBeforeWhileBreaking*.npy"
    pattern = pattern_base.format(patientid)

    # Use fnmatch to filter filenames based on the pattern
    files = fnmatch.filter(os.listdir(data_path), pattern)

    # Print the matching filenames
    #print(files)
    captured_numbers = []

    # Iterate through the matching filenames
    for filename in files:
        # Use regular expression to extract the desired part
        match = re.search(r'^(\d+)planscoreBeforeWhileBreaking(\d+).npy$', filename)
        
        # Check if a match is found
        if match:
            captured_numbers.append(int(match.group(2)))
        else:
            print("No match found for:", filename)

    # Find the maximum value for this list
    max_value = max(captured_numbers)

    return max_value

# Sorting data to get them in one variable which is called scoreData, also creating scoreDataMax and scoreDataInitial to store the initial and maximum planscores
patient_num = 30
scoreData = []
scoreDataInitial = []
scoreDataMax = []
for patientid in range(patient_num):
	max_step = maximum_step(patientid)
	scoreDataRow = np.load(planscoreSavePath+str(patientid)+'planscoreInsideIf.npy')
	scoreDataRow = np.array([scoreDataRow])
	scoreDataInitial.append(scoreDataRow)
	# print(scoreDataRow)
	for j in range(1, max_step+1, 1):
		Y = np.load(planscoreSavePath+str(patientid)+'planscoreBeforeWhileBreaking'+str(j)+'.npy')
		Y = np.array([Y])
		# print(Y)
		scoreDataRow = np.concatenate([scoreDataRow,Y])
	maxScore = np.array([max(scoreDataRow)])	
	scoreDataMax.append(maxScore)
	scoreData.append(scoreDataRow)

scoreDataInitial = list(map(list, zip(*scoreDataInitial)))
scoreDataMax = list(map(list, zip(*scoreDataMax)))
# print(scoreData[10][5])
print('scoreDataInitial', scoreDataInitial)
print('scoreDataMax', scoreDataMax)
# I should try to save the scoreData with pickle, it can not be saved with np.save as np.save is for homogenous data only
# np.save(planscoreSummaryPath+'AllPatientsScoreData',scoreData)
# np.save(planscoreSummaryPath+'scoreDataInitial', scoreDataInitial)
# np.save(planscoreSummaryPath+ 'scoreDataMax', scoreDataMax)

# np.save(data_result_path + 'scoreDataInitialUTSW63', scoreDataInitial)
# np.save(data_result_path + 'scoreDataMaxUTSW63', scoreDataMax)

# np.save(data_result_path + 'scoreDataInitialUTSW99', scoreDataInitial)
# np.save(data_result_path + 'scoreDataMaxUTSW99', scoreDataMax)

np.save(data_result_path + 'scoreDataInitialCORT', scoreDataInitial)
np.save(data_result_path + 'scoreDataMaxCORT', scoreDataMax)




# data1 = np.random.uniform(0, 9, 1000)
# data2 = np.random.uniform(0, 9, 1000)  # Example: 1000 random data points between 0 and 1

# # Plot the histogram
values1, bins = np.histogram(scoreDataInitial, bins=np.linspace(0.5, 9.5, 10))
values2, bins = np.histogram(scoreDataMax, bins=np.linspace(0.5, 9.5, 10))

score = np.arange(1,10,1)
# Set the width of the bars
bar_width = 0.35

# Set the positions for the bars
bar_positions_set1 = score - bar_width/2
bar_positions_set2 = score + bar_width/2

# Plot the first set of bars
plt.bar(bar_positions_set1, values1, width=bar_width, label='InitialScores', color='blue')

# Plot the second set of bars
plt.bar(bar_positions_set2, values2, width=bar_width, label='MaxScores', color='orange')

# Add labels and title
plt.xlabel('Plan Scores')
plt.ylabel('Number of Patients')
plt.title('Number of Patients vs Plan Scores')
plt.legend()
# plt.savefig(planscoreSummaryPath+'InitialAndFinalPlanComparison.png', dpi = 1200)

# Customize y-axis ticks to show percentages
#plt.gca().set_yticklabels(['{:.0%}'.format(x) for x in plt.gca().get_yticks()])

# Show the plot
plt.show()