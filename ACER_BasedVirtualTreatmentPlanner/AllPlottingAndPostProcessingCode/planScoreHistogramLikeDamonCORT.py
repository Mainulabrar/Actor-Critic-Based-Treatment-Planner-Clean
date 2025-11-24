import matplotlib.pyplot as plt
import numpy as np

# scoreDataInitial = np.load('/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/AllInitialScoreACERPaper.npy')
# scoreDataMax = np.load('/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/AllFinalScoreACERPaper.npy')

# scoreDataInitial = np.load('/data2/mainul/DataAndGraph/AllInitialScoreACERrandom6CEdited.npy')
# scoreDataMax = np.load('/data2/mainul/DataAndGraph/AllFinalScoreACERrandom6CEdited.npy')

# scoreDataInitial = np.load('/data2/mainul/DataAndGraph/AllInitialScoreACERrandom6CEdited.npy')
# scoreDataMax = np.load('/data2/mainul/DataAndGraph/AllFinalScoreACERrandom6CEdited.npy')

# scoreDataInitial = np.load('/data2/mainul/DataAndGraph/AllInitialScoreACERrandom6CEdited.npy')
# scoreDataMax = np.load('/data2/mainul/DataAndGraph/AllFinalScoreACERrandom6CEdited.npy')

# scoreDataInitial = np.load('/data2/mainul/DataAndGraph/AllInitialScoreACERrandom6CEdited.npy')
# scoreDataMax = np.load('/data2/mainul/DataAndGraph/AllFinalScoreACERrandom6CEdited.npy')

scoreDataInitial = np.load("/data2/mainul/DataAndGraph/scoreDataInitialCORT.npy")
scoreDataMax = np.load("/data2/mainul/DataAndGraph/scoreDataMaxCORT.npy")


# print(scoreDataMax)
graphSavePath = '/data2/mainul/DataAndGraph/'


# This block is for TROTS one time preprocessing==============================================
# scoreDataInitial1 = np.array([8]*30)
# scoreDataMax1 = np.array([9]*27+[8]*3)
# print('1', scoreDataInitial1)
# print('2', scoreDataMax1)


# scoreDataInitial2 = np.load("/data2/mainul/DataAndGraph/AllInitialScoreACERTROTS2nd30.npy")
# scoreDataMax2 = np.load("/data2/mainul/DataAndGraph/AllFinalScoreACERTROTS2nd30.npy")

# scoreDataInitial3 = np.load("/data2/mainul/DataAndGraph/AllInitialScoreACERTROTS3rd30.npy")
# scoreDataMax3 = np.load("/data2/mainul/DataAndGraph/AllFinalScoreACERTROTS3rd30.npy")

# scoreDataInitial = np.concatenate((scoreDataInitial1, scoreDataInitial2, scoreDataInitial3))
# scoreDataMax = np.concatenate((scoreDataMax1, scoreDataMax2, scoreDataMax3))

# print('I', scoreDataInitial)
# print('F', scoreDataMax)
# np.save('/data2/mainul/DataAndGraph/AllTROTSInitial', scoreDataInitial)
# np.save('/data2/mainul/DataAndGraph/AllTROTSMax', scoreDataMax)
# The follwoing is for plotting the score distribution =======================================
# graphSavePath = '/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/'

bin_edges = np.linspace(0, 10, 11)
epsilon = 1e-10  # Small value to subtract from bin edges
bin_edges[:-1] -= epsilon
# # Plot the histogram
values1, bins = np.histogram(scoreDataInitial, bins= bin_edges)
print(bins)
# errors1 = np.std(values1, ddof=0)
values2, bins = np.histogram(scoreDataMax, bins=bin_edges)
print(values2)
# errors2 = np.std(values2, ddof=0)

score = np.arange(0,10,1)
# Set the width of the bars
bar_width = 0.35

# Set the positions for the bars
bar_positions_set1 = score - bar_width/2
# bin_centers1 = (bar_positions_set1[:-1] + bar_positions_set1[1:]) / 2
bar_positions_set2 = score + bar_width/2
# bin_centers2 = (bar_positions_set2[:-1] + bar_positions_set2[1:]) / 2

# Plot the first set of bars
plt.bar(bar_positions_set1, values1, width=bar_width, label='Initial Scores', color='blue')
# plt.errorbar(bar_positions_set1, values1, yerr=errors1, fmt='o', color='red', label='Error bars')

# Plot the second set of bars
plt.bar(bar_positions_set2, values2, width=bar_width, label='Final Scores', color='orange')
# plt.errorbar(bar_positions_set2, values2, yerr=errors2, fmt='o', color='red', label='Error bars')

plt.xticks(fontsize =17)
plt.yticks(fontsize =17)
# Add labels and title
plt.xlabel('Plan Scores', fontsize = 17)
plt.xlim(1,10)
plt.ylabel('Number of Plans', fontsize = 17)
# plt.title('Number of Patients vs Plan Scores')
# plt.legend(loc = 'upper center', fontsize = 17)
plt.tight_layout()
# plt.savefig(graphSavePath+'InitialAndFinalPlanComparisonACERUTSWPaper.png', dpi = 1200)
# plt.savefig(graphSavePath+'InitialAndFinalPlanComparisonACERUTSWrandom6Cedited.png', dpi = 1200)
plt.savefig(graphSavePath+'InitialAndFinalPlanComparisonACERCORT.png', dpi = 1200)
plt.show()
plt.close()
# ===========================================================================


# ScoreDistArray = np.zeros((8,4))
# for i in range(2,10):
# 	patientPosition = np.where((scoreDataInitial>=i) & (scoreDataInitial<i+1))
# 	patientScoreInit = scoreDataInitial[patientPosition]
# 	patientScoreFinal = scoreDataMax[patientPosition]
# 	ScoreDistArray[i-2][0] = np.mean(patientScoreInit)
# 	ScoreDistArray[i-2][1] = np.std(patientScoreInit, ddof = 0)
# 	ScoreDistArray[i-2][2] = np.mean(patientScoreFinal)
# 	ScoreDistArray[i-2][3] = np.std(patientScoreFinal, ddof = 0)

# print(ScoreDistArray)

# score = np.arange(1,9,1)
# # Set the width of the bars
# bar_width = 0.35

# # Set the positions for the bars
# bar_positions_set1 = score - bar_width/2
# # bin_centers1 = (bar_positions_set1[:-1] + bar_positions_set1[1:]) / 2
# bar_positions_set2 = score + bar_width/2
# # bin_centers2 = (bar_positions_set2[:-1] + bar_positions_set2[1:]) / 2


# # Plot the first set of bars
# plt.bar(bar_positions_set1, ScoreDistArray[:,0], width=bar_width, color='blue')
# plt.errorbar(bar_positions_set1, ScoreDistArray[:,0], yerr=ScoreDistArray[:,1], fmt='o', capsize=5, color='black')

# # Plot the second set of bars
# plt.bar(bar_positions_set2, ScoreDistArray[:,2], width=bar_width, color='orange')
# plt.errorbar(bar_positions_set2, ScoreDistArray[:,2], yerr=ScoreDistArray[:,3], fmt='o', capsize=5, color='black')


# plt.xticks(fontsize =17)
# plt.yticks(fontsize =17)
# plt.xlabel('Plan Groups', fontsize = 17)
# plt.ylabel('Plan Scores', fontsize = 17)
# # plt.title('Plan Scores vs Patient Group Index', fontsize = 28)
# # plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fancybox=True, shadow=True)
# # plt.legend(fontsize = 24)
# plt.ylim(0,10)
# plt.tight_layout()
# plt.savefig(graphSavePath+'PatientIndexGraphRandom6CEdited.png', dpi = 1200)
# # plt.savefig(graphSavePath+'PatientIndexGraphPaper.png', dpi = 1200)
# # plt.show()