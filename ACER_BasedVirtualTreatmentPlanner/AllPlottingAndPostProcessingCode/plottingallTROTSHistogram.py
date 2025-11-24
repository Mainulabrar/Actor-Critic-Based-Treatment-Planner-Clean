# bin_edges = np.linspace(0, 10, 11)
# epsilon = 1e-10  # Small value to subtract from bin edges
# bin_edges[:-1] -= epsilon
# # # Plot the histogram
# values1, bins = np.histogram(scoreDataInitial, bins= bin_edges)
# print(bins)
# # errors1 = np.std(values1, ddof=0)
# values2, bins = np.histogram(scoreDataMax, bins=bin_edges)
# print(values2)
# # errors2 = np.std(values2, ddof=0)

# score = np.arange(0,10,1)
# # Set the width of the bars
# bar_width = 0.35

# # Set the positions for the bars
# bar_positions_set1 = score - bar_width/2
# # bin_centers1 = (bar_positions_set1[:-1] + bar_positions_set1[1:]) / 2
# bar_positions_set2 = score + bar_width/2
# # bin_centers2 = (bar_positions_set2[:-1] + bar_positions_set2[1:]) / 2

# # Plot the first set of bars
# plt.bar(bar_positions_set1, values1, width=bar_width, label='InitialScores', color='blue')
# # plt.errorbar(bar_positions_set1, values1, yerr=errors1, fmt='o', color='red', label='Error bars')

# # Plot the second set of bars
# plt.bar(bar_positions_set2, values2, width=bar_width, label='MaxScores', color='orange')
# # plt.errorbar(bar_positions_set2, values2, yerr=errors2, fmt='o', color='red', label='Error bars')

# # Add labels and title
# plt.xlabel('Plan Scores')
# plt.ylabel('Number of Patients')
# plt.title('Number of Patients vs Plan Scores')
# plt.legend()
# plt.savefig(graphSavePath+'InitialAndFinalPlanComparisonACERUTSWRandom6C.png', dpi = 1200)
# plt.show()
