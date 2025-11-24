import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# data_path = '/data2/mainul/DQNYinResults65/'


# test_set = ['014', '015', '023', '073', '098']


# Load the CSV file into a pandas DataFrame, assuming the first row is not the header
df = pd.read_csv('/data2/mainul/DataAndGraphDQN/DQNActionProb.csv', header=None)  # If no header, specify header=None
numpyData = df.to_numpy()
print(df.to_numpy())

# Step 1: Subtract the max value from the input array to prevent overflow
# numpyData_exp = np.exp(numpyData)

# # Step 2: Compute the softmax
# NumpyDatasoftmax = numpyData_exp / np.sum(numpyData_exp)
# print('softmax',NumpyDatasoftmax)
# # numpyDataWithoutStatic = numpyData[:, ::2]
# columnsToDelete = np.arange(1, 26, 3)
# numpyDataWithoutStatic = np.delete(NumpyDatasoftmax, columnsToDelete, axis = 1)
# numpyDataWithoutStaticRaw = np.delete(numpyData, columnsToDelete, axis = 1)

# print(columnsToDelete)
# # print(numpyDataWithoutStatic)
# print(numpyDataWithoutStaticRaw)

print(numpyData.shape)

numpyDataWIthoutLastSix = numpyData[:, 0:18]
numpyDataWIthoutLastSix = numpyData

print(numpyDataWIthoutLastSix.shape)

# df = pd.DataFrame(numpyDataWithoutStatic)
# df = pd.DataFrame(numpyDataWithoutStaticRaw)
# df = pd.DataFrame(numpyDataWIthoutLastSix)
# # Calculate the mean and standard deviation for each column
# means = df.mean()
# std_devs = df.std()

means = np.mean(numpyDataWIthoutLastSix, axis = 0)
std_devs = np.std(numpyDataWIthoutLastSix, axis = 0, ddof = 1)

print('Means', means)
# Create a range of column indices for the x-axis
# x = range(df.shape[1])
# x = ['inc_tPTV', 'dec_tPTV', 'inc_tBLA', 'dec_tBLA', 'inc_tREC', 'dec_tREC', 'inc_lambdaPTV', 'dec_lambdaPTV', 'inc_lambdaBLA', 'dec_lambdaBLA', 'inc_lambdaREC', 'dec_lambdaREC', 'inc_VPTV', 'dec_VPTV', 'inc_VBLA', 'dec_VBLA', 'inc_VREC', 'dec_VREC']
# x x = [r"$inc\_t_{PTV}$", r"$dec\_t_{PTV}$", r"$inc\_t_{BLA}$", r"$dec\_t_{BLA}$", r"$inc\_t_{REC}$", r"$dec\_t_{REC}$", r'$inc\_\lambda_{PTV}$', r'$dec\_\lambda_{PTV}$', r'$inc\_\lambda_{BLA}$', r'$dec\_\lambda_{BLA}$', r'$inc\_\lambda_{REC}$', r'$dec\_\lambda_{REC}$', r'$inc\_V_{PTV}$', r'$dec\_V_{PTV}$', r'$inc\_V_{BLA}$', r'$dec\_V_{BLA}$', r'$inc\_V_{REC}$', r'$dec\_V_{REC}$']

# x = [r"$inc\_t_{PTV}$", r"$no\ change\_t_{PTV}$",  r"$dec\_t_{PTV}$", r"$inc\_t_{BLA}$", r"$no change\_t_{BLA}$",  r"$dec\_t_{BLA}$", r"$inc\_t_{REC}$", r"$no change\_t_{REC}$", r"$dec\_t_{REC}$", r'$inc\_\lambda_{PTV}$', r"$no change\_\lambda_{PTV}$", r'$dec\_\lambda_{PTV}$', r'$inc\_\lambda_{BLA}$', r"$no change\_\lambda_{BLA}$", r'$dec\_\lambda_{BLA}$', r'$inc\_\lambda_{REC}$', r"$no change\_\lambda_{REC}$", r'$dec\_\lambda_{REC}$']
x = [r"$inc\_t_{PTV}$", r"$no\ change\_t_{PTV}$",  r"$dec\_t_{PTV}$", r"$inc\_t_{BLA}$", r"$no change\_t_{BLA}$",  r"$dec\_t_{BLA}$", r"$inc\_t_{REC}$", r"$no change\_t_{REC}$", r"$dec\_t_{REC}$", r'$inc\_\lambda_{PTV}$', r"$no change\_\lambda_{PTV}$", r'$dec\_\lambda_{PTV}$', r'$inc\_\lambda_{BLA}$', r"$no change\_\lambda_{BLA}$", r'$dec\_\lambda_{BLA}$', r'$inc\_\lambda_{REC}$', r"$no change\_\lambda_{REC}$", r'$dec\_\lambda_{REC}$', r'$inc\_V_{PTV}$', r"$no\_change\_V_{PTV}$", r'$dec\_V_{PTV}$', r'$inc\_V_{BLA}$', r"$no\_change\_V_{BLA}$", r'$dec\_V_{BLA}$', r'$inc\_V_{REC}$', r"$no\_change\_V_{REC}$", r'$dec\_V_{REC}$']


# x = np.arange(27)

# plt.yscale('log')
# Plot the mean with standard deviation as error bars
# errorbar_plot = plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, label='Mean with Std Dev')

plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, label=r'$Mean \pm Std.$')
# Set X-axis and Y-axis labels and the title
# plt.xlabel('Action Index')
plt.ylabel('Mean Value', fontsize = 16)
plt.title('Mean and Standard Deviation of Probability for Each Action', fontsize = 16)
# plt.xticks(ticks=x, rotation=45, fontsize = 14)
# plt.xticks(x, [f"Label {i}" for i in x], rotation=45, ha='right')
plt.xticks(ticks = x, rotation=45, ha='right', fontsize = 16)
plt.yticks(fontsize = 14)

# Add a legend
plt.legend(fontsize = 15)
plt.tight_layout()

# handles = [errorbar_plot]  # You can add more handles here if you have more plot elements
# labels = ['Mean with Std Dev']  # Custom label for the error bars plot

# # Add the custom legend to the plot
# plt.legend(handles, labels)
# plt.savefig('/data2/mainul/DataAndGraphDQN/DQNActionErrorbar.png', dpi = 1200)
# Display the plot
plt.show()
