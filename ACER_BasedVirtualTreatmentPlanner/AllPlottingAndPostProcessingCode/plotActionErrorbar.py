import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame, assuming the first row is not the header
df = pd.read_csv('/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/Data_percentage.csv', header=None)  # If no header, specify header=None

# Calculate the mean and standard deviation for each column
means = df.mean()
std_devs = df.std()

# Create a range of column indices for the x-axis
# x = range(df.shape[1])
x = ['inc_tPTV', 'dec_tPTV', 'inc_tBLA', 'dec_tBLA', 'inc_tREC', 'dec_tREC', 'inc_lambdaPTV', 'dec_lambdaPTV', 'inc_lambdaBLA', 'dec_lambdaBLA', 'inc_lambdaREC', 'dec_lambdaREC', 'inc_VPTV', 'dec_VPTV', 'inc_VBLA', 'dec_VBLA', 'inc_VREC', 'dec_VREC']

plt.yscale('log')
# Plot the mean with standard deviation as error bars
# errorbar_plot = plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, label='Mean with Std Dev')

plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, label='Mean with Std Dev')
# Set X-axis and Y-axis labels and the title
# plt.xlabel('Action Index')
plt.ylabel('Mean Probability', fontsize = 16)
plt.title('Mean and Standard Deviation of Each Action', fontsize = 16)
plt.xticks(ticks=x, rotation=45, fontsize = 15)
plt.yticks(fontsize = 14)

# Add a legend
plt.legend(fontsize = 16)
plt.tight_layout()
plt.savefig('/data2/mainul/DataAndGraph/ACEREActionErrorbar.png', dpi = 1200)
# handles = [errorbar_plot]  # You can add more handles here if you have more plot elements
# labels = ['Mean with Std Dev']  # Custom label for the error bars plot

# # Add the custom legend to the plot
# plt.legend(handles, labels)

# Display the plot
plt.show()
