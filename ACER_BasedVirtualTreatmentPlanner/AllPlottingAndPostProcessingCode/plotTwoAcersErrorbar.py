import numpy as np
import matplotlib.pyplot as plt

Y = np.load('/data2/mainul/DataAndGraph/All120499Policy5step0.npy')

means = np.mean(Y, axis = 0)
std_devs = np.std(Y, axis=0, ddof = 1)



x = [r"$inc\_t_{PTV}$", r"$dec\_t_{PTV}$", r"$inc\_t_{BLA}$", r"$dec\_t_{BLA}$", r"$inc\_t_{REC}$", r"$dec\_t_{REC}$", r'$inc\_\lambda_{PTV}$', r'$dec\_\lambda_{PTV}$', r'$inc\_\lambda_{BLA}$', r'$dec\_\lambda_{BLA}$', r'$inc\_\lambda_{REC}$', r'$dec\_\lambda_{REC}$', r'$inc\_V_{PTV}$', r'$dec\_V_{PTV}$', r'$inc\_V_{BLA}$', r'$dec\_V_{BLA}$', r'$inc\_V_{REC}$', r'$dec\_V_{REC}$']


plt.yscale('log')
# Plot the mean with standard deviation as error bars
# errorbar_plot = plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, label='Mean with Std Dev')

# plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, label='Mean with Std Dev')
plt.errorbar(x, means, yerr=std_devs, fmt='o', markersize = 10, capsize=10, elinewidth = 3,  label=r'$\mathrm{Mean \pm Std.}$', color = 'black')
# Set X-axis and Y-axis labels and the title
# plt.xlabel('Action Index')
plt.ylabel('Probability', fontsize = 20)
# plt.title('Mean and Standard Deviation of Each Action', fontsize = 16)
plt.xticks(ticks=x, rotation=45, fontsize = 20)
plt.yticks(fontsize = 20)

# Add a legend
plt.legend(fontsize = 20)
plt.tight_layout()
# plt.savefig('/data2/mainul/DataAndGraph/ACEREActionErrorbar159999.png', dpi = 1200)
# handles = [errorbar_plot]  # You can add more handles here if you have more plot elements
# labels = ['Mean with Std Dev']  # Custom label for the error bars plot

# # Add the custom legend to the plot
# plt.legend(handles, labels)

# Display the plot
plt.show()
