import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data = np.load("/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/Data_percentage.csv")

df = pd.read_csv('/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/Data_percentage.csv', header = None)

# Plot each column against its index
# for column in df.columns:
#     plt.plot(df.index, df[column], label=None)

for col_idx in range(df.shape[1]):  # Loop through each column index
    plt.plot(df.index, df[col_idx], label=f'Action {col_idx}')

plt.yscale('log')
# Add labels and title
plt.xlabel('Patient Case')
plt.ylabel('Probability')
plt.title('Plot of Probability vs Patient Case')

# # Add a legend to show which line corresponds to which column
# plt.legend()

# Add a legend and place it outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Actions")

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Display the plot
plt.show()
