import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ConvergenceMap = np.load("/data2/mainul/DataAndGraph/ConvergenceArray.npy")

# xAxis = np.ara
plt.plot(((np.arange(0, len(ConvergenceMap)))*1), ConvergenceMap)

def x_label_formatter(x, pos):
    return f'{int(x * 0.5)}'

# Set the x-axis label formatter to multiply values by 500
plt.gca().xaxis.set_major_formatter(FuncFormatter(x_label_formatter))

plt.xlabel(r'Epoch ($ \times 10^3$)', fontsize = 14)
plt.ylabel('Plan Score', fontsize = 14)
plt.title("Convergence Map", fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize=12) 
plt # Major ticks font size
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('/data2/mainul/DataAndGraph/ConvergenceMap.png', dpi = 1200)
plt.show()