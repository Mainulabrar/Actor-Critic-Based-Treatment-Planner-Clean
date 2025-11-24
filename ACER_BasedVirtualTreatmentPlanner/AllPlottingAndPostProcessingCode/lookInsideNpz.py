import numpy as np

data_path = "/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/66tpptuning120499.npz"
data = np.load(data_path)

for key in data.keys():
    print(f"Array '{key}':")
    print(data[key])
