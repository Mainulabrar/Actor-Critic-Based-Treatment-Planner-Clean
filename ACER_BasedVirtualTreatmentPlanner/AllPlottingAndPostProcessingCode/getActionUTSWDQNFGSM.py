import numpy as np
import re

unperturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSMPaper/unperturbedValueArray1.npy")
# unperturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSM/unperturbedValueArray.npy")

max_indicesU = np.argmax(unperturbedArray, axis = 1)
maxValueU = np.max(unperturbedArray, axis = 1)
print(unperturbedArray)
print(max_indicesU)

perturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSMPaper/perturbedValueArray1.npy")
# perturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSM/perturbedValueArray.npy")

max_indicesP = np.argmax(perturbedArray, axis = 1)
maxValueP = perturbedArray[np.arange(perturbedArray.shape[0]), max_indicesU]
print(perturbedArray)
print(max_indicesP)
print(max_indicesU.size)
print('maxValueP', maxValueP)
PercentageChange = ((maxValueP - maxValueU)*100)/maxValueU
PercentageChange30 = PercentageChange[1:31]
print(PercentageChange30.size)
print(np.mean(PercentageChange30))
print(np.std(PercentageChange30, ddof = 0))
# print(((maxValueP - maxValueU)*100)/maxValueU)

print(max_indicesU - max_indicesP)