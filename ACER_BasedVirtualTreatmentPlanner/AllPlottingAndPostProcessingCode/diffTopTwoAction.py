import numpy as np
import re

# unperturbedArray = np.load("/data2/mainul/DataAndGraph/unperturbedACERPaper.npy")
unperturbedArray = np.load("/data2/mainul/DataAndGraph/unperturbedACERPaper1.npy")
# unperturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSMPaper/unperturbedValueArray.npy")
# unperturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSM/unperturbedValueArray.npy")
unperturbedArraySort = np.sort(unperturbedArray, axis=1)[:, ::-1]
print(unperturbedArray.shape)
print(unperturbedArraySort)

highest = unperturbedArraySort[:, 0]
second_highest = unperturbedArraySort[:, 1]

percentageDiff = ((highest - second_highest)/(highest))*100
print(percentageDiff.shape)

print(np.mean(percentageDiff), np.std(percentageDiff, ddof = 0 ))

# next part is for DQN =================================================

# # perturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSMPaper/perturbedValueArray.npy")
# perturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSMPaper/perturbedValueArray.npy")

# max_indicesU = np.argmax(unperturbedArray, axis = 1)
# maxValueU = np.max(unperturbedArray, axis = 1)
# # print(unperturbedArray)
# # print(max_indicesU)

# # perturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSMPaper/perturbedValueArray.npy")
# # perturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSM/perturbedValueArray.npy")
# perturbedArray = np.load("/data2/mainul/DataAndGraphDQN/FGSMPaper/perturbedValueArray.npy")

# max_indicesP = np.argmax(perturbedArray, axis = 1)
# maxValueP = perturbedArray[np.arange(perturbedArray.shape[0]), max_indicesU]
# # print(perturbedArray)
# # print(max_indicesP)
# # print(max_indicesU.size)
# # print('maxValueP', maxValueP)
# PercentageChange = ((maxValueP - maxValueU)*100)/maxValueU
# PercentageChange30 = PercentageChange[1:31]
# # print(PercentageChange30.size)
# # print(np.mean(PercentageChange30))
# # print(np.std(PercentageChange30, ddof = 0))
# # # print(((maxValueP - maxValueU)*100)/maxValueU)

# # print(max_indicesU - max_indicesP)
