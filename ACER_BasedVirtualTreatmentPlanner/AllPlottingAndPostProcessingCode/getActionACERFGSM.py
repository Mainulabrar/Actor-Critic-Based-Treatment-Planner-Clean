import numpy as np


data_result_path = '/data2/mainul/DataAndGraph/'

# # The following part is for getting from the Mixed Array=============================
# PolicyArray = np.load("/data2/mainul/DataAndGraph/PolicyFGSMACER.npy")
# print(PolicyArray.shape[1])
# UnperturbedArray = np.zeros((50, 18))
# perturbedArray = np.zeros((50, 18))

# for i in range(PolicyArray.shape[0]):
# 	if i%2 == 1:
# 		perturbedArray[int((i-1)/2),:] = PolicyArray[i,:]
# 	else:
# 		# print(i/2)
# 		UnperturbedArray[int(i/2),:] = PolicyArray[i,:]

# np.save(data_result_path+ 'UnperturbedACER', UnperturbedArray)
# np.save(data_result_path+ 'perturbedACER', perturbedArray)
# # =====================================================================

# UnperturbedArray = np.load(data_result_path+"UnperturbedACER.npy")
# perturbedArray = np.load(data_result_path+"perturbedACER.npy")

UnperturbedArray = np.load(data_result_path+"unperturbedACERPaper01.npy")
perturbedArray = np.load(data_result_path+"perturbedValueACERPaper01.npy")


print('UnperturbedArray',UnperturbedArray)
print('perturbedArray', perturbedArray)

max_indicesU = np.argmax(UnperturbedArray, axis = 1)
max_valueU = np.max(UnperturbedArray, axis =1)
print('maxValue', max_valueU)
# print(UnperturbedArray)
print(max_indicesU)
max_valueUchanged = perturbedArray[np.arange(perturbedArray.shape[0]), max_indicesU]
print('maxUchanged',max_valueUchanged)
max_indicesP = np.argmax(perturbedArray, axis = 1)
max_valueP = np.max(perturbedArray, axis = 1)
print('maxValuep',max_valueP)
# print(perturbedArray)
# PercentageChange = ((max_valueP - max_valueU)*100)/max_valueU
PercentageChange = ((max_valueUchanged - max_valueU)*100)/max_valueU
PercentageChange30 = PercentageChange[1:31]
print(np.mean(PercentageChange30))
print(np.std(PercentageChange30, ddof = 0))
print('PercentageChange', PercentageChange)
print(max_indicesP)
print(max_indicesU - max_indicesP)


