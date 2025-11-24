import numpy as np

InitialScore = np.load("/data2/mainul/DataAndGraph/scoreDataInitialCORT.npy")
# InitialScore = np.load('/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/AllInitialScoreACER.npy')
MeanInit = np.mean(InitialScore)
stdInit = np.std(InitialScore, ddof = 0)
print(InitialScore)

print('MeanInit', MeanInit)
print('stdInit', stdInit)
print('InitScoreSize', InitialScore.size)

FinalScore = np.load("/data2/mainul/DataAndGraph/scoreDataMaxCORT.npy")
# FinalScore = np.load('/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/AllFinalScoreACER.npy')
MeanFinal = np.mean(FinalScore)
stdFinal = np.std(FinalScore, ddof = 0)

print(FinalScore)
print('MeanFinal', MeanFinal)
print('stdFinal', stdFinal)
