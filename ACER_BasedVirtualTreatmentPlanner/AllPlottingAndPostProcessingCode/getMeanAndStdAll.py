import numpy as np

InitialScore1 = np.load("/data2/mainul/DataAndGraph/scoreDataInitialCORT.npy").flatten()
print(InitialScore1.shape)
InitialScore2 = np.load("/data2/mainul/DataAndGraph/AllInitialScoreACERrandom6CEdited.npy")
print(InitialScore2.shape)
InitialScore3 = np.load("/data2/mainul/DataAndGraph/AllInitialScoreACERPaper.npy")
print(InitialScore3.shape)

InitialScore4 = np.load("/data2/mainul/DataAndGraph/AllTROTSInitial.npy")
print(InitialScore4.shape)
InitialScore5 = np.array([4.651588583737209959e+00,
2.000000000000000000e+00,
5.000000000000000000e+00,
2.998168162667156800e+00,
5.000000000000000000e+00,
2.366569916339423152e+00,
5.000000000000000000e+00,
5.000000000000000000e+00,
3.195770392749245126e+00,
5.000000000000000000e+00,
5.409228115567055184e+00,
5.000000000000000000e+00,
5.981427434332715620e+00,
3.771090959054686831e+00,
5.000000000000000000e+00,
4.819601507808292773e+00,
2.000000000000000000e+00,
5.000000000000000000e+00,
2.730719912071808864e+00,
5.000000000000000000e+00,
2.177053098856070346e+00,
5.000000000000000000e+00,
5.000000000000000000e+00,
3.275528700906344515e+00,
5.000000000000000000e+00,
5.555843035791290241e+00,
5.000000000000000000e+00,
5.455221897422271482e+00,
3.875515251442704034e+00,
5.000000000000000000e+00,
5.000000000000000000e+00,
4.000000000000000000e+00,
4.000000000000000000e+00,
4.428571428571428825e+00,
4.000000000000000000e+00,
2.000000000000000000e+00,
3.834425664022078184e+00,
2.388754743014833259e+00,
4.000000000000000000e+00,
4.000000000000000000e+00,
5.000000000000000000e+00,
4.000000000000000000e+00,
4.000000000000000000e+00,
4.000000000000000000e+00,
5.000000000000000000e+00,
4.000000000000000000e+00,
5.000000000000000000e+00,
4.674715419110038184e+00,
4.000000000000000000e+00,
4.000000000000000000e+00,
5.217687074829932214e+00,
4.000000000000000000e+00,
2.000000000000000000e+00,
4.014487754398069264e+00,
3.000000000000000000e+00,
5.000000000000000000e+00,
4.000000000000000000e+00,
4.000000000000000000e+00,
3.464298033804760202e+00,
4.000000000000000000e+00
])
print(InitialScore5.shape)
InitialScore6 = np.array([])
print(InitialScore6.shape)

InitialScore = np.concatenate((InitialScore1,InitialScore2,InitialScore3,InitialScore4,InitialScore5,InitialScore6))
# InitialScore = np.load('/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/AllInitialScoreACER.npy')

MeanInit = np.mean(InitialScore)
MeanInitPaper = np.mean(InitialScore3)

stdInitPaper = np.std(InitialScore3)
stdInit = np.std(InitialScore, ddof = 0)
print(InitialScore.shape)

MeanInitRandom = np.mean(InitialScore2)

stdInitRandom = np.std(InitialScore2, ddof= 0)

MeanInitTROTS = np.mean(InitialScore4)

stdInitTROTS = np.std(InitialScore4, ddof= 0)

# print('MeanInitRandom', MeanInitRandom)
# print('stdInitRandom', stdInitRandom)

print('MeanInitTROTS', MeanInitTROTS)
print('stdInitTROTS', stdInitTROTS)

# print('MeanInitPaper', MeanInitPaper)
# print('stdInitPaper', stdInitPaper)
print('MeanInit', MeanInit)
print('stdInit', stdInit)
print('InitScoreSize', InitialScore.size)

FinalScore1 = np.load("/data2/mainul/DataAndGraph/scoreDataMaxCORT.npy").flatten()
FinalScore2 = np.array([9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
8.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
8.000000000000000000e+00,
7.342391667223075125e+00,
9.000000000000000000e+00,
8.000000000000000000e+00,
8.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
8.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00,
9.000000000000000000e+00
])
FinalScore3 = np.array([])
FinalScore4 = np.load("/data2/mainul/DataAndGraph/AllFinalScoreACERrandom6CEdited.npy")
FinalScore5 = np.load("/data2/mainul/DataAndGraph/AllFinalScoreACERPaper.npy")
FinalScore6 = np.load("/data2/mainul/DataAndGraph/AllTROTSMax.npy")

print(FinalScore5[(FinalScore5 <9) & (FinalScore5>=7)])


MeanFinalPaper = np.mean(FinalScore5)

stdFinalPaper = np.std(FinalScore5, ddof = 0)
stdInit = np.std(InitialScore, ddof = 0)

# print('MeanFinalPaper', MeanFinalPaper)
# print('stdFinalPaper', stdFinalPaper)

MeanFinalRandom = np.mean(FinalScore4)
stdFinalRandom = np.std(FinalScore4, ddof = 0)

MeanFinalTROTS = np.mean(FinalScore6)
stdFinalTROTS = np.std(FinalScore6, ddof = 0)

# print('MeanFinalRandom', MeanFinalRandom)
# print('stdFinalRandom', stdFinalRandom)

print('MeanFinalTROTS', MeanFinalTROTS)
print('stdFinalTROTS', stdFinalTROTS)

# MeanFinalPaper = np.mean(FinalScore4)

# stdFinalPaper = np.std(FinalScore4, ddof = 0)

FinalScore = np.concatenate((FinalScore1,FinalScore2,FinalScore3,FinalScore4,FinalScore5,FinalScore6))

print("nines", FinalScore[FinalScore == 9].shape)
print("eight to nine", FinalScore[(9.0>FinalScore) & (FinalScore>= 8.0)])
print("seven to eight", FinalScore[(8.0>FinalScore) & (FinalScore >= 7.0)].shape)
# FinalScore = np.load('/home/mainul/Actor-critic-based-treatment-planning/DataAndGraph/AllFinalScoreACER.npy')
MeanFinal = np.mean(FinalScore)
stdFinal = np.std(FinalScore, ddof = 0)

print(FinalScore.shape)
print('MeanFinal', MeanFinal)
print('stdFinal', stdFinal)
