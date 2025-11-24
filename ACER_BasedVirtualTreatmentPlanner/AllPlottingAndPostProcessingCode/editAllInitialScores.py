import numpy as np

previousInit = np.load("/data2/mainul/DataAndGraph/AllInitialScoreACERrandom6C.npy")
print("pevious", previousInit)

NewInit = np.load("/data2/mainul/DataAndGraph/AllInitialScoreACERrandom3C1.npy")
NewInit3C = np.load("/data2/mainul/DataAndGraph/AllInitialScoreACERrandom3C.npy")
print("New", NewInit)

previousInit[9] = NewInit[0]
previousInit[13] = NewInit[1]
previousInit[88] = NewInit[2]
previousInit[91] = NewInit[3]
previousInit[138] = NewInit[4]

previousInit[14] = NewInit3C[1] 
previousInit[140] = NewInit3C[4]
print(previousInit)

NewInitBefore92 = previousInit[0: 135]
print('NewInitBefore92', NewInitBefore92)

NewInitAfter92 = previousInit[135:147]
print("NewInitAfter92", NewInitAfter92)

Init92 = np.load("/data2/mainul/DataAndGraph/AllInitialScoreACERrandom3C092.npy") 

EditeInit = np.concatenate((NewInitBefore92,Init92,NewInitAfter92))
print("edited", EditeInit)
