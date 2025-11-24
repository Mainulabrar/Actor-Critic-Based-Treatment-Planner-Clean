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
print("edited", len(EditeInit[3:]))
np.save("/data2/mainul/DataAndGraph/AllInitialScoreACERrandom6CEdited", EditeInit[3:])

# The next part is for processing the final array ===============================================
previousFinal = np.load("/data2/mainul/DataAndGraph/AllFinalScoreACERrandom6C.npy")
print("pevious", previousFinal)


previousFinal[9] = 9
previousFinal[13] = 9
previousFinal[88] = 9
previousFinal[91] = 9
previousFinal[138] = 9

previousFinal[14] = 9 
previousFinal[140] = 9

NewFinalBefore92 = previousFinal[0:135]
print('NewFinalBefore92', NewFinalBefore92)

NewFinalAfter92 = previousFinal[135:147]
print(NewFinalAfter92)


Final92 = np.load("/data2/mainul/DataAndGraph/AllFinalScoreACERrandom3C092.npy") 

EditeFinal = np.concatenate((NewFinalBefore92, Final92, NewFinalAfter92))

np.save("/data2/mainul/DataAndGraph/AllFinalScoreACERrandom6CEdited", EditeFinal[3:])

print("edited Final", len(EditeFinal[3:]))