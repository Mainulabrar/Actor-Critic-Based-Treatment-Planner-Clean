import numpy as np
#import time
#import math as m
from math import pi
def planIQ_train(MPTV, MBLA, MREC, xVec,pdose,check):
    # score of treatment plan, two kinds of scores:
    # 1: score from standard criterion, 2: score_fined for self-defined in order to emphasize ptv
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)

    scoreall = np.zeros((11,))
    #    tt = time.time()
    ind = round(0.03 / 0.015) - 1

    max_limit = 1

    # avg_DPTV = (DPTV[ind] + DPTV[ind + 1] + DPTV[ind - 1]+DPTV[ind + 2] + DPTV[ind + 3]) / 5
    avg_DPTV = (DPTV[ind] + DPTV[ind + 1] + DPTV[ind - 1]) / 3
    # avg_DPTV = DPTV[ind]
    if check == True:
        print("avg_DPTV:",avg_DPTV)
    score2 =  (avg_DPTV - 1.1)/(-0.03)
    if score2 > max_limit:
        score2 = 1
    if score2 < 0:
        score2 = 0
    delta2 = 0.08
    if (avg_DPTV > 1.05):
        score2_fine = (1 / pi * np.arctan(-(avg_DPTV - 1.075) / delta2) + 0.5) * 8
    else:
        score2_fine = 6########################################
    # score2_fine = score2
    scoreall[0] = score2

    DBLA1 = DBLA[DBLA >= 1.01]
    avg_DBLA1 = DBLA1.shape[0] / DBLA.shape[0]
    score5 = (avg_DBLA1 - 0.2 )/(-0.05)
    if score5 > max_limit:
        score5 = 1
    if score5 < 0:
        score5 = 0
    delta3 = 0.05
    if avg_DBLA1 < 0.2:
        score3_fine = 1 / pi * np.arctan(-(avg_DBLA1 - 0.175) / delta3) + 0.5
    else:
        score3_fine = 0
    scoreall[3] = score5

    DBLA2 = DBLA[DBLA >= pdose * 0.947]
    avg_DBLA2 = DBLA2.shape[0] / DBLA.shape[0]
    score6 = (avg_DBLA2 - 0.3 )/(-0.05)
    if score6 > max_limit:
        score6 = 1
    if score6 < 0:
        score6 = 0
    delta4 = 0.05
    if avg_DBLA2 < 0.3:
        score4_fine = 1 / pi * np.arctan(-(avg_DBLA2 - 0.55 / 2) / delta4) + 0.5
    else:
        score4_fine = 0
    scoreall[4] = score6

    DBLA3 = DBLA[DBLA >= 0.8838]
    avg_DBLA3 = DBLA3.shape[0] / DBLA.shape[0]
    score7 = (avg_DBLA3 - 0.4 )/(-0.05)
    if score7 > max_limit:
        score7 = 1
    if score7 < 0:
        score7 = 0
    delta5 = 0.05
    if avg_DBLA3 < 0.4:
        score5_fine = 1 / pi * np.arctan(-(avg_DBLA3 - 0.75 / 2) / delta5) + 0.5
    else:
        score5_fine = 0
    scoreall[5] = score7

    DBLA4 = DBLA[DBLA >= 0.8207]
    avg_DBLA4 = DBLA4.shape[0] / DBLA.shape[0]
    score8 = (avg_DBLA4 - 0.55)/(-0.05)
    if score8 > max_limit:
        score8 = 1
    if score8 < 0:
        score8 = 0
    delta6 = 0.05
    if avg_DBLA4 < 0.55:
        score6_fine = 1 / pi * np.arctan(-(avg_DBLA4 - 1.05 / 2) / delta6) + 0.5
    else:
        score6_fine = 0
    scoreall[6] = score8

    DREC1 = DREC[DREC >= 0.947]
    avg_DREC1 = DREC1.shape[0] / DREC.shape[0]
    score9 = (avg_DREC1 - 0.2)/(-0.05)
    if score9 > max_limit:
        score9 = 1
    if score9 < 0:
        score9 = 0
    delta7 = 0.05
    if avg_DREC1 < 0.2:
        score7_fine = 1 / pi * np.arctan(-(avg_DREC1 - 0.35 / 2) / delta7) + 0.5
    else:
        score7_fine = 0
    scoreall[7] = score9

    DREC2 = DREC[DREC >= 0.8838]
    avg_DREC2 = DREC2.shape[0] / DREC.shape[0]
    score10 = (avg_DREC2 - 0.3)/(-0.05)
    if score10 > max_limit:
        score10 = 1
    if score10 < 0:
        score10 = 0
    delta8 = 0.05
    if avg_DREC2 < 0.3:
        score8_fine = 1 / pi * np.arctan(-(avg_DREC2 - 0.55 / 2) / delta8) + 0.5
    else:
        score8_fine = 0

    scoreall[8] = score10

    DREC3 = DREC[DREC >= 0.8207]
    avg_DREC3 = DREC3.shape[0] / DREC.shape[0]
    score11 = (avg_DREC3 - 0.4)/(-0.05)
    if score11 > max_limit:
        score11 = 1
    if score11 < 0:
        score11 = 0
    delta9 = 0.05
    if avg_DREC3 < 0.4:
        score9_fine = 1 / pi * np.arctan(-(avg_DREC3 - 0.75 / 2) / delta9) + 0.5
    else:
        score9_fine = 0

    scoreall[9] = score11

    DREC4 = DREC[DREC >= 0.7576]
    avg_DREC4 = DREC4.shape[0] / DREC.shape[0]
    score12 = (avg_DREC4 - 0.55)/(-0.05)
    if score12 > max_limit:
        score12 = 1
    if score12 < 0:
        score12 = 0
    delta10 = 0.05
    if avg_DREC4 < 0.55:
        score10_fine = 1 / pi * np.arctan(-(avg_DREC4 - 1.05 / 2) / delta10) + 0.5
    else:
        score10_fine = 0

    scoreall[10] = score12
    #    elapsedTime = time.time()-tt
    #    print('time:{}',format(elapsedTime))

    score = score2 + score5 + score6 + score7 + score8 + score9 + score10 + score11 + score12
    print(score2, score5, score6, score7, score8, score9, score10, score11, score12)
    if score2_fine > 0.5:
        score_fine = score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine
    else:
        score_fine = (
                    score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine)

    return score_fine, score, scoreall


#     scoreall = np.zeros((11,))
# #    tt = time.time()
#     ind = round(0.03/0.015)-1
#     a = 1/(1.07*pdose-1.1*pdose)#(1.07*pdose-1.1*pdose)
#     b = 1-a*1.07*pdose
#     score2 = a * (DPTV[ind] + DPTV[ind + 1] + DPTV[ind - 1]) / 3 + b
#     if score2>1:
#         score2=0
#     else:
#         score2=-1
# 
#     # score2_fine = score2
#     scoreall[0]=score2
# 
#     DBLA1 = DBLA[DBLA>=pdose*1.01]
#     a = 1/(0.15-0.20)
#     b = 1-a*0.15
#     score5 = a*DBLA1.shape[0]/DBLA.shape[0]+b
#     if score5>1:
#         score5=0
#     else:
#         score5=-1
# 
#     scoreall[3] = score5
# 
#     DBLA2 = DBLA[DBLA >= pdose * 0.947]
#     a = 1 / (0.25 - 0.30)
#     b = 1 - a * 0.25
#     score6 = a * DBLA2.shape[0] / DBLA.shape[0] + b
#     if score6 > 1:
#         score6 = 0
#     else:
#         score6 = -1
# 
#     scoreall[4] = score6
# 
#     DBLA3 = DBLA[DBLA >= pdose * 0.8838]
#     a = 1 / (0.35 - 0.40)
#     b = 1 - a * 0.35
#     score7 = a * DBLA3.shape[0] / DBLA.shape[0] + b
#     if score7 > 1:
#         score7 = 0
#     else:
#         score7 = -1
# 
#     scoreall[5] = score7
# 
#     DBLA4 = DBLA[DBLA >= pdose * 0.8207]
#     a = 1 / (0.5 - 0.55)
#     b = 1 - a * 0.5
#     score8 = a * DBLA4.shape[0] / DBLA.shape[0] + b
#     if score8 > 1:
#         score8 = 0
#     else:
#         score8 = -1
# 
#     scoreall[6] = score8
# 
# 
#     DREC1 = DREC[DREC >= pdose * 0.947]
#     a = 1 / (0.15 - 0.20)
#     b = 1 - a * 0.15
#     score9 = a * DREC1.shape[0] / DREC.shape[0] + b
#     if score9 > 1:
#         score9 = 0
#     else:
#         score9 = -1
# 
#     scoreall[7] = score9
# 
# 
#     DREC2 = DREC[DREC >= pdose * 0.8838]
#     a = 1 / (0.25 - 0.30)
#     b = 1 - a * 0.25
#     score10 = a * DREC2.shape[0] / DREC.shape[0] + b
#     if score10 > 1:
#         score10 = 0
#     else:
#         score10 = -1
# 
#     scoreall[8] = score10
# 
#     DREC3 = DREC[DREC >= pdose * 0.8207]
#     a = 1 / (0.35 - 0.40)
#     b = 1 - a * 0.35
#     score11 = a * DREC3.shape[0] / DREC.shape[0] + b
#     if score11 > 1:
#         score11 = 0
#     else:
#         score11 = -1
# 
#     scoreall[9] = score11
# 
#     DREC4 = DREC[DREC >= pdose * 0.7576]
#     a = 1 / (0.50 - 0.55)
#     b = 1 - a * 0.50
#     score12 = a * DREC4.shape[0] / DREC.shape[0] + b
#     if score12 > 1:
#         score12 = 0
#     else:
#         score12 = -1
# 
#     scoreall[10] = score12
# #    elapsedTime = time.time()-tt
# #    print('time:{}',format(elapsedTime))
# 
# 
#     score = score2+score5+score6+score7+score8+score9+score10+score11+score12
#     if (check == True):
#         print(score2, score5, score6, score7, score8, score9, score10, score11, score12)
#     score_fine= 0
# 
# 
#     return score_fine, score, scoreall


def planIQ_test(MPTV, MBLA, MREC, xVec,pdose):
    # score of treatment plan, two kinds of scores:
    # 1: score from standard criterion, 2: score_fined for self-defined in order to emphasize ptv
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)

    scoreall = np.zeros((11,))
#    tt = time.time()
    ind = round(0.03/0.015)-1
    a = 1/(1.07*pdose-1.1*pdose)#(1.07*pdose-1.1*pdose)
    b = 1-a*1.07*pdose
    score2 = a*(DPTV[ind]+DPTV[ind+1]+DPTV[ind-1])/3+b
    if score2>1:
        score2=1
    if score2<0:
        score2=0
    delta2 = 0.08
    if (DPTV[ind]+DPTV[ind+1]+DPTV[ind-1])/3>1.05:
        score2_fine = (1 / pi * np.arctan(-((DPTV[ind] + DPTV[ind + 1] + DPTV[ind - 1]) / 3 - (1.05 * pdose + 1.1 * pdose) / 2) / delta2) + 0.5)*8
    else:
        score2_fine=6
    # score2_fine = score2
    scoreall[0]=score2

    # ind = round(0.03/0.015)-1
    # a = 1/(1.05*pdose-1.07*pdose)#(1.07*pdose-1.1*pdose)
    # b = 1-a*1.05*pdose
    # score3 = a*(DPTV[ind]+DPTV[ind+1]+DPTV[ind+2]+DPTV[ind-1])/4+b
    # if score3>1:
    #     score3=1
    # if score3<0:
    #     score3=0
    # scoreall[1]=score3
    #
    # ind = round(0.03/0.015)-1
    # a = 1/(1.03*pdose-1.05*pdose)#(1.07*pdose-1.1*pdose)
    # b = 1-a*1.03*pdose
    # score4 = a*(DPTV[ind]+DPTV[ind+1]+DPTV[ind+2]+DPTV[ind+3]+DPTV[ind-1])/5+b
    # if score4>1:
    #     score4=1
    # if score4<0:
    #     score4=0
    # scoreall[2]=score4


    DBLA1 = DBLA[DBLA>=pdose*1.01]
    a = 1/(0.15-0.20)
    b = 1-a*0.15
    score5 = a*DBLA1.shape[0]/DBLA.shape[0]+b
    if score5>1:
        score5=1
    if score5<0:
        score5=0
    delta3 = 0.05
    if DBLA1.shape[0]/DBLA.shape[0]<0.2:
        score3_fine = 1 / pi * np.arctan(-(DBLA1.shape[0]/DBLA.shape[0] - (0.15 + 0.20) / 2) / delta3) + 0.5
    else:
        score3_fine=0
    scoreall[3] = score5

    DBLA2 = DBLA[DBLA >= pdose * 0.947]
    a = 1 / (0.25 - 0.30)
    b = 1 - a * 0.25
    score6 = a * DBLA2.shape[0] / DBLA.shape[0] + b
    if score6 > 1:
        score6 = 1
    if score6 < 0:
        score6 = 0
    delta4 = 0.05
    if DBLA2.shape[0] / DBLA.shape[0]<0.3:
        score4_fine = 1 / pi * np.arctan(-(DBLA2.shape[0] / DBLA.shape[0] - (0.25 + 0.30) / 2) / delta4) + 0.5
    else:
        score4_fine = 0
    scoreall[4] = score6

    DBLA3 = DBLA[DBLA >= pdose * 0.8838]
    a = 1 / (0.35 - 0.40)
    b = 1 - a * 0.35
    score7 = a * DBLA3.shape[0] / DBLA.shape[0] + b
    if score7 > 1:
        score7 = 1
    if score7 < 0:
        score7 = 0
    delta5 = 0.05
    if DBLA3.shape[0] / DBLA.shape[0]<0.4:
        score5_fine = 1 / pi * np.arctan(-(DBLA3.shape[0] / DBLA.shape[0] - (0.35 + 0.40) / 2) / delta5) + 0.5
    else:
        score5_fine = 0
    scoreall[5] = score7

    DBLA4 = DBLA[DBLA >= pdose * 0.8207]
    a = 1 / (0.5 - 0.55)
    b = 1 - a * 0.5
    score8 = a * DBLA4.shape[0] / DBLA.shape[0] + b
    if score8 > 1:
        score8 = 1
    if score8 < 0:
        score8 = 0
    delta6 = 0.05
    if DBLA4.shape[0] / DBLA.shape[0]<0.55:
        score6_fine = 1 / pi * np.arctan(-(DBLA4.shape[0] / DBLA.shape[0] - (0.5 + 0.55) / 2) / delta6) + 0.5
    else:
        score6_fine = 0
    scoreall[6] = score8


    DREC1 = DREC[DREC >= pdose * 0.947]
    a = 1 / (0.15 - 0.20)
    b = 1 - a * 0.15
    score9 = a * DREC1.shape[0] / DREC.shape[0] + b
    if score9 > 1:
        score9 = 1
    if score9 < 0:
        score9 = 0
    delta7 = 0.05
    if DREC1.shape[0] / DREC.shape[0]<0.2:
        score7_fine = 1 / pi * np.arctan(-(DREC1.shape[0] / DREC.shape[0] - (0.15 + 0.20) / 2) / delta7) + 0.5
    else:
        score7_fine = 0
    scoreall[7] = score9


    DREC2 = DREC[DREC >= pdose * 0.8838]
    a = 1 / (0.25 - 0.30)
    b = 1 - a * 0.25
    score10 = a * DREC2.shape[0] / DREC.shape[0] + b
    if score10 > 1:
        score10 = 1
    if score10 < 0:
        score10 = 0
    delta8 = 0.05
    if DREC2.shape[0] / DREC.shape[0]<0.3:
        score8_fine = 1 / pi * np.arctan(-(DREC2.shape[0] / DREC.shape[0] - (0.25 + 0.30) / 2) / delta8) + 0.5
    else:
        score8_fine = 0

    scoreall[8] = score10

    DREC3 = DREC[DREC >= pdose * 0.8207]
    a = 1 / (0.35 - 0.40)
    b = 1 - a * 0.35
    score11 = a * DREC3.shape[0] / DREC.shape[0] + b
    if score11 > 1:
        score11 = 1
    if score11 < 0:
        score11 = 0
    delta9 = 0.05
    if DREC3.shape[0] / DREC.shape[0]<0.4:
        score9_fine = 1 / pi * np.arctan(-(DREC3.shape[0] / DREC.shape[0] - (0.35 + 0.40) / 2) / delta9) + 0.5
    else:
        score9_fine = 0

    scoreall[9] = score11

    DREC4 = DREC[DREC >= pdose * 0.7576]
    a = 1 / (0.50 - 0.55)
    b = 1 - a * 0.50
    score12 = a * DREC4.shape[0] / DREC.shape[0] + b
    if score12 > 1:
        score12 = 1
    if score12 < 0:
        score12 = 0
    delta10 = 0.05
    if DREC4.shape[0] / DREC.shape[0]<0.55:
        score10_fine = 1 / pi * np.arctan(-(DREC4.shape[0] / DREC.shape[0] - (0.50 + 0.55) / 2) / delta10) + 0.5
    else:
        score10_fine = 0

    scoreall[10] = score12
#    elapsedTime = time.time()-tt
#    print('time:{}',format(elapsedTime))


    score = score2+score5+score6+score7+score8+score9+score10+score11+score12
    print(score2,score5,score6,score7,score8,score9,score10,score11,score12)
    if score2_fine>0.5:
        score_fine = score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine
    else:
        score_fine = (score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine)

    return score_fine, score, scoreall