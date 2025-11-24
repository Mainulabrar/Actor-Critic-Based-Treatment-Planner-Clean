import math as m
from .myconfig import *

def action_original(action,tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC):
    # The upper and lower limits for the Treatment planning parameters
    paraMax = 100000  # change in validation as well
    paraMin = 0
    paraMax_tPTV = 1.2
    paraMin_tPTV = 1
    paraMax_tOAR = 1
    paraMax_VOAR = 1
    paraMax_VPTV = 0.3

    if action > actionnum() or action < 0:
        print("ERROR! Action value is out of range")
    else:
        if action == 0:
            action_factor = 1.1
            tPTV = tPTV * action_factor
            if tPTV >= paraMax_tPTV:
                tPTV = paraMax_tPTV
        if action == 2:
            action_factor = 0.9
            tPTV = tPTV * action_factor
            if tPTV <= paraMin_tPTV:
                tPTV = paraMin_tPTV
        if action == 3:
            action_factor = 1.25
            tBLA = tBLA * action_factor
            if tBLA >= paraMax_tOAR:
                tBLA = paraMax_tOAR
        if action == 5:
            action_factor = 0.8
            tBLA = tBLA * action_factor
            if tBLA <= paraMin:
                tBLA = paraMin
        if action == 6:
            action_factor = 1.25
            tREC = tREC * action_factor
            if tREC >= paraMax_tOAR:
                tREC = paraMax_tOAR
        if action == 8:
            action_factor = 0.8
            tREC = tREC * action_factor
            if tREC <= paraMin:
                tREC = paraMin
        if action == 9:
            action_factor = m.exp(0.5)
            lambdaPTV = lambdaPTV * action_factor
            if lambdaPTV >= paraMax:
                lambdaPTV = paraMax
        if action == 11:
            action_factor = m.exp(-0.5)
            lambdaPTV = lambdaPTV * action_factor
            if lambdaPTV <= paraMin:
                lambdaPTV = paraMin
        if action == 12:
            action_factor = m.exp(0.5)
            lambdaBLA = lambdaBLA * action_factor
            if lambdaBLA >= paraMax:
                lambdaBLA = paraMax
        if action == 14:
            action_factor = m.exp(-0.5)
            lambdaBLA = lambdaBLA * action_factor
            if lambdaBLA <= paraMin:
                lambdaBLA = paraMin
        if action == 15:
            action_factor = m.exp(0.5)
            lambdaREC = lambdaREC * action_factor
            if lambdaREC >= paraMax:
                lambdaREC = paraMax
        if action == 17:
            action_factor = m.exp(-0.5)
            lambdaREC = lambdaREC * action_factor
            if lambdaREC <= paraMin:
                lambdaREC = paraMin
        if action == 18:
            action_factor = 1.4
            VPTV = VPTV * action_factor
            if VPTV >= paraMax_VPTV:
                VPTV = paraMax_VPTV
        if action == 20:
            action_factor = 0.6
            VPTV = VPTV * action_factor
            if VPTV <= paraMin:
                VPTV = paraMin
        if action == 21:
            action_factor = 1.25
            VBLA = VBLA * action_factor
            if VBLA >= paraMax_VOAR:
                VBLA = paraMax_VOAR
        if action == 23:
            action_factor = 0.8
            VBLA = VBLA * action_factor
            if VBLA <= paraMin:
                VBLA = paraMin
        if action == 24:
            action_factor = 1.25
            VREC = VREC * action_factor
            if VREC >= paraMax_VOAR:
                VREC = paraMax_VOAR
        if action == 26:
            action_factor = 0.8
            VREC = VREC * action_factor
            if VREC <= paraMin:
                VREC = paraMin


    return tPTV,tBLA, tREC, lambdaPTV,lambdaBLA, lambdaREC, VPTV, VBLA, VREC



def action_new(action,tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC):
    # The upper and lower limits for the Treatment planning parameters



    return tPTV,tBLA, tREC, lambdaPTV,lambdaBLA, lambdaREC, VPTV, VBLA, VREC


    #     if action == 0:
    #         action_factor = 1.1
    #         tPTV = tPTV * action_factor
    #         if tPTV >= paraMax_tPTV:
    #             tPTV = paraMax_tPTV
    #     if action == 2:
    #         action_factor = 0.9
    #         tPTV = tPTV * action_factor
    #         if tPTV <= paraMin_tPTV:
    #             tPTV = paraMin_tPTV
    #     if action == 3:
    #         action_factor = 1.25
    #         tBLA = tBLA * action_factor
    #         if tBLA >= paraMax_tOAR:
    #             tBLA = paraMax_tOAR
    #     if action == 5:
    #         action_factor = 0.8
    #         tBLA = tBLA * action_factor
    #         if tBLA <= paraMin:
    #             tBLA = paraMin
    #     if action == 6:
    #         action_factor = 1.25
    #         tREC = tREC * action_factor
    #         if tREC >= paraMax_tOAR:
    #             tREC = paraMax_tOAR
    #     if action == 8:
    #         action_factor = 0.8
    #         tREC = tREC * action_factor
    #         if tREC <= paraMin:
    #             tREC = paraMin
    #     if action == 9:
    #         action_factor = m.exp(0.5)
    #         lambdaPTV = lambdaPTV * action_factor
    #         if lambdaPTV >= paraMax:
    #             lambdaPTV = paraMax
    #     if action == 11:
    #         action_factor = m.exp(-0.5)
    #         lambdaPTV = lambdaPTV * action_factor
    #         if lambdaPTV <= paraMin:
    #             lambdaPTV = paraMin
    #     if action == 12:
    #         action_factor = m.exp(0.5)
    #         lambdaBLA = lambdaBLA * action_factor
    #         if lambdaBLA >= paraMax:
    #             lambdaBLA = paraMax
    #     if action == 14:
    #         action_factor = m.exp(-0.5)
    #         lambdaBLA = lambdaBLA * action_factor
    #         if lambdaBLA <= paraMin:
    #             lambdaBLA = paraMin
    #     if action == 15:
    #         action_factor = m.exp(0.5)
    #         lambdaREC = lambdaREC * action_factor
    #         if lambdaREC >= paraMax:
    #             lambdaREC = paraMax
    #     if action == 17:
    #         action_factor = m.exp(-0.5)
    #         lambdaREC = lambdaREC * action_factor
    #         if lambdaREC <= paraMin:
    #             lambdaREC = paraMin
    #     if action == 18:
    #         action_factor = 1.4
    #         VPTV = VPTV * action_factor
    #         if VPTV >= paraMax_VPTV:
    #             VPTV = paraMax_VPTV
    #     if action == 20:
    #         action_factor = 0.6
    #         VPTV = VPTV * action_factor
    #         if VPTV <= paraMin:
    #             VPTV = paraMin
    #     if action == 21:
    #         action_factor = 1.25
    #         VBLA = VBLA * action_factor
    #         if VBLA >= paraMax_VOAR:
    #             VBLA = paraMax_VOAR
    #     if action == 23:
    #         action_factor = 0.8
    #         VBLA = VBLA * action_factor
    #         if VBLA <= paraMin:
    #             VBLA = paraMin
    #     if action == 24:
    #         action_factor = 1.25
    #         VREC = VREC * action_factor
    #         if VREC >= paraMax_VOAR:
    #             VREC = paraMax_VOAR
    #     if action == 26:
    #         action_factor = 0.8
    #         VREC = VREC * action_factor
    #         if VREC <= paraMin:
    #             VREC = paraMin
    #
    #
    # return tPTV,tBLA, tREC, lambdaPTV,lambdaBLA, lambdaREC, VPTV, VBLA, VREC
