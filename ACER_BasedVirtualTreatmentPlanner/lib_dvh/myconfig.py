""" This is the config"""
import torch
def maxepoch():
	max_epoch = 3
	return max_epoch

def maxstep():
	max_step = 2
	return max_step

def actionnum():#28 default, 18
	actionnum = 18
	return actionnum

def iscuda():
	# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
	device = torch.device("cpu")
	return device


	

