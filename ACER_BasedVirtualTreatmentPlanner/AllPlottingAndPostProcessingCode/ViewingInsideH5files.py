import numpy as np
import scipy as sp
import glob
import re
import h5py
import h5sparse
from scipy.sparse import csr_matrix, hstack, vstack, csc_matrix
import matplotlib.pyplot as plt
from collections import Counter

file = h5py.File("/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/Prostate_TORTS/test_structure_mask_TORTS01.h5", 'r')

print(file['oar_ptvs']['Rectum'])
bladdertemp = file['oar_ptvs']['Rectum']
blad_ind = np.nonzero(bladdertemp)
print(np.shape(blad_ind))