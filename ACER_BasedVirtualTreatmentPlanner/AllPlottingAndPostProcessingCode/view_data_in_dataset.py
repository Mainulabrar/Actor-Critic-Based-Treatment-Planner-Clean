import h5py
import numpy as np

np.set_printoptions(threshold=np.inf)
# Open the .h5 file in read-only mode
with h5py.File('/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/008.h5', 'r') as f:
    # group = f['Dij']
    group = f['oar_ptvs'] 
    # group = f['fluence_grp']
    # subgroup = group['096']
    # Access the data in a specific dataset
    data = group['dose']
    # data = group['fluence']
    #data_index = dataset_name['indptr']
    #data = dataset_name['data']
    dataArray = np.array(data)
    print("Data in 'dataset_name':", dataArray.shape)
    # dosemask = np.reshape(dataArray, (dataArray.shape[0] * dataArray.shape[1] * dataArray.shape[2],), order='C')
    # print(dataArray[1:10])
    # print(dosemask.shape)
    # print(dosemask[np.nonzero(dosemask)].shape)

