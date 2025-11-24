import numpy as np
import h5py
import scipy.sparse as sp

# data_result_path = '/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_result_path = '/data2/mainul/results_CORS1PaperDiffBeam7/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
data_result_path = '/data2/mainul/results_CORS1PaperDiffBeam/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
data_path = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/'


patient_list = [ '008',  '010',     '020',    '027',   '031',   '039', '042',  '046',  '061',  '070',    '084',  '087',  '092',  '095',  '098']

# patient_list = ['008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '093', '095', '097', '098']
# patient_list = ['001']

epoch = 120499

# import h5py
import scipy.sparse

shapeRow = []
shapeColumn = []
# Open the HDF5 file
for i in patient_list:
	filename = data_path+i+'.hdf5'
	with h5py.File(filename, "r") as f:
	    # Assume the dataset is stored as 'data', 'indices', 'indptr', 'shape'
	    # data = f['Dij']['000']["data"][:]
	    # indices = f['Dij']['000']["indices"][:]
	    indptr = f['Dij']['000']["indptr"][:]
	    # shape = tuple(f["shape"][:])  # Convert to tuple

	    # Reconstruct the sparse matrix (assuming CSR format)
	    # sparse_matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

	dosematrix = np.load(data_result_path+str(i)+'doseMatrix' + str(epoch)  + 'step' + str(0)+'.npy', allow_pickle = True)
	# Get the shape
	if isinstance(dosematrix, np.ndarray) and hasattr(dosematrix.item(), "shape"):
	    sparse_matrix = dosematrix.item()  # Extract stored sparse matrix
	    print("if Shape of sparse matrix:", sparse_matrix.shape)
	elif isinstance(dosematrix, sp.spmatrix):  # If it loads directly as a sparse matrix
	    print("Shape of sparse matrix:", dosematrix.shape)
	else:
	    print("Unknown data format.")

	# print("Sparse matrix shape:", dosematrix.shape)
	shapeRow.append(sparse_matrix.shape[1])
	# shapeColumn.append(indptr.shape[1])

print(np.min(shapeRow))
print(np.max(shapeRow))
print('Mean', np.mean(shapeRow))
print('std', np.std(shapeRow))