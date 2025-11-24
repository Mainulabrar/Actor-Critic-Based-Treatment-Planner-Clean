import numpy as np

data_result_path = '/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'

# patient_list = ['001', '008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '093', '095', '097', '098']
# patient_list = ['008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '093', '095', '097', '098']
# patient_list = ['008']
patient_list = ['001']

epoch = 120499

PTVSizes = []
BladderSizes = []
RecSizes = []

BlaOverlapPercentage = []
RecOverlapPercentage = []

for id1 in patient_list:
    bladderLabel = np.load(data_result_path+str(id1)+'bladderLabel' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)
    rectumLabel= np.load(data_result_path+str(id1)+'rectumLabel' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)
    PTVLabel = np.load(data_result_path+str(id1)+'PTVLabel' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)
    PTVSizes.append(np.size(np.nonzero(PTVLabel)))
    BladderSizes.append(np.size(np.nonzero(bladderLabel)))
    RecSizes.append(np.size(np.nonzero(rectumLabel)))

    Blaoverlap = np.loadtxt(data_result_path+str(id1)+'PTVBlaOverlap')
    Recoverlap = np.loadtxt(data_result_path+str(id1)+'PTVRecOverlap')

    BlaOverlapPercentage.append(np.size(np.nonzero(Blaoverlap))/np.size(np.nonzero(PTVLabel)))
    RecOverlapPercentage.append(np.size(np.nonzero(Recoverlap))/np.size(np.nonzero(PTVLabel)))


print('PTV Size Mean', np.mean(PTVSizes))
print('Bladder Size Mean', np.mean(BladderSizes))
print('Rec Size Mean', np.mean(RecSizes))
print('BlaOverlap Mean', np.mean(BlaOverlapPercentage)*100)
print('RecOverlap Mean', np.mean(RecOverlapPercentage)*100)


print('PTV Size Spread', np.std(PTVSizes))
print('Bladder Size Spread', np.std(BladderSizes))
print('Rec Size Spread', np.std(RecSizes))
print('BlaOverlap Spread', np.std(np.array(BlaOverlapPercentage)*100))
print('RecOverlap Spread', np.std(np.array(RecOverlapPercentage)*100))