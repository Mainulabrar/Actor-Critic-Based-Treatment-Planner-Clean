import re
import numpy as np

# Test:120499,8 ,8.28225048809266
ConvergenceArray = []

for i in range(0, 250010, 500):
	if 127999>i>124999:
		continue
	if i == 0:
		print(0)
		ConvergenceArray.append(6.0)
	else:
		print(i-1)
		AllScoreArray = []
		pattern = rf'^Test:{i-1},.* ,.*$'
		with open('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/extractedValidationLines.txt', 'r') as input_file, open('extracted_linesfinalScore.txt', 'w') as output_file:
			for line in input_file:
				match = re.search(rf'^Test:{i-1},(\d+(\.\d+)?) ,.*$', line)
				if match:
					# print(match.group(1))
					AllScoreArray.append(match.group(1))
		# print(AllScoreArray)
		AllScoreArray = list(map(float, AllScoreArray))
		# index = [i-1 for i in range(1, len(AllScoreArray)) if AllScoreArray[i] < AllScoreArray[i-1]]
		PatientSevenArray = AllScoreArray[1:21]
		nineScore = 9.0
		if nineScore in PatientSevenArray:
			PatientSevenResult = 9.0
		else:
			PatientSevenResult = PatientSevenArray[-1]
		# print(index[0])
		PatientTwelveResult = AllScoreArray[-1]
		AverageResult = (PatientSevenResult + PatientTwelveResult)/2.0
		# print(PatientSevenResult)
		# print(PatientTwelveResult)
		print('Average', AverageResult)
		ConvergenceArray.append(AverageResult)

print('c',ConvergenceArray)
ConvergenceArray = np.array(ConvergenceArray)
np.save("/data2/mainul/DataAndGraph/ConvergenceArray", ConvergenceArray)