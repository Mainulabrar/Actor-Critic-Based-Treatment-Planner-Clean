import re
import numpy as np

# patternPolicy = r'policy tensor(\[\[.*?\]\])\s+grad_fn=<[\w]+>'

with open('/data2/mainul/DataAndGraph/extracted_linesFGSM.txt', 'r') as file:
	text = file.read()

print(text)

pattern = r'policy tensor\(\[\[(.*?)\]\],\s+grad_fn=<[\w]+>\)'
AllPolicyArray = np.zeros((100,18))

# Use re.findall to extract all occurrences of the pattern
matches = re.findall(pattern, text, re.DOTALL)
# for i, match in enumerate(matches):
# 	print(np.array(match))
# 	AllPolicyArray[i,:] = match

# print(matches)
# Split each match into arrays of numbers and convert them into Python lists
arrays = [list(map(float, match.split(','))) for match in matches]
arrays = np.array(arrays)

print(arrays)
print(arrays.shape)
np.save('/data2/mainul/DataAndGraph/PolicyFGSMACER', arrays)
# # 	# Output the extracted arrays
# for i, array in enumerate(arrays):
#     print(f"Array {i+1}: {array}")
# print('AllPolicyArray',AllPolicyArray)