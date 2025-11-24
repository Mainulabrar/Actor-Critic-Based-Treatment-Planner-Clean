import numpy as np
import re

patternIteration = r'^=.*iteration.*$'
patternUnperturbed = r'^========================================Patient 0$'
patternPolicy = r'^policy tensor.*$'
# patternPolicy = r'policy tensor(\[\[.*?\]\])\s+grad_fn=<[\w]+>'
# \n(.*\n){3}
patternperturbed = r'^========================================Patient 1$'

PerturbedOrUnperturbed = 0
PolicyStart = []
outputLine = 0
patternOutputLine = 0
inputLineNumber = 10000

with open("/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/FGSM_Attack1001CrossEntropy.txt", 'r') as input_file, open('/data2/mainul/DataAndGraph/extracted_linesFGSM.txt', 'w') as output_file:
    # Read line by line
    for lineNumber, line in enumerate(input_file, start= 1):
        # Search for the pattern
        if re.search(patternIteration, line):
            # Write the matching line to the new file
            print(line)
            output_file.write(line)
            outputLine = outputLine+1
        if re.search(patternUnperturbed, line):
            # Write the matching line to the new file
            print(line)
            output_file.write(line)
            outputLine = outputLine+1
            PerturbedOrUnperturbed = lineNumber
        if re.search(patternPolicy, line):
            # Write the matching line to the new file
            print(line)
            patternOutputLineNew = outputLine + 1
            if patternOutputLineNew != patternOutputLine+1:
            	outputLine = outputLine+1
            	output_file.write(line)
            	inputLineNumber = lineNumber
            	patternOutputLine = patternOutputLineNew
            # patternOutputLine = outputLine
            # match = re.match(r"policy tensor", line)
            # if match.group() not in policyMatch:
            # 	output_file.write(line)
            # 	policyMatch.append(match.group())
            # else:
            # 	policyMatch

        if re.search(patternperturbed, line):
            # Write the matching line to the new file
            print(line)
            output_file.write(line)
            outputLine = outputLine+1

        if inputLineNumber+4>lineNumber>inputLineNumber:
        	print(line)
        	output_file.write(line)


            
            # output_file.write(line)




