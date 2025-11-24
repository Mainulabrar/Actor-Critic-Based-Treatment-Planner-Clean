import numpy as np


# Step 1: Open the text file from Notepad and read the content
with open("/data2/mainul/DataAndGraphDQN/DQNActionProbPlotWihoutComma.txt", 'r') as file:
    lines = file.readlines()

# Step 2: Prepare the lines with commas between numbers
with open('output.csv', 'w') as csv_file:
    for line in lines:
        # Strip leading/trailing whitespace and split by spaces
        numbers = line.strip().split()
        
        # Join the numbers with commas and write to the CSV file
        csv_file.write(",".join(numbers) + "\n")

print("File has been converted to CSV with commas.")
