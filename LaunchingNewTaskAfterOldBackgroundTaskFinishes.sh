#!/bin/bash

# File where the PID is stored
PID_FILE="script.pid"
echo "The beginning time is: $(date +"%H:%M:%S")"

# Check if the file exists
if [ ! -f $PID_FILE ]; then
    echo "PID file not found!"
    exit 1
fi

# Read the PID from the file
SCRIPT_PID=$(cat $PID_FILE)
echo $SCRIPT_PID

# Function to check if a process is running
is_process_running() {
    ps -p $SCRIPT_PID > /dev/null 2>&1
    return $?
}

# Loop until the process is no longer running, using a longer sleep interval
echo "Waiting for process $SCRIPT_PID to finish..."
while is_process_running; do
    sleep 5  # Sleep for 5 seconds before checking again to reduce CPU usage
done
echo "process_is_finished"
echo "The executing time is: $(date +"%H:%M:%S")"

python file1.py &



