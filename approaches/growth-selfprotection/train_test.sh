#!/bin/bash

# Define common arguments
training_dataset="--training-dataset ../../data/training-english/"
validation_dataset="--validation-dataset ../../data/validation-english/"

# List of Model trainings (only script names)
scripts=(
    "python3 train_prev-sentences.py"
    "python3 train_VAD.py"
    "python3 train_prev-sentences_VAD.py"
)

# Loop through the scripts
for cmd in "${scripts[@]}"; do
    # Extract the script name (without arguments)
    script_name=$(echo "$cmd" | awk '{print $2}')
    
    # Remove the .py extension and train_ prefix
    base_name=$(basename "$script_name" .py)
    base_name=${base_name#train_}
    
    # Add common arguments and model directory dynamically
    full_cmd="$cmd $training_dataset $validation_dataset --model-directory models/$base_name"
    
    # Print header
    echo "====================================="
    echo "Executing: $full_cmd"
    echo "====================================="
    
    # Run the command and save output
    ($full_cmd | tee "results/$base_name.txt")
    
    # Print footer
    echo "-------------------------------------"
    echo "Finished: $full_cmd"
    echo "-------------------------------------"
    echo # Add an extra blank line for clarity
done
