#!/bin/bash

# Define common arguments
training_dataset="--training-dataset ../../data/training-english/"
validation_dataset="--validation-dataset ../../data/validation-english/"

# List of Model trainings (only script names)
scripts=(
    "VAD"
    "VAD --previous-sentences"
    "EmoLex"
    "EmoLex --previous-sentences"
    "EmotionIntensity"
    "EmotionIntensity --previous-sentences"
    "WorryWords"
    "WorryWords --previous-sentences"
    "LIWC"
    "LIWC --previous-sentences"
)

# Loop through the scripts
for cmd in "${scripts[@]}"; do
    # Extract the script name (without arguments)
    script_name=$(echo "$cmd" | awk '{print $1}')
    if [ "$script_name" == "Text" ]; then
        lexicon=""
    else
        lexicon=" --lexicon $script_name"
    fi

    previous_sentences=$(echo "$cmd" | awk '{print $2}')
    if [ ! -z "${previous_sentences}" ]; then
        previous_sentences=" $previous_sentences"
    fi
    
    # Add common arguments and model directory dynamically
    full_cmd="python3 train_all.py $training_dataset $validation_dataset$previous_sentences$lexicon --model-directory models/$script_name"
    
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
