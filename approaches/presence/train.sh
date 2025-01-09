#!/bin/bash

# Define common arguments
training_dataset="--training-dataset ../../data/training-english/"
validation_dataset="--validation-dataset ../../data/validation-english/"

# List of Model trainings (only script names and optional arguments)
scripts=(
    "Text --previous-sentences"
    "VAD"
    "VAD --previous-sentences"
    "EmoLex"
    "EmoLex --previous-sentences"
    "EmotionIntensity"
    "EmotionIntensity --previous-sentences"
    "WorryWords"
)

# Function to parse arguments and construct the command
construct_command() {
    local cmd="$1"
    local script_name
    local lexicon=""
    local previous_sentences=""
    local linguistic_features=""
    local prev_sent=""
    local ling_feat=""

    # Extract the script name and optional arguments
    script_name=$(echo "$cmd" | awk '{print $1}')
    if [ "$script_name" != "Text" ]; then
        lexicon=" --lexicon $script_name"
    fi

    # Check for additional flags in the command
    if echo "$cmd" | grep -q -- "--previous-sentences"; then
        previous_sentences=" --previous-sentences"
        prev_sent="-prev-sentences"
    fi
    if echo "$cmd" | grep -q -- "--linguistic-features"; then
        linguistic_features=" --linguistic-features"
        ling_feat="-ling-feat"
    fi

    # Construct and return the full command and result file prefix
    local fullname="${script_name}${prev_sent}${ling_feat}"
    local model_directory="models/${fullname}"
    local result_file="results/${fullname}.txt"
    echo "python3 main.py $training_dataset $validation_dataset$previous_sentences$linguistic_features$lexicon --model-directory $model_directory --model-name $fullname"
    echo "$result_file"
}

# Loop through the scripts
for cmd in "${scripts[@]}"; do
    # Construct the command and result file
    full_cmd=$(construct_command "$cmd" | head -n 1)
    result_file=$(construct_command "$cmd" | tail -n 1)

    # Print header
    echo "====================================="
    echo "Executing: $full_cmd"
    echo "====================================="

    # Run the command and save output
    eval "$full_cmd" | tee "$result_file"

    # Print footer
    echo "-------------------------------------"
    echo "Finished: $full_cmd"
    echo "-------------------------------------"
    echo # Add an extra blank line for clarity
done