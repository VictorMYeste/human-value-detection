#!/usr/bin/env bash

# Usage:
#   ./combine_files.sh <folder_with_python_files> <path_to_main_py> <output_markdown>
#
# Example:
#   ./combine_files.sh ./my_python_folder ./main.py merged_code.md

FOLDER=$1
MAIN_PY=$2
OUTPUT_MD=$3

# Safety checks
if [ -z "$FOLDER" ] || [ -z "$MAIN_PY" ] || [ -z "$OUTPUT_MD" ]; then
  echo "Usage: $0 <folder_with_python_files> <path_to_main_py> <output_markdown>"
  exit 1
fi

# Create or overwrite the output file
echo "# Collected Python Files" > "$OUTPUT_MD"

# Iterate over all .py files in the specified folder
for FILE in "$FOLDER"/*.py; do
  # Skip if no .py files are found
  if [ ! -f "$FILE" ]; then
    continue
  fi

  FILENAME=$(basename "$FILE")
  echo "## $FILENAME" >> "$OUTPUT_MD"
  echo '```python' >> "$OUTPUT_MD"
  cat "$FILE" >> "$OUTPUT_MD"
  echo -e '\n```\n-----------' >> "$OUTPUT_MD"
done

# Append the main.py contents
FILENAME_MAIN=$(basename "$MAIN_PY")
echo "## $FILENAME_MAIN" >> "$OUTPUT_MD"
  echo -e '\n' >> "$OUTPUT_MD"
echo '```python' >> "$OUTPUT_MD"
cat "$MAIN_PY" >> "$OUTPUT_MD"
echo -e '\n```\n-----------' >> "$OUTPUT_MD"

echo "All Python files and main.py have been merged into $OUTPUT_MD."
