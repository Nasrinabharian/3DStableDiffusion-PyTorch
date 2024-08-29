#!/bin/bash

# Navigate to Trash directory
cd ~/.local/share/Trash/files/

# Loop through all files in Trash
for file in *; do
  # Extract the original path from the .trashinfo file
  original_path=$(grep '^Path=' ~/.local/share/Trash/info/"$file".trashinfo | cut -d'=' -f2-)

  # Create the original directory if it doesn't exist
  mkdir -p "$(dirname "$original_path")"

  # Move the file back to the original location
  mv "$file" "$original_path"
done

