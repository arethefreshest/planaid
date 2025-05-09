# Save as: concat_tree_files.sh
output="all_code_text.txt"
> "$output" # empty the output file

# List of file extensions to include - only essential code files
include_ext="py|cs|ts|tsx|js|env|yml|yaml|sh"

# Directories to exclude
exclude_dirs="node_modules|.git|__pycache__|.venv|venv|.mypy_cache|bin|obj|logs|metrics|Debug|dist|build|.vscode|.idea|results|metrics_analysis|public|tests"

# Maximum file size in bytes (500KB)
MAX_FILE_SIZE=512000

# Maximum total output size in bytes (100MB)
MAX_TOTAL_SIZE=104857600
current_size=0

# Use tree to get the file list in order, then process each file
tree -a -I "$exclude_dirs" --prune -fi | \
  grep -E "\.($include_ext)$|Dockerfile|\.gitignore|\.env$" | \
  grep -v "\.(tex|csv)$" | \
  while read -r file; do
    if [ -f "$file" ]; then
      # Skip files larger than MAX_FILE_SIZE
      file_size=$(stat -f%z "$file")
      if [ $file_size -gt $MAX_FILE_SIZE ]; then
        echo "Skipping large file: $file" >&2
        continue
      fi
      
      # Check if adding this file would exceed total size limit
      if [ $((current_size + file_size)) -gt $MAX_TOTAL_SIZE ]; then
        echo "Reached maximum total size limit. Stopping." >&2
        break
      fi
      
      # Get relative path by removing leading ./
      rel_path="${file#./}"
      echo "===== $rel_path =====" >> "$output"
      cat "$file" >> "$output"
      echo -e "\n" >> "$output"
      
      # Update current size
      current_size=$((current_size + file_size))
    fi
  done