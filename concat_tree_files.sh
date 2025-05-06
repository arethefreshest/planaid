# Save as: concat_tree_files.sh
output="all_code_text.txt"
> "$output" # empty the output file

# List of file extensions to include (edit as needed)
include_ext="py|cs|ts|js|json|md|txt|yml|yaml|env|sh|dockerfile|gitignore|sln|csv"

# Use tree to get the file list in order, then process each file
tree -a -I 'node_modules|.git|__pycache__|.venv|venv|.mypy_cache|bin|obj|logs|metrics|Debug|dist|build|.vscode|.idea' --prune -fi | \
  grep -E "\.($include_ext)$|Dockerfile|\.gitignore|\.env$|README.md|Makefile" | \
  while read -r file; do
    if [ -f "$file" ]; then
      echo "===== $file =====" >> "$output"
      cat "$file" >> "$output"
      echo -e "\n" >> "$output"
    fi
  done