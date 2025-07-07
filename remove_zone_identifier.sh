#/bin/bash

cd "$(git rev-parse --show-toplevel)" || exit 1

echo "Searching for Zone identifier files..."


FILES=$(git ls-files | grep -E ':Zone.Identifier')

if [ -z "$FILES" ]; then
  echo "No Zone identifier files were found"
  exit 0
fi

for file in $FILES; do
  echo "Remove from index: $file"
  git update-index --force-remove "$file"

  if [ -f "$file" ]; then
    echo "Deleting file: $file"
    rm -f "$file"
  fi
done