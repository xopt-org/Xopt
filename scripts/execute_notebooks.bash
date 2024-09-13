#!/bin/bash

NOTEBOOKS=$(find . -type f -name "*.ipynb" -not -path '*/.*')

SKIP="Xparallel"

echo $NOTEBOOKS

for file in $NOTEBOOKS
do
    if [[ "$file" == *"$SKIP"* ]]; then
        echo "Skipping $file"
        continue
    fi

    echo "Executing $file"
    jupyter nbconvert --to notebook --execute $file --inplace
done
