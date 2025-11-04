#!/bin/bash

NOTEBOOKS=$(find . -type f -name "*.ipynb" -not -path '*/.*')
SKIP="Xparallel"
FAILED=0

echo $NOTEBOOKS

for file in $NOTEBOOKS
do
    if [[ "$file" == *"$SKIP"* ]]; then
        echo "Skipping $file"
        continue
    fi

    echo "Executing $file"
    jupyter nbconvert --to notebook --execute $file --inplace --ExecutePreprocessor.raise_on_error=True
    if [[ $? -ne 0 ]]; then
        echo "FAILED: $file"
        FAILED=1
    fi
done

exit $FAILED
