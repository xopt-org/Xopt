#!/bin/bash

NOTEBOOKS=$(find . -type f -name "*.ipynb" -not -path '*/.*')

SKIP="Xparallel"

echo $NOTEBOOKS

# Track execution statistics
TOTAL=0
SKIPPED=0
FAILED=0
SUCCESS=0
TOTAL_TIME=0

# Arrays to store notebook lists and timing info
FAILED_NOTEBOOKS=()
SKIPPED_NOTEBOOKS=()
SUCCESS_NOTEBOOKS=()
TIMING_INFO=()

# Record overall start time
SCRIPT_START=$(date +%s)

for file in $NOTEBOOKS
do
    TOTAL=$((TOTAL + 1))
    
    if [[ "$file" == *"$SKIP"* ]]; then
        echo "Skipping $file"
        SKIPPED=$((SKIPPED + 1))
        SKIPPED_NOTEBOOKS+=("$file")
        continue
    fi

    echo "Executing $file"
    
    # Record start time for this notebook
    START_TIME=$(date +%s)
    
    # Execute notebook and capture exit code
    if jupyter nbconvert --to notebook --execute $file --inplace; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "✓ Successfully executed $file (${DURATION}s)"
        SUCCESS=$((SUCCESS + 1))
        SUCCESS_NOTEBOOKS+=("$file")
        TIMING_INFO+=("$file: ${DURATION}s")
        TOTAL_TIME=$((TOTAL_TIME + DURATION))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "✗ Failed to execute $file (${DURATION}s)" >&2
        FAILED=$((FAILED + 1))
        FAILED_NOTEBOOKS+=("$file")
        TIMING_INFO+=("$file: ${DURATION}s (FAILED)")
        TOTAL_TIME=$((TOTAL_TIME + DURATION))
        # Continue to next notebook instead of exiting
    fi
done

# Calculate total script duration
SCRIPT_END=$(date +%s)
SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))

echo ""
echo "========================================="
echo "Notebook execution summary:"
echo "  Total notebooks found: $TOTAL"
echo "  Successfully executed: $SUCCESS"
echo "  Skipped: $SKIPPED"
echo "  Failed: $FAILED"
echo "  Total execution time: ${TOTAL_TIME}s"
echo "  Total script time: ${SCRIPT_DURATION}s"
echo "========================================="

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "❌ Failed notebooks:"
    for notebook in "${FAILED_NOTEBOOKS[@]}"; do
        echo "  - $notebook"
    done
fi

if [ $SKIPPED -gt 0 ]; then
    echo ""
    echo "⏭️ Skipped notebooks:"
    for notebook in "${SKIPPED_NOTEBOOKS[@]}"; do
        echo "  - $notebook"
    done
fi

if [ $SUCCESS -gt 0 ]; then
    echo ""
    echo "✅ Successfully executed notebooks:"
    for notebook in "${SUCCESS_NOTEBOOKS[@]}"; do
        echo "  - $notebook"
    done
fi

echo ""
echo "⏱️ Execution timing details:"
for timing in "${TIMING_INFO[@]}"; do
    echo "  $timing"
done

echo ""
if [ $FAILED -gt 0 ]; then
    echo "❌ Some notebooks failed to execute! Check the failed list above."
    exit 1
else
    echo "✅ All notebooks executed successfully!"
    exit 0
fi
