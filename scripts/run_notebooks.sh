#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_notebooks.sh <group> <total_groups> <summary_title>
#
# - group:        1-based group index (e.g. 1, 2, 3, 4)
# - total_groups: total number of groups (e.g. 4)
# - summary_title: text label for the group (e.g. "Group 1")
#
# This script:
#   - finds all .ipynb notebooks (excluding hidden and *Xparallel*)
#   - executes the slice of notebooks belonging to this group in parallel
#   - uses jitter + retries to handle transient kernel/port issues
#   - writes per-notebook timing/failed status files into TMPDIR
#   - prints a per-group summary to stdout (for GitHub summaries)

GROUP=${1:?"group index (1-based) is required"}
TOTAL_GROUPS=${2:?"total_groups is required"}
SUMMARY_LABEL=${3:-"Group $GROUP"}

set +e  # don't exit on first error, we track failures per notebook

NOTEBOOKS=$(find . -type f -name "*.ipynb" -not -path '*/.*' -not -path '*Xparallel*')
NOTEBOOK_ARRAY=($NOTEBOOKS)

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT
SCRIPT_START=$(date +%s)

execute_notebook() {
    local file=$1
    local tmpdir=$2

    # Add random jitter (0-2 seconds) to desynchronize parallel starts
    sleep "$(awk -v seed="$RANDOM" 'BEGIN{srand(seed); printf "%.2f", rand() * 2}')"

    local START_TIME END_TIME DURATION
    local MAX_RETRIES=3
    local RETRY_COUNT=0
    local SUCCESS=false
    # Per-notebook log will be used for final error reporting
    local FINAL_LOG

    START_TIME=$(date +%s)

    while [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; do
        local LOG_FILE="/tmp/nbconvert_output_$$.$RETRY_COUNT.log"
        FINAL_LOG="$LOG_FILE"
        if uv run jupyter nbconvert --to notebook --execute "$file" --inplace \
              --ExecutePreprocessor.timeout=600 \
              >"$LOG_FILE" 2>&1; then
            SUCCESS=true
            break
        fi

        # Check if error is due to transient kernel/port issues
        if [ -f "$LOG_FILE" ] && grep -qE "Address already in use|ZMQError|Kernel died before replying" "$LOG_FILE"; then
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; then
                local WAIT_TIME=$((2 ** RETRY_COUNT))  # 2, 4, 8 seconds
                echo "Port or kernel issue for $file, retrying in ${WAIT_TIME}s (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)..." >&2
                sleep "$WAIT_TIME"
                continue
            else
                echo "Failed to execute $file after $MAX_RETRIES attempts (kernel/port-related failure)" >&2
            fi
        fi

        # Non-port-collision error or retries exhausted: do not retry
        break
    done

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    local status_file="$tmpdir/$(basename "$file").status"
    local log_copy="$tmpdir/$(basename "$file").log"

    if [ "$SUCCESS" = true ]; then
        echo "$DURATION" >"$status_file"
        echo "✓ Successfully executed $file (${DURATION}s)"
        # For successful notebooks, log is not strictly necessary; if present, we can drop it
        [ -n "${FINAL_LOG:-}" ] && rm -f "$FINAL_LOG"
    else
        echo "failed:$DURATION" >"$status_file"
        echo "✗ Failed to execute $file (${DURATION}s)" >&2
        # Preserve the last attempt's log alongside the status so it can be shown in summaries
        if [ -n "${FINAL_LOG:-}" ] && [ -f "$FINAL_LOG" ]; then
            cp "$FINAL_LOG" "$log_copy"
            echo "--- Begin nbconvert log for $file ---" >&2
            cat "$FINAL_LOG" >&2
            echo "--- End nbconvert log for $file ---" >&2
        fi
    fi
}

export -f execute_notebook

# Execute notebooks in this group in parallel (all notebooks in this group)
for ((i=GROUP-1; i<${#NOTEBOOK_ARRAY[@]}; i+=TOTAL_GROUPS)); do
    execute_notebook "${NOTEBOOK_ARRAY[$i]}" "$TMPDIR" &
done
wait

# Collect results and echo a markdown summary to stdout
TOTAL=0
SUCCESS=0
FAILED=0
TOTAL_TIME=0

echo "## $SUMMARY_LABEL Results"
echo ""
for ((i=GROUP-1; i<${#NOTEBOOK_ARRAY[@]}; i+=TOTAL_GROUPS)); do
    file="${NOTEBOOK_ARRAY[$i]}"
    TOTAL=$((TOTAL + 1))
    status_file="$TMPDIR/$(basename "$file").status"
    log_file="$TMPDIR/$(basename "$file").log"
    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [[ "$status" == failed:* ]]; then
            FAILED=$((FAILED + 1))
            duration=${status#failed:}
            TOTAL_TIME=$((TOTAL_TIME + duration))
            echo "- ❌ \`$file\` (${duration}s)"
            # If we have a preserved log for this notebook, include a collapsible block
            if [ -f "$log_file" ]; then
                echo "  <details>"
                echo "  <summary>View error log</summary>"
                echo ""
                echo '\```text'
                cat "$log_file"
                echo '\```'
                echo "  </details>"
            fi
        else
            SUCCESS=$((SUCCESS + 1))
            TOTAL_TIME=$((TOTAL_TIME + status))
            echo "- ✅ \`$file\` (${status}s)"
        fi
    fi
done

SCRIPT_END=$(date +%s)
SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))

echo ""
echo "**Summary for $SUMMARY_LABEL:**"
echo "- Total: $TOTAL"
echo "- Success: $SUCCESS"
echo "- Failed: $FAILED"
echo "- Total execution time: ${TOTAL_TIME}s"
echo "- Wall clock time: ${SCRIPT_DURATION}s"

# Exit code: non-zero if any notebook failed
if [ "$FAILED" -gt 0 ]; then
  exit 1
else
  exit 0
fi
