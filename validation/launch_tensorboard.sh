#!/bin/bash
# Launch TensorBoard for all experiment logs
#
# This script launches TensorBoard to visualize training metrics
# from all synced experiment artifacts.
#
# Usage: ./launch_tensorboard.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGDIR="/Volumes/Extra FAT/Amanita-Validation"

# Activate venv
if [ -f "$PROJECT_DIR/.amanita/bin/activate" ]; then
    source "$PROJECT_DIR/.amanita/bin/activate"
else
    echo "Error: Virtual environment not found at $PROJECT_DIR/.amanita"
    echo "Create it with: python3 -m venv $PROJECT_DIR/.amanita && pip install tensorboard"
    exit 1
fi

# Check if directory exists
if [ ! -d "$LOGDIR" ]; then
    echo "Error: Log directory not found: $LOGDIR"
    exit 1
fi

# Count experiment directories
NUM_EXPS=$(ls -d "$LOGDIR"/exp-* 2>/dev/null | wc -l | tr -d ' ')
echo "Found $NUM_EXPS experiment directories in $LOGDIR"

# Launch TensorBoard
echo "Launching TensorBoard on http://localhost:6006"
tensorboard --logdir="$LOGDIR" --port=6006
