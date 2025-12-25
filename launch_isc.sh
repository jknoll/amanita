#!/bin/bash
#
# Example launch script for Strong Compute ISC cluster
# Reference: https://docs.strongcompute.com/
#

set -e  # Exit on error

echo "================================================"
echo "Launching Multi-Task BEiT Training on ISC"
echo "================================================"

# Configuration
PROJECT_NAME="fungitastic-multitask"
SCRIPT_NAME="train_isc_multitask.py"

# Data paths (adjust these to match your ISC setup)
DATASET_PATH="/data/uds-fern-absorbed-dugong-251223/full_300px"
TRAIN_CSV="${DATASET_PATH}/metadata/FungiTastic/FungiTastic-Train.csv"
VAL_CSV="${DATASET_PATH}/metadata/FungiTastic/FungiTastic-Val.csv"
TEST_CSV="${DATASET_PATH}/metadata/FungiTastic/FungiTastic-Test.csv"
MAPPINGS_PATH="./taxonomic_mappings.json"
CLASS_WEIGHTS_PATH="./class_weights.pt"

# Model configuration
MODEL_NAME="hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224"

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=16  # Per GPU
LEARNING_RATE=0.0001
NUM_WORKERS=4

# Number of GPUs
NUM_GPUS=4

echo "Configuration:"
echo "  Project: ${PROJECT_NAME}"
echo "  Script: ${SCRIPT_NAME}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo ""

# Example 1: Using torchrun (PyTorch distributed launcher)
echo "Example 1: Using torchrun"
echo "Command:"
cat << 'EOF'
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_isc_multitask.py \
    --train-path ${TRAIN_CSV} \
    --val-path ${VAL_CSV} \
    --test-path ${TEST_CSV} \
    --mappings-path ${MAPPINGS_PATH} \
    --class-weights-path ${CLASS_WEIGHTS_PATH} \
    --model-name ${MODEL_NAME} \
    --pretrained \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --workers ${NUM_WORKERS} \
    --use-class-weights
EOF
echo ""

# Example 2: Using ISC CLI (if available)
echo "Example 2: Using Strong Compute ISC CLI (if installed)"
echo "Command:"
cat << 'EOF'
isc launch \
    --config isc_config.yaml \
    --gpus 4 \
    --name fungitastic-multitask-run1
EOF
echo ""

# Example 3: Direct python execution (single GPU)
echo "Example 3: Single GPU (for testing)"
echo "Command:"
cat << 'EOF'
python train_isc_multitask.py \
    --train-path ${TRAIN_CSV} \
    --val-path ${VAL_CSV} \
    --test-path ${TEST_CSV} \
    --mappings-path ${MAPPINGS_PATH} \
    --class-weights-path ${CLASS_WEIGHTS_PATH} \
    --model-name ${MODEL_NAME} \
    --pretrained \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --workers ${NUM_WORKERS} \
    --use-class-weights \
    --is-master
EOF
echo ""

echo "================================================"
echo "For more information, see:"
echo "  - https://docs.strongcompute.com/"
echo "  - https://github.com/StrongResearch/isc-demos"
echo "================================================"
