#!/bin/bash
# Sequential training and cleanup script with full output logging

# Create folder for logs
mkdir -p ablation_study

# Downsample rates to iterate over
RATES=(0.005 0.008 0.1 0.2 0.5)

# Master log for all runs
MASTER_LOG="ablation_study/all_runs_log.txt"

echo "===== Starting all runs at $(date) =====" | tee -a "$MASTER_LOG"

for rate in "${RATES[@]}"; do
    echo "=== Starting run with DOWNSAMPLE_RATE_FOR_CHUNKING=$rate at $(date) ===" | tee -a "$MASTER_LOG"

    # Set the runtime log path for this run
    RUNTIME_LOG="ablation_study/runtime_${rate}.txt"

    # Run Python with unbuffered output (-u) and pass environment variable
    {
        echo ">>> BEGIN LOG for rate=$rate at $(date)"
        DOWNSAMPLE_RATE_FOR_CHUNKING=$rate \
        RUNTIME_LOG="$RUNTIME_LOG" \
        python -u test.py --config config/s3dis/s3dis_stratified_transformer.yaml
        echo ">>> END LOG for rate=$rate at $(date)"
    } 2>&1 | tee "ablation_study/run_${rate}.log" | tee -a "$MASTER_LOG"

    # Clean up saved_when_testing folder
    rm -rf runs/s3dis_stratified_transformer/saved_when_testing/*

    echo "=== Finished run with DOWNSAMPLE_RATE_FOR_CHUNKING=$rate at $(date) ===" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
done

echo "===== All runs completed at $(date) =====" | tee -a "$MASTER_LOG"
