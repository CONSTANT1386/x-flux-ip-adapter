MAX_RETRIES=10
RETRY_COUNT=0

TRAIN_COMMAND=(
    "accelerate" "launch" "x-flux/train_flux_deepspeed_IPAdapter.py"
    "--config=x-flux/train_configs/test_ip_adapter.yaml"
)

run_training() {
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "Starting training attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES"
        
        "${TRAIN_COMMAND[@]}"
        if [ $? -eq 0 ]; then
            echo "Training completed successfully."
            return 0
        else
            echo "Training failed. Retrying..."
            RETRY_COUNT=$((RETRY_COUNT + 1))
            sleep 10
        fi
    done

    echo "Maximum retry attempts reached. Training failed."
    return 1
}

if run_training; then
    echo "Training completed. Shutting down..."
    shutdown -h now
else
    echo "Training failed after maximum retries. Shutting down."
    shutdown -h now
fi
