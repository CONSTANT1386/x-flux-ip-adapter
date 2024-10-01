import subprocess
import time
import sys
import os

MAX_RETRIES = 10
RETRY_COUNT = 0

TRAIN_COMMAND = [
    "accelerate", "launch", "/root/x-flux/train_flux_deepspeed_IPAdapter.py",
    "--config=/root/x-flux/train_configs/test_ip_adapter.yaml",
]

def run_training():
    global RETRY_COUNT
    while RETRY_COUNT < MAX_RETRIES:
        print(f"Starting training attempt {RETRY_COUNT + 1} of {MAX_RETRIES}")
        
        try:
            subprocess.run(TRAIN_COMMAND, check=True)
            print("Training completed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Training failed. Retrying...")
            RETRY_COUNT += 1
            time.sleep(10) 
    
    print("Maximum retry attempts reached. Training failed.")
    return False

if __name__ == "__main__":
    if run_training():
        print("Training completed. Shutting down...")
        os.system("shutdown -h now")
    else:
        print("Training failed after maximum retries. Shutting down.")
        os.system("shutdown -h now")