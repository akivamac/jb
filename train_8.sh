#!/bin/bash
# Train 8 experts that were never trained at seq_len1024
# Skips coding and cot (reverted from overfitting, await RT3 data merge)
cd /Users/dev/github-projects/joe-brain

EXPERTS="emotion fish greeting horse knowledge python reptiles tree"
MAX_PARALLEL=2
RUNNING=0
PIDS=""

for expert in $EXPERTS; do
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        # Wait for any child to finish
        wait -n 2>/dev/null || sleep 60
        RUNNING=$((RUNNING - 1))
    done
    
    echo "$(date) Starting $expert"
    nohup python3 training/train_expert.py \
        --name "$expert" --steps 4000 \
        --resume --push 100 \
        --log 25 --sample 25 \
        >> "data/experts/$expert/training.log" 2>&1 &
    
    RUNNING=$((RUNNING + 1))
    sleep 5
done

echo "$(date) All 8 experts launched, waiting..."
wait
echo "$(date) All done."
