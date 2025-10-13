#!/bin/bash
# Example training run with good hyperparameters

echo "Starting OpenScope RL Training"
echo "This will train for a few hours. Monitor progress with Ctrl+C to stop."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run training with wandb logging
python train.py \
    --config config/training_config.yaml \
    --wandb \
    --device cuda

echo ""
echo "Training complete! Check the checkpoints/ and logs/ directories."

