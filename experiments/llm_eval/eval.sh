#!/bin/bash

# Step 1: Clone the lm-evaluation-harness repository
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

# Step 3: Run the evaluation
# Example: Evaluate GPT-J-6B model on the hellaswag task using a CUDA-compatible GPU
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --limit 10
