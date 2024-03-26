#!/bin/bash

# Initialize variables with default values
BUCKET="slice-prompt-store"
KEY="auto-character-generation.json"
DOWNLOAD_PATH="./downloaded_file.json"
MODEL_ENGINE="text-davinci-002"
API_KEY="sk-addkeyhere"
TEMPERATURE="0.85"
MAX_TOKENS="7000"
USE_LLAMA=true
FAST_RUN=${FAST_RUN:-true} 
CKPT_DIR="./model/llama-2-7b-chat"
TOKENIZER_PATH="./model/tokenizer.model"

# Environment variables
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export PYTHONUNBUFFERED=1

# Parse arguments
while getopts ":b:k:d:m:a:t:x:l:f:c:z:" opt; do
  case $opt in
    b) BUCKET="$OPTARG";;
    k) KEY="$OPTARG";;
    d) DOWNLOAD_PATH="$OPTARG";;
    m) MODEL_ENGINE="$OPTARG";;
    a) API_KEY="$OPTARG";;
    t) TEMPERATURE="$OPTARG";;
    x) MAX_TOKENS="$OPTARG";;
    l) USE_LLAMA=true;;
    f) FAST_RUN="$OPTARG";;
    c) CKPT_DIR="$OPTARG";;
    z) TOKENIZER_PATH="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2;;
  esac
done

# Check if ckpt_dir exists, if not download from S3
if [ ! -d "$CKPT_DIR" ]; then
  aws s3 sync s3://sliced-models/llama-2-7b-chat/ $CKPT_DIR
fi

# Check if tokenizer_path exists, if not download from S3
if [ ! -f "$TOKENIZER_PATH" ]; then
  aws s3 cp s3://sliced-models/tokenizers/llama2_tokenizer.model $TOKENIZER_PATH
fi

# First command
torchrun --nproc_per_node 1 generate.py \
--bucket $BUCKET \
--key $KEY \
--download_path $DOWNLOAD_PATH \
--model_engine $MODEL_ENGINE \
--api_key $API_KEY \
--temperature $TEMPERATURE \
--max_tokens $MAX_TOKENS \
--use_llama \
--fast_run $FAST_RUN \
--ckpt_dir $CKPT_DIR \
--tokenizer_path $TOKENIZER_PATH 

# Check if the first command was successful
if [ $? -eq 0 ]; then
    # Second command
    nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9
    python fine_tuner.py
else
    echo "The first command failed. Not executing the second command."
fi
