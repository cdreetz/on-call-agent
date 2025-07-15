#!/bin/bash

# Build and run GRPO training with Docker
# This script handles the multi-GPU setup automatically

set -e

echo "Starting GRPO training setup..."

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# Check if WANDB is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set"
    exit 1
fi

# Check if WANDB PROJECT is set
if [ -z "$WANDB_PROJECT" ]; then
    echo "Error: WANDB_PROJECT environment variable is not set"
    exit 1
fi

# Create necessary directories
mkdir -p models outputs cache

# No need to create zero3.yaml - using the one from verifiers repo

# Build the Docker image
echo "Building Docker image..."
docker build -t grpo-training .

# Start inference server on GPUs 0,1
echo "Starting VLLM inference server..."
docker run -d \
    --name vllm-server \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -v $(pwd):/workspace \
    -v $(pwd)/models:/workspace/models \
    -v $(pwd)/cache:/root/.cache \
    -p 8000:8000 \
    --gpus '"device=0,1"' \
    grpo-training \
    bash -c "cd /workspace && vllm serve --model Qwen/Qwen3-4B --host 0.0.0.0 --port 8000"

echo "Waiting for VLLM server to start..."
sleep 30

# Check if VLLM server is running
if ! docker ps | grep -q vllm-server; then
    echo "Error: VLLM server failed to start"
    echo "VLLM server logs:"
    docker logs vllm-server
    exit 1
fi

echo "VLLM server is running successfully"

# Run training on GPUs 2,3
echo "Starting GRPO training..."
docker run \
    --name grpo-trainer \
    -e CUDA_VISIBLE_DEVICES=2,3 \
    -e NVIDIA_VISIBLE_DEVICES=2,3 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e HF_TOKEN="$HF_TOKEN" \
    -v $(pwd):/workspace \
    -v $(pwd)/models:/workspace/models \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/cache:/root/.cache \
    --gpus '"device=2,3"' \
    --network container:vllm-server \
    grpo-training \
    bash -c "cd /workspace && CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config-file configs/zero3.yaml main.py"

# Clean up
echo "Cleaning up containers..."
docker stop vllm-server grpo-trainer || true
docker rm vllm-server grpo-trainer || true

echo "Training completed!"