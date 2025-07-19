import os
from openai import OpenAI

import verifiers as vf
from obs_env import IncidentAnalysisEnv, load_incident_dataset

"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml 
"""

judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

dataset = load_incident_dataset("cdreetz/on-call-agent-grpo-dataset", split="train")

incident_env = IncidentAnalysisEnv(
    judge_client=judge_client,
    dataset=dataset,
    max_turns=10
)

model_name = "Qwen/Qwen3-4B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "on-call-agent-grpo_" + model_name.split("/")[-1].lower()

training_args=vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch=12
training_args.num_generations=24
training_args.gradient_accumulation_steps=12
training_args.num_iterations=1
training_args.num_train_epochs=5
training_args.max_prompt_length=1024
training_args.max_completion_length=4096
training_args.max_steps=500
training_args.save_steps=50


trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=incident_env,
    args=training_args
)

trainer.train()
