from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit")

tool_structure = """
{
    'tool_name': 'check_status',
    'arg1': 'service name to check status of. ex: payment-service',
    'arg2': 'date to check in mm-yy format'
}
"""

today = "july 13, 2025"
prompt = "can you check to see if the payment service is down today?"

if tokenizer.chat_template is not None:
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Please respond with valid JSON for function calling. The expected output structure is {tool_structure}\n\nTodays date: {today}"},
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=512)

