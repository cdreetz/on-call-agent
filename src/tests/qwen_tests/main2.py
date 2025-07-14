from mlx_lm import load, generate

#model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit")
model, tokenizer = load("Qwen/Qwen3-4B")

prompt = """
You are an expert on-call engineer responsible for investigating and diagnosing production incidents quickly and efficiently.

Your goal is to identify the root cause of incidents using the minimum number of investigation steps. You have access to several tools that have different costs and information value:

AVAILABLE TOOLS:
check_deployments: Check recent deployment history for code changes that might cause issues.

Arguments:
  - service (Optional): Specific service to check deployments for (optional, checks all if not specified) (default: None)
  - hours (int): Number of hours to look back in deployment history (default: 24) (default: 24)

Returns: (str)

check_status_page: Check external service status page for outages and degradations.

Arguments:
  - service (str): Service name to check (aws, stripe, cloudflare, github, datadog)

Returns: (str)

query_logs: Query service logs for detailed error analysis and debugging information.

Arguments:
  - service (str): Service name to query logs for (payment-service, user-service, database, etc.)
  - time_range (str): Time range to query (10m, 30m, 1h, 2h) (default: 10m)
  - filters (Optional): Log level filter to apply (error, warn, info) (default: None)

Returns: (str)

search_slack: Search team Slack messages for incident reports and outage announcements.

Arguments:
  - channel (Optional): Slack channel to search in (oncall-alerts, customer-support, backend-eng, etc.) (default: None)
  - keywords (Optional): Keywords to search for in message content (space-separated) (default: None)

Returns: (str)

RESPONSE FORMAT:
To use a tool, respond with valid JSON:
{"function": "tool_name", "arguments": {"param": "value"}}

To provide your final diagnosis, respond with:
{"diagnosis": "Clear description of the root cause and what service is affected"}

Important: Respond with only valid JSON, do not provide any comentary or code blocks.

INCIDENT ALERT: High error rate alert: stripe-api - 15% errors in last 5 minutes

Investigate this incident and determine the root cause.
"""


if tokenizer.chat_template is not None:
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Please respond with valid JSON for function calling. {prompt}"},
        {"role": "user", "content": "Diagnose the incident."}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=512)

