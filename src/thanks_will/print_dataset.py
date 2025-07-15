import json
from datasets import load_from_disk

# Load the saved dataset
dataset = load_from_disk("randomized_incidents")

print("="*80)
print("GENERATED INCIDENT EXAMPLES:")
print("="*80)

for i, example in enumerate(dataset, 1):
    print(f"\n--- Example {i} ---")
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer']}")
    print(f"Issue Type: {example['issue_type']}")
    print(f"State:")
    print(json.dumps(example['state'], indent=2)) 