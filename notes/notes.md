## Environment Design: Overview and Rationale

- Goal of the environment is to simulate that of a real on call engineer and the actions they take to resolve the incident
- While also ensuring a level of complexity that allows the model to learn to perform what will be real world actions without over burdening it with noise that slows the learning unnecessarily
- On-call engineer heuristics are built in including checking status pages, checking slack channels, checking recent deployments, and finally querying observability logging and traces

## Reward Function Design: Overview and Rationale

- Goal In Large: Agent 'Efficiently' Handles Issues, where efficiency is both correctness but time to resolution
- MTTR (mean time to resolve) can be translated to MTokTR (mean tokens to resolve)
- MTokTR is a combination of avg correct diagnosis across issues and avg tokens/time to diagnose
- R = a _ accuracy + (1 - a) _ efficiency
- Minimizing Token Cost acts as both a Time Proxy and partial Cost Proxy, while cost of tokens is most likely not a concern
- Heuristic-First Bias of the reward means it implicitly encourages the model to:

  - try common, low-cost heuristics first like downstream dep downage, restarting services
  - then escalate to more complex and time consuming things like in depth log analysis while widening its search

- This reward also aims to avoid likely reward hacking if both acc and efficiency were rewards up front, model would quickly learn to just min tokens in exchange for acc or correct resolution

## Planned Ablation Studies

- Efficiency vs Accuracy Reward Threshold, initially updating the reward at a fixed acc score but want to explore a ramp up in the assigned weight of each task to slowly introduce efficiency instead of at once
- My intuition tells me that a fixed threshold is fine as long as the model does not struggle to meet it, where a ramp may prove to be beneficial if the fixed threshold results in the model overly trading off correctness for the efficiency reward
- Soft Ramp Reward Function = o(k\*(accuracy-t)) , where o() is the sigmoid function and t is the acc threshold
