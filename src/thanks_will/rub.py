from verifiers.rubrics import JudgeRubric, Rubric

class IncidentAnalysisRubric(Rubric):
    def __init__(self, judge_client, judge_model="gpt-4o-mini"):
        super().__init__()

        judge_prompt = """
You are evaluating an indident analysis response.

Ground Truth Diagnosis:
{answer}

Agent Analysis:
{response}

Question: Does the agent analysis correctly identify the root cause described in the ground truth?

Respond with only "Yes" or "No".
"""

        judge_rubric = JudgeRubric(
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompts=judge_prompt,
            parser=self.parser
        )

        self.add_reward_func(judge_rubric.judge_reward_func, weight=1.0)
        self.add_reward_func(self.efficiency_reward_func, weight=1.0) # ramp up this weight?

    def efficiency_reward_func(self, completion, **kwargs) -> float:
        """
        Reward for minimizing tool calls.
        Max reward 1.0, subtract 0.1 per call.
        """
        tool_call_count = self.count_tool_calls(completion)
        efficiency_reward = max(0.0, 1.0 - (tool_call_count * 0.1))

        return efficiency_reward

    def count_tool_calls(self, completion) -> int:
        if isinstance(completion, str):
            return completion.count('<tool>')
        elif isinstance(completion, list)
            tool_calls = 0
            for message in completion:
                if message.get('role') == 'assistant':
                    content = message.get('content', '')
                    tool_calls += content.count('<tool>')
            return tool_calls
        return 0
