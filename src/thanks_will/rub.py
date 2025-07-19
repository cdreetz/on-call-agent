from verifiers.rubrics import JudgeRubric, Rubric
from verifiers.parsers import Parser

class IncidentAnalysisRubric(Rubric):
    def __init__(self, judge_client, judge_model="gpt-4o-mini"):
        super().__init__()

        judge_prompt = """
You are evaluating an incident analysis response.

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

    def get_format_reward_func(self):
        """Return a function that checks format compliance."""
        def format_reward_func(completion, **kwargs) -> float:
            """Reward for proper format (having <answer> tags)."""
            if isinstance(completion, str):
                if '<answer>' in completion and '</answer>' in completion:
                    return 1.0
                else:
                    return 0.0
            elif isinstance(completion, list):
                # Check last assistant message for answer format
                for message in reversed(completion):
                    if message.get('role') == 'assistant':
                        content = message.get('content', '')
                        if '<answer>' in content and '</answer>' in content:
                            return 1.0
                        break
                return 0.0
            return 0.0
        
        return format_reward_func

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
        elif isinstance(completion, list):
            tool_calls = 0
            for message in completion:
                if message.get('role') == 'assistant':
                    content = message.get('content', '')
                    tool_calls += content.count('<tool>')
            return tool_calls
        return 0
