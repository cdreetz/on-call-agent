"""
Prompts and prompt templates for the on-call agent
"""

from typing import Dict, List, Any


class PromptTemplate:
    """Base class for prompt templates"""
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables"""
        return self.template.format(**kwargs)


class OnCallPrompts:
    """Collection of prompts for the on-call investigation agent"""
    
    @staticmethod
    def get_system_prompt(tool_descriptions: str) -> str:
        """Main system prompt that defines the agent's role and capabilities"""
        return f"""You are an expert on-call engineer responsible for investigating and diagnosing production incidents quickly and efficiently.

Your goal is to identify the root cause of incidents using the minimum number of investigation steps. You have access to several tools that have different costs and information value:

INVESTIGATION STRATEGY:
1. Check external service status pages FIRST - these are quick and solve ~60% of incidents
2. Check recent deployments SECOND - many issues are caused by recent changes  
3. Search communication channels for known issues
4. Query logs only when necessary - these are expensive but provide detailed information

AVAILABLE TOOLS:
{tool_descriptions}

RESPONSE FORMAT:
To use a tool, respond with valid JSON:
{{"function": "tool_name", "arguments": {{"param": "value"}}}}

To provide your final diagnosis, respond with:
{{"diagnosis": "Clear description of the root cause and what service is affected"}}

IMPORTANT GUIDELINES:
- Always start with the cheapest, highest-signal tools first
- Status page checks often reveal external dependency issues immediately
- Recent deployments are a common cause of new issues
- Only dive into detailed log analysis if simpler methods don't reveal the cause
- Provide a clear, actionable diagnosis that identifies both the service and root cause
- Be efficient - aim to diagnose issues in as few steps as possible

Remember: The goal is to minimize time-to-diagnosis while maintaining accuracy."""

    @staticmethod
    def get_investigation_start_prompt(alert: str) -> str:
        """Prompt to start investigation with an alert"""
        return f"""INCIDENT ALERT: {alert}

You have been paged to investigate this incident. Please begin your investigation using the most efficient approach. 

Start with checking external dependencies and recent changes before diving into detailed log analysis."""

    @staticmethod
    def get_tool_result_prompt(tool_name: str, result: str) -> str:
        """Prompt to provide tool execution results back to the agent"""
        return f"""TOOL RESULT from {tool_name}:
{result}

Based on this information, either:
1. Use another tool if you need more information
2. Provide your diagnosis if you have enough information to determine the root cause

Continue your investigation or provide your final diagnosis."""

    @staticmethod
    def get_diagnosis_request_prompt() -> str:
        """Prompt to request final diagnosis when max steps reached"""
        return """You have reached the maximum number of investigation steps. Based on all the information you have gathered, please provide your final diagnosis of the incident.

Respond with: {"diagnosis": "Your final diagnosis of the root cause"}"""

    @staticmethod
    def get_conversation_context_prompt(conversation_history: List[Dict[str, str]]) -> str:
        """Build conversation context from history"""
        context = ""
        for entry in conversation_history:
            role = entry["role"].upper()
            content = entry["content"]
            context += f"{role}: {content}\n\n"
        return context

    @staticmethod
    def get_training_system_prompt() -> str:
        """System prompt specifically for training scenarios"""
        return """You are training to become an expert on-call engineer. Your goal is to learn efficient incident investigation patterns.

You will be evaluated on:
1. ACCURACY: Correctly identifying the root cause of incidents
2. EFFICIENCY: Using the minimum number of investigation steps and tokens

The most successful on-call engineers follow this hierarchy:
1. Check status pages for external service outages (solves ~60% of incidents quickly)
2. Check recent deployments for code-related issues (solves ~25% of remaining incidents)  
3. Search team communication for known issues
4. Analyze detailed logs only when necessary (expensive but comprehensive)

Learn to recognize patterns:
- Payment failures often = Stripe/external payment processor issues
- Database timeouts often = AWS RDS or database connectivity problems
- Memory/CPU alerts often = Resource exhaustion from recent deployments
- Widespread errors often = Load balancer or infrastructure issues

Your training will reward investigations that reach correct diagnoses efficiently."""

    @staticmethod
    def get_few_shot_examples() -> List[Dict[str, Any]]:
        """Provide few-shot examples of efficient investigations"""
        return [
            {
                "alert": "High error rate in payment-service - 15% 5xx responses",
                "efficient_approach": [
                    {"function": "check_status_page", "arguments": {"service": "stripe"}},
                    {"diagnosis": "Stripe API outage causing payment service failures"}
                ],
                "inefficient_approach": [
                    {"function": "query_logs", "arguments": {"service": "payment-service", "time_range": "30m"}},
                    {"function": "query_logs", "arguments": {"service": "user-service", "time_range": "30m"}},
                    {"function": "check_status_page", "arguments": {"service": "stripe"}},
                    {"diagnosis": "Stripe API outage causing payment service failures"}
                ],
                "lesson": "Check external dependencies first - same diagnosis, 3x fewer tokens"
            },
            {
                "alert": "Database connection timeouts in user-service",
                "efficient_approach": [
                    {"function": "check_status_page", "arguments": {"service": "aws"}},
                    {"function": "check_deployments", "arguments": {"service": "user-service", "hours": 6}},
                    {"diagnosis": "Recent deployment introduced database connection leak"}
                ],
                "inefficient_approach": [
                    {"function": "query_logs", "arguments": {"service": "user-service", "time_range": "1h", "filters": "error"}},
                    {"function": "query_logs", "arguments": {"service": "database", "time_range": "1h"}},
                    {"function": "check_deployments", "arguments": {"service": "user-service", "hours": 6}},
                    {"diagnosis": "Recent deployment introduced database connection leak"}
                ],
                "lesson": "Check recent deployments early - they're a common cause of new issues"
            }
        ]

    @staticmethod
    def get_error_handling_prompt(error_message: str) -> str:
        """Prompt for handling tool execution errors"""
        return f"""There was an error executing your last tool call:
ERROR: {error_message}

Please try again with a corrected tool call, or use a different tool to continue your investigation."""

    @staticmethod
    def get_investigation_guidance_prompt(step_count: int, max_steps: int) -> str:
        """Provide guidance based on investigation progress"""
        remaining_steps = max_steps - step_count
        
        if remaining_steps <= 2:
            return f"""You have {remaining_steps} investigation steps remaining. Focus on the most likely causes and prepare to provide your diagnosis soon."""
        elif step_count == 1:
            return """Good start! Consider what this result tells you about the incident. Do you need more information, or can you already identify the likely cause?"""
        elif step_count >= 4:
            return """You've gathered substantial information. Consider if you have enough evidence to make a diagnosis, or if there's one specific piece of information still needed."""
        else:
            return """Continue your investigation based on what you've learned so far."""

    @staticmethod
    def format_investigation_prompt(alert: str, tool_descriptions: str, few_shot: bool = False) -> str:
        """Format complete investigation prompt with all components"""
        system_prompt = OnCallPrompts.get_system_prompt(tool_descriptions)
        start_prompt = OnCallPrompts.get_investigation_start_prompt(alert)
        
        full_prompt = f"{system_prompt}\n\n{start_prompt}"
        
        if few_shot:
            examples = OnCallPrompts.get_few_shot_examples()
            examples_text = "\n\nEXAMPLE EFFICIENT INVESTIGATIONS:\n"
            for i, example in enumerate(examples[:2], 1):
                examples_text += f"\nExample {i}: {example['alert']}\n"
                examples_text += f"Efficient approach: {len(example['efficient_approach'])} steps\n"
                examples_text += f"Lesson: {example['lesson']}\n"
            full_prompt += examples_text
        
        return full_prompt
