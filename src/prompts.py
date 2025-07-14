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

AVAILABLE TOOLS:
{tool_descriptions}

RESPONSE FORMAT:
To use a tool, respond with valid JSON:
{{"function": "tool_name", "arguments": {{"param": "value"}}}}

To provide your final diagnosis, respond with:
{{"diagnosis": "Clear description of the root cause and what service is affected"}}

Important: Respond with only valid JSON, do not provide any comentary or code blocks."""

    @staticmethod
    def get_investigation_start_prompt(alert: str) -> str:
        """Prompt to start investigation with an alert"""
        return f"""INCIDENT ALERT: {alert}

Investigate this incident and determine the root cause."""

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
        return """Provide your final diagnosis based on the investigation."""

    @staticmethod
    def get_error_handling_prompt(error_message: str) -> str:
        """Prompt to handle errors"""
        return f"""Error: {error_message}"""

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
    def format_investigation_prompt(alert: str, tool_descriptions: str) -> str:
        """Format complete investigation prompt with all components"""
        system_prompt = OnCallPrompts.get_system_prompt(tool_descriptions)
        start_prompt = OnCallPrompts.get_investigation_start_prompt(alert)

        return f"{system_prompt}\n\n{start_prompt}"
        
