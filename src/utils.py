"""
Utilities for parsing model responses and handling reasoning tokens
"""

import json
import re
from typing import Optional, Tuple, Dict, Any

class ResponseParser:
    """Parse responses from reasoning models"""
    
    @staticmethod
    def extract_response_from_reasoning(full_response: str) -> Tuple[str, str]:
        """Extract actual response from reasoning model output"""
        if "<think>" in full_response and "</think>" in full_response:
            # Extract reasoning and response
            think_start = full_response.find("<think>")
            think_end = full_response.find("</think>") + len("</think>")
            
            reasoning = full_response[think_start:think_end]
            response = full_response[think_end:].strip()
            
            return response, reasoning
        else:
            # No reasoning tags, entire output is response
            return full_response, ""
    
    @staticmethod
    def extract_json_from_response(response: str) -> Optional[str]:
        """Extract JSON from response, handling malformed thinking tags and formatting"""
        
        # Remove malformed thinking tags more aggressively
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)  # Handle unclosed tags
        response = re.sub(r'.*</think>', '', response, flags=re.DOTALL)  # Handle unopened tags
        
        # Remove "JSON." or "Assistant:" prefixes
        response = re.sub(r'^(JSON\.|Assistant:)\s*', '', response, flags=re.MULTILINE)
        
        # Remove code block markers
        response = re.sub(r'```(?:python|json)?\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find all potential JSON objects
        json_patterns = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        
        # Try each pattern to see if it's valid JSON
        for pattern in json_patterns:
            pattern = pattern.strip()
            try:
                json.loads(pattern)
                return pattern
            except json.JSONDecodeError:
                continue
        
        # If no complete JSON found, try to find partial JSON and complete it
        simple_patterns = re.findall(r'\{"function":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}', response)
        for pattern in simple_patterns:
            try:
                json.loads(pattern)
                return pattern
            except json.JSONDecodeError:
                continue
        
        return None 
    
    @staticmethod
    def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from response"""
        json_str = ResponseParser.extract_json_from_response(response)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        return None
    
    @staticmethod
    def parse_diagnosis(response: str) -> Optional[str]:
        """Extract diagnosis from response"""
        parsed = ResponseParser.parse_json_response(response)
        if parsed and "diagnosis" in parsed:
            return parsed["diagnosis"]
        return None
    
    @staticmethod
    def parse_tool_call(response: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parse tool call from response"""
        parsed = ResponseParser.parse_json_response(response)
        if not parsed:
            return None
        
        # Handle standard format: {"function": "...", "arguments": {...}}
        if "function" in parsed and "arguments" in parsed:
            return parsed["function"], parsed["arguments"]
        
        return None


class ToolArgumentMapper:
    """Map generic argument names to tool-specific names"""
    
    # Define expected argument names for each tool
    TOOL_ARGS = {
        "check_status_page": ["service"],
        "check_deployments": ["service", "hours"],
        "search_slack": ["channel", "keywords"],
        "query_logs": ["service", "time_range", "filters"]
    }
    
    @staticmethod
    def map_arguments(tool_name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map generic argument names (arg1, arg2, etc.) to expected names.
        
        Args:
            tool_name: Name of the tool
            raw_args: Raw arguments from model (may use arg1, arg2, etc.)
            
        Returns:
            Mapped arguments with proper names
        """
        expected_args = ToolArgumentMapper.TOOL_ARGS.get(tool_name, [])
        mapped_args = {}
        
        # First, preserve any already correctly named arguments
        for key, value in raw_args.items():
            if key in expected_args:
                mapped_args[key] = value
        
        # Then map generic args (arg1, arg2, etc.)
        generic_args = [(k, v) for k, v in raw_args.items() if k.startswith("arg")]
        generic_args.sort(key=lambda x: x[0])  # Sort by arg number
        
        for i, (_, value) in enumerate(generic_args):
            if i < len(expected_args) and expected_args[i] not in mapped_args:
                mapped_args[expected_args[i]] = value
        
        # Special handling for numeric values
        if tool_name == "check_deployments" and "hours" in mapped_args:
            try:
                mapped_args["hours"] = int(mapped_args["hours"])
            except (ValueError, TypeError):
                mapped_args["hours"] = 24  # Default
        
        return mapped_args


class TokenCounter:
    """Handle token counting for reasoning models"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text, truncation=True, max_length=4096))
    
    def count_response_tokens(self, full_response: str) -> Tuple[int, int]:
        """
        Count reasoning and response tokens separately.
        
        Returns:
            (reasoning_tokens, response_tokens)
        """
        response, reasoning = ResponseParser.extract_response_from_reasoning(full_response)
        
        reasoning_tokens = self.count_tokens(reasoning) if reasoning else 0
        response_tokens = self.count_tokens(response)
        
        return reasoning_tokens, response_tokens
