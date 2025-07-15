import json
from typing import List, Dict, Any
from datetime import datetime
import re
import verifiers as vf
from datasets import Dataset, load_dataset
from rub import IncidentAnalysisRubric

class IncidentDataStore:
    def __init__(self, incident_data: Dict[str, Any]):
        self.data = incident_data
        if 'state' in incident_data:
            self.state = incident_data['state']
        else:
            self.state = incident_data
    
    def get_status_pages(self) -> List[Dict[str, str]]:
        return self.state.get('status_pages', [])
    
    def get_slack_messages(self) -> List[Dict[str, Any]]:
        return self.state.get('slack', [])
    
    def get_deployments(self) -> List[Dict[str, Any]]:
        return self.state.get('deployments', [])
    
    def get_logs(self) -> List[Dict[str, Any]]:
        return self.state.get('logs', [])

class IncidentAnalysisEnv(vf.ToolEnv):
    def __init__(self, judge_client, dataset=None, **kwargs):
        rubric = IncidentAnalysisRubric(judge_client)
        
        tools = [
            self.check_status_pages,
            self.search_slack_messages, 
            self.check_deployment_failures,
            self.query_logs,
        ]
        
        system_prompt = """
You are an on call engineer. Use the available tools to investigate the incident and determine the root cause.

Available tools:
- check_status_pages(service: str)
- search_slack_messages(query: str)
- check_deployment_failures(hours_back: int)
- query_logs(error_pattern: str)

{tool_descriptions}

You will be given an incident, and you must use the tools to diagnose the cause.

You may make up to 10 tool calls before giving your final answer.

In each turn, respond in the following format:
<think>
[your thoughts here]
</think>
<tool>
{{
  "name": "check_status_pages", # name of the tool to call
  "args": {{
    "service": "stripe" # arguments to pass to the tool
  }}
}}
</tool>

When you have enough information:
<think>
[your thoughts here]
</think>
<answer>
[final answer here]
</answer>
"""
        super().__init__(
            dataset=dataset,
            tools=tools,
            system_prompt=system_prompt,
            rubric=rubric,
            **kwargs
        )

    async def rollout(self,
                      client,
                      model: str,
                      prompt,
                      answer: str,
                      task: str = "default",
                      info: Dict[str, Any] = {},
                      sampling_args: Dict[str, Any] = {},
                      **kwargs) -> tuple:
        """
        Override rollout to set up incident store for each rollout.
        """
        # Extract state from info (this is where the dataset row will be passed)
        if 'state' in info:
            self.incident_store = IncidentDataStore(info)
        else:
            # Create empty state as fallback
            self.incident_store = IncidentDataStore({'state': {}})
        
        return await super().rollout(client, model, prompt, answer, task, info, sampling_args, **kwargs)

    def check_status_pages(self, service: str = "") -> str:
        """Check the status of services from status pages."""
        status_pages = self.incident_store.get_status_pages()
        
        if not status_pages:
            return "No status page data available"
        
        results = []
        for page in status_pages:
            # Handle different formats from the dataset
            if isinstance(page, dict):
                for service_name, status in page.items():
                    if not service or service.lower() in service_name.lower():
                        if isinstance(status, dict):
                            # Handle format: {'service': 'name', 'status': 'healthy', 'message': ''}
                            service_status = status.get('status', 'unknown')
                            message = status.get('message', '')
                            results.append(f"{service_name}: {service_status}" + (f" - {message}" if message else ""))
                        else:
                            # Handle format: {'service_name': 'status_string'}
                            results.append(f"{service_name}: {status}")
            else:
                # Handle list format from dataset
                service_name = page.get('service', 'unknown')
                status = page.get('status', 'unknown')
                message = page.get('message', '')
                if not service or service.lower() in service_name.lower():
                    results.append(f"{service_name}: {status}" + (f" - {message}" if message else ""))
        
        return "\n".join(results) if results else f"No status found for service: {service}"

    def search_slack_messages(self, query: str, limit: int = 10) -> str:
        """Search Slack messages for relevant information."""
        slack_messages = self.incident_store.get_slack_messages()
        
        if not slack_messages:
            return "No Slack messages available"
        
        # Simple text search in message content
        matching_messages = []
        for msg in slack_messages:
            # Handle different message formats from dataset
            content = msg.get('message', msg.get('content', ''))
            if query.lower() in content.lower():
                timestamp = msg.get('datetime', 'Unknown time')
                user = msg.get('user', 'Unknown user')
                channel = msg.get('channel', '')
                channel_info = f"#{channel} " if channel else ""
                matching_messages.append(f"[{timestamp}] {channel_info}{user}: {content}")
        
        # Sort by timestamp and limit results
        matching_messages = matching_messages[:limit]
        
        if not matching_messages:
            return f"No Slack messages found matching: {query}"
        
        return "\n\n".join(matching_messages)

    def check_deployment_failures(self, hours_back: int = 24) -> str:
        """Check for recent deployment failures."""
        deployments = self.incident_store.get_deployments()
        
        if not deployments:
            return "No deployment data available"
        
        # Filter recent deployments
        now = datetime.now()
        recent_deployments = []
        
        for deployment in deployments:
            # Parse deployment datetime
            deploy_time_str = deployment.get('datetime', '')
            try:
                deploy_time = datetime.fromisoformat(deploy_time_str.replace('Z', '+00:00'))
                hours_diff = (now - deploy_time).total_seconds() / 3600
                
                if hours_diff <= hours_back:
                    status = "SUCCESS" if deployment.get('succeeded', False) else "FAILED"
                    service = deployment.get('service', 'Unknown')
                    version = deployment.get('version', 'Unknown')
                    recent_deployments.append(
                        f"{deploy_time_str} - {service} v{version} - {status}"
                    )
            except ValueError:
                continue
        
        if not recent_deployments:
            return f"No deployments found in the last {hours_back} hours"
        
        return "\n".join(recent_deployments)

    def query_logs(self, error_pattern: str = "", limit: int = 20) -> str:
        """Query application logs for errors or patterns."""
        logs = self.incident_store.get_logs()
        
        if not logs:
            return "No log data available"
        
        matching_logs = []
        for log_entry in logs:
            # Handle different log formats from dataset
            log_content = log_entry.get('message', str(log_entry)).lower()
            
            # If no pattern specified, look for common error indicators
            if not error_pattern:
                error_indicators = ['error', 'exception', 'failed', '500', '503', '504']
                if any(indicator in log_content for indicator in error_indicators):
                    matching_logs.append(log_entry)
            else:
                # Search for specific pattern
                if error_pattern.lower() in log_content:
                    matching_logs.append(log_entry)
        
        # Limit results and format
        matching_logs = matching_logs[:limit]
        
        if not matching_logs:
            pattern_msg = f" matching '{error_pattern}'" if error_pattern else " with error indicators"
            return f"No log entries found{pattern_msg}"
        
        formatted_logs = []
        for log in matching_logs:
            timestamp = log.get('datetime', 'Unknown time')
            message = log.get('message', str(log))
            response_time = log.get('response_time', '')
            rt_info = f" (RT: {response_time}ms)" if response_time else ""
            formatted_logs.append(f"[{timestamp}]{rt_info} {message}")
        
        return "\n".join(formatted_logs)


def load_incident_dataset(dataset_name: str = "cdreetz/on-call-agent-grpo-dataset", split: str = "train"):
    """Load the incident dataset from HuggingFace."""
    dataset = load_dataset(dataset_name, split=split)
    
    # Add preprocessing to ensure compatibility
    def preprocess_dataset_row(example):
        # Ensure state is properly parsed if it's a string
        if isinstance(example['state'], str):
            try:
                example['state'] = json.loads(example['state'])
            except json.JSONDecodeError:
                pass
        
        # Ensure consistent format
        return {
            'question': example['question'],
            'answer': example['answer'],
            'state': example['state'],
            'issue_type': example.get('issue_type', 'unknown')
        }
    
    dataset = dataset.map(preprocess_dataset_row)
    return dataset


def create_incident_env(judge_client, dataset_name: str = "cdreetz/on-call-agent-grpo-dataset"):
    """Create an environment for incident analysis using HuggingFace dataset."""
    
    # Load the dataset
    dataset = load_incident_dataset(dataset_name, split="train")
    eval_dataset = load_incident_dataset(dataset_name, split="train")  # You might want a separate eval split
    
    # Create environment
    env = IncidentAnalysisEnv(
        judge_client=judge_client,
        dataset=dataset,
        eval_dataset=eval_dataset,
        max_turns=10  
    )
    
    return env

incident_data = {
    'question': "INCIDENT ALERT - INC-12345\nTitle: Payment Processing Failures\nUsers reporting failed payments on checkout",
    'answer': "recent deployment introduced null pointer exception", 
    'source': "status_page",
    'state': {
        'status_pages': [
            {'stripe': 'degraded', 'aws': 'healthy', 'database': 'healthy'}
        ],
        'slack': [
            {
                'datetime': '2024-01-15T14:30:00Z',
                'user': 'john.doe',
                'content': 'Seeing payment failures spike after the 2pm deployment'
            },
            {
                'datetime': '2024-01-15T14:45:00Z', 
                'user': 'jane.smith',
                'content': 'NullPointerException in payment service logs'
            }
        ],
        'deployments': [
            {
                'datetime': '2024-01-15T14:00:00Z',
                'service': 'payment-service',
                'version': '1.2.3',
                'succeeded': True
            }
        ],
        'logs': [
            {
                'datetime': '2024-01-15T14:35:00Z',
                'message': 'java.lang.NullPointerException at PaymentProcessor.process()',
                'response_time': 5000
            }
        ]
    }
}

# Create the environment
#env = create_incident_env(incident_data)

