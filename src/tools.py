"""
Tool interface for LLM agent with automatic schema inference
"""

import inspect
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from environment import InvestigationEnvironment
from log_entry import LogLevel


def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()
    
    # Extract examples if present
    examples = []
    return_description = ""
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
        elif part.startswith("Returns:"):
            return_description = part.split("\n")[1].strip() if len(part.split("\n")) > 1 else ""
    
    return_type = str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any")
    
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        # Skip 'self' parameter
        if name == 'self':
            continue
            
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name)+1:].strip()
        
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default
    
    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": f"{return_description} ({return_type})" if return_description else f"({return_type})",
        "examples": examples
    }


def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        
        if schema['args']:
            desc.append("\nArguments:")
            for arg_name, arg_info in schema['args'].items():
                default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
                desc.append(f"  - {arg_name} ({arg_info['type']}): {arg_info['description']}{default}")
        
        if schema['examples']:
            desc.append("\nExamples:")
            for example in schema['examples']:
                desc.append(f"  {example}")
        
        if schema['returns'] and schema['returns'] != "(any)":
            desc.append(f"\nReturns: {schema['returns']}")
        
        descriptions.append("\n".join(desc))
    
    return "\n\n".join(descriptions)


class InvestigationTools:
    """Investigation tools with automatic schema inference"""
    
    def __init__(self, environment: InvestigationEnvironment):
        self.environment = environment
        
        # Get all tool methods (those not starting with _)
        self.tool_methods = [
            getattr(self, method_name) 
            for method_name in dir(self)
            if not method_name.startswith('_') and callable(getattr(self, method_name))
            and method_name not in ['get_tool_schemas', 'get_tool_descriptions', 'execute_tool']
        ]
        
        # Generate schemas automatically
        self.tool_schemas = [infer_schema_from_function(method) for method in self.tool_methods]
        self.tool_registry = {schema['name']: method for schema, method in zip(self.tool_schemas, self.tool_methods)}
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get list of all tool schemas"""
        return self.tool_schemas
    
    def get_tool_descriptions(self) -> str:
        """Get formatted description of all available tools"""
        return format_tool_descriptions(self.tool_schemas)
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool with given arguments"""
        if tool_name not in self.tool_registry:
            available_tools = list(self.tool_registry.keys())
            return f"Error: Unknown tool '{tool_name}'. Available tools: {available_tools}"
        
        try:
            tool_function = self.tool_registry[tool_name]
            return tool_function(**arguments)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def check_status_page(self, service: str) -> str:
        """Check external service status page for outages and degradations.
        
        Args:
            service: Service name to check (aws, stripe, cloudflare, github, datadog)
            
        Returns:
            Current status information including operational state and any incident messages
            
        Examples:
            check_status_page("stripe")
            check_status_page("aws")
        """
        service = service.lower().strip()
        
        if service not in self.environment.status_pages:
            available_services = list(self.environment.status_pages.keys())
            return f"Service '{service}' not found. Available services: {available_services}"
        
        page = self.environment.status_pages[service]
        
        #status_emoji = {
        #    "operational": "✅",
        #    "degraded": "⚠️", 
        #    "down": "🚨"
        #}
        #
        #emoji = status_emoji.get(page.status, "❓")
        
        return f"""{page.service.title()} Status: {page.status.upper()}
Last Updated: {datetime.now().strftime('%H:%M')}
Message: {page.message}"""
    
    def check_deployments(self, service: Optional[str] = None, hours: int = 24) -> str:
        """Check recent deployment history for code changes that might cause issues.
        
        Args:
            service: Specific service to check deployments for (optional, checks all if not specified)
            hours: Number of hours to look back in deployment history (default: 24)
            
        Returns:
            List of recent deployments with timestamps, versions, and deployers
            
        Examples:
            check_deployments("payment-service", 12)
            check_deployments(hours=6)
            check_deployments("user-service")
        """
        if hours < 1 or hours > 168:  # 1 hour to 1 week
            hours = 24
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_deployments = [
            d for d in self.environment.deployments 
            if d.deployed_at > cutoff_time
        ]
        
        # Filter by service if specified
        if service:
            service = service.lower().strip()
            recent_deployments = [
                d for d in recent_deployments 
                if d.service.lower() == service
            ]
        
        if not recent_deployments:
            service_text = f" for {service}" if service else ""
            return f"No deployments found{service_text} in the last {hours} hours"
        
        # Sort by most recent first
        recent_deployments.sort(key=lambda x: x.deployed_at, reverse=True)
        
        result = f"Recent deployments in last {hours} hours:\n"
        for deployment in recent_deployments[:5]:  # Show max 5
            time_ago = datetime.now() - deployment.deployed_at
            hours_ago = int(time_ago.total_seconds() / 3600)
            minutes_ago = int((time_ago.total_seconds() % 3600) / 60)
            
            if hours_ago > 0:
                time_str = f"{hours_ago}h{minutes_ago}m ago"
            else:
                time_str = f"{minutes_ago}m ago"
            
            result += f"📦 {deployment.service} {deployment.version}\n"
            result += f"   Deployed by: {deployment.deployed_by}\n"
            result += f"   Time: {time_str} ({deployment.deployed_at.strftime('%m/%d %H:%M')})\n\n"
        
        return result.strip()
    
    def search_slack(self, channel: Optional[str] = None, keywords: Optional[str] = None) -> str:
        """Search team Slack messages for incident reports and outage announcements.
        
        Args:
            channel: Slack channel to search in (oncall-alerts, customer-support, backend-eng, etc.)
            keywords: Keywords to search for in message content (space-separated)
            
        Returns:
            Recent relevant messages with timestamps and user information
            
        Examples:
            search_slack("oncall-alerts", "stripe payment")
            search_slack("customer-support", "error failure")
            search_slack(keywords="database timeout")
            search_slack("backend-eng")
        """
        messages = self.environment.slack_messages
        
        # Filter by channel if specified
        if channel:
            channel = channel.lower().strip()
            if channel.startswith('#'):
                channel = channel[1:]
            messages = [m for m in messages if m.channel.lower() == channel]
        
        # Filter by keywords if specified
        if keywords:
            keyword_list = [kw.lower().strip() for kw in keywords.split()]
            messages = [
                m for m in messages 
                if any(kw in m.message.lower() for kw in keyword_list)
            ]
        
        if not messages:
            search_desc = ""
            if channel and keywords:
                search_desc = f" in #{channel} with keywords '{keywords}'"
            elif channel:
                search_desc = f" in #{channel}"
            elif keywords:
                search_desc = f" with keywords '{keywords}'"
            
            return f"No messages found{search_desc}"
        
        # Sort by most recent first
        messages.sort(key=lambda x: x.timestamp, reverse=True)
        
        result = "Recent Slack messages:\n\n"
        for i, message in enumerate(messages[:3]):  # Show max 3 messages
            time_ago = datetime.now() - message.timestamp
            minutes_ago = int(time_ago.total_seconds() / 60)
            
            if minutes_ago < 60:
                time_str = f"{minutes_ago}m ago"
            else:
                hours_ago = int(minutes_ago / 60)
                time_str = f"{hours_ago}h ago"
            
            result += f"💬 #{message.channel} - {time_str}\n"
            result += f"   @{message.user}: {message.message}\n"
            if i < len(messages[:3]) - 1:
                result += "\n"
        
        return result
    
    def query_logs(self, service: str, time_range: str = "10m", filters: Optional[str] = None) -> str:
        """Query service logs for detailed error analysis and debugging information.
        
        Args:
            service: Service name to query logs for (payment-service, user-service, database, etc.)
            time_range: Time range to query (10m, 30m, 1h, 2h)
            filters: Log level filter to apply (error, warn, info)
            
        Returns:
            Log summary with error counts and sample error messages
            
        Examples:
            query_logs("payment-service", "30m", "error")
            query_logs("database", "1h")
            query_logs("user-service", "10m", "warn")
        """
        service = service.lower().strip()
        
        # Parse time range
        time_range = time_range.lower().strip()
        if "10m" in time_range or "10 min" in time_range:
            cutoff = datetime.now() - timedelta(minutes=10)
            range_desc = "10 minutes"
        elif "30m" in time_range or "30 min" in time_range:
            cutoff = datetime.now() - timedelta(minutes=30)
            range_desc = "30 minutes"
        elif "1h" in time_range or "hour" in time_range:
            cutoff = datetime.now() - timedelta(hours=1)
            range_desc = "1 hour"
        elif "2h" in time_range:
            cutoff = datetime.now() - timedelta(hours=2)
            range_desc = "2 hours"
        else:
            cutoff = datetime.now() - timedelta(minutes=10)
            range_desc = "10 minutes"
        
        # Get logs for the service
        service_logs = [
            log for log in self.environment.logs 
            if log.service.lower() == service and log.timestamp > cutoff
        ]
        
        if not service_logs:
            return f"No logs found for service '{service}' in the last {range_desc}"
        
        # Apply filters
        if filters:
            filters = filters.lower().strip()
            if "error" in filters:
                service_logs = [log for log in service_logs if log.level == LogLevel.ERROR]
            elif "warn" in filters or "warning" in filters:
                service_logs = [log for log in service_logs if log.level == LogLevel.WARN]
            elif "info" in filters:
                service_logs = [log for log in service_logs if log.level == LogLevel.INFO]
        
        # Count by log level
        error_count = len([log for log in service_logs if log.level == LogLevel.ERROR])
        warn_count = len([log for log in service_logs if log.level == LogLevel.WARN])
        info_count = len([log for log in service_logs if log.level == LogLevel.INFO])
        
        # Build result
        result = f"📊 Log Summary for {service} (last {range_desc}):\n"
        result += f"Total logs: {len(service_logs)}\n"
        result += f"🚨 Errors: {error_count} | ⚠️  Warnings: {warn_count} | ℹ️  Info: {info_count}\n\n"
        
        # Show sample error logs if any exist
        error_logs = [log for log in service_logs if log.level == LogLevel.ERROR]
        if error_logs:
            result += "Recent Error Logs:\n"
            for i, log in enumerate(error_logs[:3]):  # Show max 3 error logs
                time_str = log.timestamp.strftime('%H:%M:%S')
                result += f"🚨 [{time_str}] {log.message}\n"
                if log.error_type:
                    result += f"   Error Type: {log.error_type}\n"
                if log.response_time_ms and log.response_time_ms > 1000:
                    result += f"   Response Time: {log.response_time_ms:.0f}ms\n"
                if i < min(len(error_logs), 3) - 1:
                    result += "\n"
        
        # Show sample warning logs if no errors but warnings exist
        elif warn_count > 0:
            warn_logs = [log for log in service_logs if log.level == LogLevel.WARN]
            result += "Recent Warning Logs:\n"
            for i, log in enumerate(warn_logs[:2]):
                time_str = log.timestamp.strftime('%H:%M:%S')
                result += f"⚠️  [{time_str}] {log.message}\n"
                if i < min(len(warn_logs), 2) - 1:
                    result += "\n"
        
        return result


class ToolRegistry(InvestigationTools):
    """Legacy alias for InvestigationTools"""
    pass
