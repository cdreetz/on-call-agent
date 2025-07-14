"""
Simple Log Templates
Generate common log patterns for different failure scenarios
"""

from datetime import datetime, timedelta
from typing import List
import random

from log_entry import LogEntry, LogLevel, generate_trace_id, generate_request_id


class LogGenerator:
    """Generates realistic log entries for different scenarios"""
    
    def __init__(self):
        self.trace_id = generate_trace_id()
    
    def normal_request(self, service: str, timestamp: datetime) -> LogEntry:
        """Generate a normal successful request log"""
        messages = [
            f"Request processed successfully",
            f"HTTP GET /api/users - 200 OK",
            f"Database query completed",
            f"Cache hit for user data"
        ]
        
        return LogEntry(
            timestamp=timestamp,
            level=LogLevel.INFO,
            service=service,
            message=random.choice(messages),
            http_status=200,
            response_time_ms=random.uniform(50, 200),
            trace_id=self.trace_id,
            request_id=generate_request_id()
        )
    
    def database_timeout(self, service: str, timestamp: datetime) -> LogEntry:
        """Generate database timeout error"""
        return LogEntry(
            timestamp=timestamp,
            level=LogLevel.ERROR,
            service=service,
            message="Database connection timeout after 30s",
            error_type="DATABASE_TIMEOUT",
            response_time_ms=30000,
            trace_id=self.trace_id,
            request_id=generate_request_id()
        )
    
    def memory_leak(self, service: str, timestamp: datetime) -> LogEntry:
        """Generate memory leak indicators"""
        memory_usage = random.uniform(1800, 2048)  # High memory
        return LogEntry(
            timestamp=timestamp,
            level=LogLevel.WARN,
            service=service,
            message=f"High memory usage detected: {memory_usage:.1f}MB",
            memory_mb=memory_usage,
            trace_id=self.trace_id
        )
    
    def high_cpu(self, service: str, timestamp: datetime) -> LogEntry:
        """Generate high CPU usage log"""
        cpu_usage = random.uniform(85, 99)
        return LogEntry(
            timestamp=timestamp,
            level=LogLevel.WARN,
            service=service,
            message=f"CPU usage spike: {cpu_usage:.1f}%",
            cpu_percent=cpu_usage,
            trace_id=self.trace_id
        )
    
    def service_unavailable(self, service: str, timestamp: datetime) -> LogEntry:
        """Generate service unavailable error"""
        return LogEntry(
            timestamp=timestamp,
            level=LogLevel.ERROR,
            service=service,
            message="Service unavailable - connection refused",
            http_status=503,
            error_type="SERVICE_UNAVAILABLE",
            trace_id=self.trace_id,
            request_id=generate_request_id()
        )
    
    def external_api_error(self, service: str, timestamp: datetime) -> LogEntry:
        """Generate external API failure"""
        return LogEntry(
            timestamp=timestamp,
            level=LogLevel.ERROR,
            service=service,
            message="External API call failed - stripe payment processing error",
            http_status=502,
            error_type="EXTERNAL_API_ERROR",
            response_time_ms=random.uniform(5000, 10000),
            trace_id=self.trace_id,
            request_id=generate_request_id()
        )
    
    def slow_query(self, service: str, timestamp: datetime) -> LogEntry:
        """Generate slow database query"""
        query_time = random.uniform(2000, 8000)
        return LogEntry(
            timestamp=timestamp,
            level=LogLevel.WARN,
            service=service,
            message=f"Slow query detected: {query_time:.0f}ms",
            response_time_ms=query_time,
            trace_id=self.trace_id
        )
