"""
Simple Log Entry Structure for RL Training
Just the essential fields we need for incident simulation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import uuid


class LogLevel(Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    """Simple log entry for incident simulation"""
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    
    # Optional fields for different types of issues
    http_status: Optional[int] = None
    response_time_ms: Optional[float] = None
    error_type: Optional[str] = None
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    
    # IDs for correlation
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "service": self.service,
            "message": self.message,
            "http_status": self.http_status,
            "response_time_ms": self.response_time_ms,
            "error_type": self.error_type,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def generate_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:8]}"


def generate_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"
