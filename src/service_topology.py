"""
Simple Service Topology
Just enough services to create realistic failure scenarios
"""

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class Service:
    name: str
    depends_on: List[str]  # What this service needs
    used_by: List[str]     # What depends on this service


class ServiceTopology:
    """Simple service dependency graph"""
    
    def __init__(self):
        # Keep it simple - just 6 core services
        self.services = {
            "api-gateway": Service(
                name="api-gateway",
                depends_on=["auth-service", "rate-limiter"],
                used_by=[]
            ),
            "auth-service": Service(
                name="auth-service", 
                depends_on=["database", "redis"],
                used_by=["api-gateway", "user-service"]
            ),
            "user-service": Service(
                name="user-service",
                depends_on=["database", "auth-service"],
                used_by=["api-gateway", "payment-service"]
            ),
            "payment-service": Service(
                name="payment-service",
                depends_on=["database", "user-service", "stripe-api"],
                used_by=["api-gateway"]
            ),
            "database": Service(
                name="database",
                depends_on=[],
                used_by=["auth-service", "user-service", "payment-service"]
            ),
            "redis": Service(
                name="redis",
                depends_on=[],
                used_by=["auth-service", "rate-limiter"]
            ),
            "rate-limiter": Service(
                name="rate-limiter",
                depends_on=["redis"],
                used_by=["api-gateway"]
            ),
            "stripe-api": Service(
                name="stripe-api",
                depends_on=[],  # External service
                used_by=["payment-service"]
            )
        }
    
    def get_downstream_services(self, failed_service: str) -> Set[str]:
        """Get all services affected if this one fails"""
        affected = set()
        to_check = [failed_service]
        
        while to_check:
            current = to_check.pop()
            if current in self.services:
                for dependent in self.services[current].used_by:
                    if dependent not in affected:
                        affected.add(dependent)
                        to_check.append(dependent)
        
        return affected
    
    def get_dependencies(self, service_name: str) -> List[str]:
        """Get what this service depends on"""
        if service_name in self.services:
            return self.services[service_name].depends_on
        return []


# Global instance
TOPOLOGY = ServiceTopology()
