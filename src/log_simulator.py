"""
Log Simulator with Error Probabilities
Generates realistic log streams with configurable failure scenarios
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
import random
from dataclasses import dataclass
from enum import Enum

from log_entry import LogEntry, LogLevel
from service_topology import TOPOLOGY
from simple_log_templates import LogGenerator


class FailureType(Enum):
    """Types of failures we can simulate"""
    DATABASE_TIMEOUT = "database_timeout"
    MEMORY_LEAK = "memory_leak"
    HIGH_CPU = "high_cpu"
    EXTERNAL_API_ERROR = "external_api_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    SLOW_QUERY = "slow_query"


@dataclass
class FailureScenario:
    """Configuration for a failure scenario"""
    failure_type: FailureType
    primary_service: str  # Service where failure originates
    probability: float    # Probability of this failure occurring
    duration_minutes: int # How long the failure lasts
    cascade_delay_seconds: int = 30  # Delay before cascading to other services


class LogSimulator:
    """Generates realistic log streams with failure scenarios"""
    
    def __init__(self):
        self.generator = LogGenerator()
        self.failure_scenarios = self._default_failure_scenarios()
        self.normal_log_rate = 10  # logs per minute per service during normal operation
        self.error_log_multiplier = 5  # how much more logs during errors
    
    def _default_failure_scenarios(self) -> List[FailureScenario]:
        """Default realistic failure scenarios"""
        return [
            # Database issues (most common in real systems)
            FailureScenario(
                failure_type=FailureType.DATABASE_TIMEOUT,
                primary_service="database",
                probability=0.15,  # 15% chance
                duration_minutes=10,
                cascade_delay_seconds=30
            ),
            FailureScenario(
                failure_type=FailureType.SLOW_QUERY,
                primary_service="database", 
                probability=0.25,  # 25% chance
                duration_minutes=5,
                cascade_delay_seconds=60
            ),
            
            # External API failures
            FailureScenario(
                failure_type=FailureType.EXTERNAL_API_ERROR,
                primary_service="stripe-api",
                probability=0.20,  # 20% chance
                duration_minutes=15,
                cascade_delay_seconds=10
            ),
            
            # Memory/CPU issues
            FailureScenario(
                failure_type=FailureType.MEMORY_LEAK,
                primary_service="payment-service",
                probability=0.10,  # 10% chance
                duration_minutes=20,
                cascade_delay_seconds=120
            ),
            FailureScenario(
                failure_type=FailureType.HIGH_CPU,
                primary_service="auth-service",
                probability=0.12,  # 12% chance
                duration_minutes=8,
                cascade_delay_seconds=90
            ),
            
            # Service unavailable
            FailureScenario(
                failure_type=FailureType.SERVICE_UNAVAILABLE,
                primary_service="redis",
                probability=0.08,  # 8% chance
                duration_minutes=12,
                cascade_delay_seconds=20
            ),
        ]
    
    def generate_normal_logs(self, 
                           start_time: datetime, 
                           duration_minutes: int,
                           services: List[str] = None) -> List[LogEntry]:
        """Generate normal operational logs"""
        if services is None:
            services = list(TOPOLOGY.services.keys())
        
        logs = []
        current_time = start_time
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while current_time < end_time:
            for service in services:
                # Generate normal logs at specified rate
                if random.random() < (self.normal_log_rate / 60):  # Convert to per-second probability
                    logs.append(self.generator.normal_request(service, current_time))
            
            current_time += timedelta(seconds=1)
        
        return logs
    
    def generate_failure_scenario(self, 
                                scenario: FailureScenario,
                                start_time: datetime) -> List[LogEntry]:
        """Generate logs for a specific failure scenario"""
        logs = []
        
        # Get the failure generator method
        failure_generators = {
            FailureType.DATABASE_TIMEOUT: self.generator.database_timeout,
            FailureType.MEMORY_LEAK: self.generator.memory_leak,
            FailureType.HIGH_CPU: self.generator.high_cpu,
            FailureType.EXTERNAL_API_ERROR: self.generator.external_api_error,
            FailureType.SERVICE_UNAVAILABLE: self.generator.service_unavailable,
            FailureType.SLOW_QUERY: self.generator.slow_query,
        }
        
        failure_generator = failure_generators[scenario.failure_type]
        current_time = start_time
        end_time = start_time + timedelta(minutes=scenario.duration_minutes)
        
        # Generate initial failure logs from primary service
        while current_time < end_time:
            # More frequent error logs during failure
            if random.random() < (self.error_log_multiplier * self.normal_log_rate / 60):
                logs.append(failure_generator(scenario.primary_service, current_time))
            
            current_time += timedelta(seconds=1)
        
        # Generate cascading failure logs
        cascade_start = start_time + timedelta(seconds=scenario.cascade_delay_seconds)
        affected_services = TOPOLOGY.get_downstream_services(scenario.primary_service)
        
        for service in affected_services:
            service_current_time = cascade_start
            # Cascading failures are shorter duration
            service_end_time = min(end_time, cascade_start + timedelta(minutes=scenario.duration_minutes // 2))
            
            while service_current_time < service_end_time:
                if random.random() < (self.error_log_multiplier * self.normal_log_rate / 60):
                    # Cascading failures often manifest as timeouts or service unavailable
                    if scenario.failure_type in [FailureType.DATABASE_TIMEOUT, FailureType.SLOW_QUERY]:
                        logs.append(self.generator.database_timeout(service, service_current_time))
                    else:
                        logs.append(self.generator.service_unavailable(service, service_current_time))
                
                service_current_time += timedelta(seconds=1)
        
        return logs
    
    def simulate_time_period(self, 
                           start_time: datetime,
                           duration_minutes: int,
                           include_failures: bool = True) -> Dict[str, any]:
        """Simulate a complete time period with normal and failure logs"""
        all_logs = []
        incidents = []
        
        # Generate baseline normal logs
        normal_logs = self.generate_normal_logs(start_time, duration_minutes)
        all_logs.extend(normal_logs)
        
        if include_failures:
            # Randomly select and generate failure scenarios
            for scenario in self.failure_scenarios:
                if random.random() < scenario.probability:
                    # Random time during the period for failure to start
                    failure_start_offset = random.randint(0, max(1, duration_minutes - scenario.duration_minutes))
                    failure_start = start_time + timedelta(minutes=failure_start_offset)
                    
                    failure_logs = self.generate_failure_scenario(scenario, failure_start)
                    all_logs.extend(failure_logs)
                    
                    incidents.append({
                        "failure_type": scenario.failure_type.value,
                        "primary_service": scenario.primary_service,
                        "start_time": failure_start,
                        "duration_minutes": scenario.duration_minutes,
                        "affected_services": list(TOPOLOGY.get_downstream_services(scenario.primary_service))
                    })
        
        # Sort all logs by timestamp
        all_logs.sort(key=lambda x: x.timestamp)
        
        return {
            "logs": all_logs,
            "incidents": incidents,
            "period": {
                "start_time": start_time,
                "duration_minutes": duration_minutes,
                "total_logs": len(all_logs)
            }
        }
    
    def generate_training_scenario(self, 
                                 scenario_type: str = "random") -> Dict[str, any]:
        """Generate a specific training scenario"""
        start_time = datetime.now()
        
        if scenario_type == "random":
            # Random 1-2 hour period with potential failures
            duration = random.randint(60, 120)
            return self.simulate_time_period(start_time, duration, include_failures=True)
        
        elif scenario_type == "database_outage":
            # Specific database failure scenario
            duration = 30
            scenario = FailureScenario(
                failure_type=FailureType.DATABASE_TIMEOUT,
                primary_service="database",
                probability=1.0,  # Guaranteed failure
                duration_minutes=15,
                cascade_delay_seconds=30
            )
            
            # Generate normal logs
            logs = self.generate_normal_logs(start_time, duration)
            
            # Add specific failure
            failure_start = start_time + timedelta(minutes=5)
            failure_logs = self.generate_failure_scenario(scenario, failure_start)
            logs.extend(failure_logs)
            logs.sort(key=lambda x: x.timestamp)
            
            return {
                "logs": logs,
                "incidents": [{
                    "failure_type": scenario.failure_type.value,
                    "primary_service": scenario.primary_service,
                    "start_time": failure_start,
                    "duration_minutes": scenario.duration_minutes,
                    "affected_services": list(TOPOLOGY.get_downstream_services("database")),
                    "root_cause": "Database connection pool exhausted"
                }],
                "period": {"start_time": start_time, "duration_minutes": duration, "total_logs": len(logs)}
            }
        
        elif scenario_type == "external_api_failure":
            # Stripe API failure scenario
            duration = 45
            scenario = FailureScenario(
                failure_type=FailureType.EXTERNAL_API_ERROR,
                primary_service="stripe-api",
                probability=1.0,
                duration_minutes=20,
                cascade_delay_seconds=10
            )
            
            logs = self.generate_normal_logs(start_time, duration)
            failure_start = start_time + timedelta(minutes=10)
            failure_logs = self.generate_failure_scenario(scenario, failure_start)
            logs.extend(failure_logs)
            logs.sort(key=lambda x: x.timestamp)
            
            return {
                "logs": logs,
                "incidents": [{
                    "failure_type": scenario.failure_type.value,
                    "primary_service": scenario.primary_service,
                    "start_time": failure_start,
                    "duration_minutes": scenario.duration_minutes,
                    "affected_services": ["payment-service"],
                    "root_cause": "Stripe API gateway timeout"
                }],
                "period": {"start_time": start_time, "duration_minutes": duration, "total_logs": len(logs)}
            }
        
        elif scenario_type == "healthy":
            # No failures, just normal operation
            duration = random.randint(30, 90)
            return self.simulate_time_period(start_time, duration, include_failures=False)
        
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    def export_logs_to_json(self, logs: List[LogEntry], filename: str):
        """Export logs to JSON file for analysis"""
        import json
        
        log_dicts = [log.to_dict() for log in logs]
        
        with open(filename, 'w') as f:
            json.dump({
                "logs": log_dicts,
                "metadata": {
                    "total_logs": len(logs),
                    "services": list(set(log.service for log in logs)),
                    "time_range": {
                        "start": min(log.timestamp for log in logs).isoformat() if logs else None,
                        "end": max(log.timestamp for log in logs).isoformat() if logs else None
                    }
                }
            }, f, indent=2)


# Convenience function for quick scenario generation
def generate_incident_scenario(scenario_type: str = "random") -> Dict[str, any]:
    """Quick function to generate a training scenario"""
    simulator = LogSimulator()
    return simulator.generate_training_scenario(scenario_type)


# Example usage
if __name__ == "__main__":
    simulator = LogSimulator()
    
    # Generate a random incident scenario
    scenario = simulator.generate_training_scenario("database_outage")
    
    print(f"Generated {len(scenario['logs'])} logs")
    print(f"Incidents: {len(scenario['incidents'])}")
    
    if scenario['incidents']:
        incident = scenario['incidents'][0]
        print(f"Primary failure: {incident['failure_type']} in {incident['primary_service']}")
        print(f"Affected services: {incident['affected_services']}")
    
    # Export for analysis
    simulator.export_logs_to_json(scenario['logs'], "sample_incident.json")
