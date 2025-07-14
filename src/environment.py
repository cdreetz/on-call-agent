"""
Environment for incident investigation simulation
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random

from log_entry import LogEntry, LogLevel


@dataclass
class StatusPageEntry:
    service: str
    status: str  # "operational", "degraded", "down"
    message: str


@dataclass
class DeploymentEntry:
    service: str
    version: str
    deployed_at: datetime
    deployed_by: str


@dataclass
class SlackMessage:
    channel: str
    user: str
    timestamp: datetime
    message: str


class InvestigationEnvironment:
    """Environment for incident investigation"""
    
    def __init__(self, scenario: Dict[str, Any]):
        self.scenario = scenario
        self.logs = scenario["logs"]
        self.incidents = scenario["incidents"]
        self.ground_truth = scenario["incidents"][0] if scenario["incidents"] else None
        
        ## Generate supporting data sources
        #self.status_pages = self._generate_status_pages()
        #self.deployments = self._generate_deployments()
        #self.slack_messages = self._generate_slack_messages()

        self.status_pages = {
            "aws": StatusPageEntry("aws", "operational", "All systems operational"),
            "stripe": StatusPageEntry("stripe", "operational", "All systems operational"),
            "cloudflare": StatusPageEntry("cloudflare", "operational", "All systems operational"),
            "github": StatusPageEntry("github", "operational", "All systems operational"),
            "datadog": StatusPageEntry("datadog", "operational", "All systems operational")
        }

        if self.ground_truth:
            if "stripe" in self.ground_truth["primary_service"]:
                self.status_pages["stripe"] = StatusPageEntry("stripe", "degraded", "Elevated error rates")
            elif "database" in self.ground_truth["primary_service"]:
                self.status_pages["aws"] = StatusPageEntry("aws", "degraded", "RDS connectivity issues")

        self.deployments = self._generate_deployments()
        self.slack_messages = self._generate_slack_messages()

    
    #def _generate_status_pages(self) -> Dict[str, StatusPageEntry]:
    #    """Generate realistic status pages based on incidents"""
    #    pages = {
    #        "aws": StatusPageEntry("aws", "operational", "All systems operational"),
    #        "stripe": StatusPageEntry("stripe", "operational", "All systems operational"),
    #        "cloudflare": StatusPageEntry("cloudflare", "operational", "All systems operational"),
    #        "github": StatusPageEntry("github", "operational", "All systems operational"),
    #        "datadog": StatusPageEntry("datadog", "operational", "All systems operational")
    #    }
    #    
    #    # Update status based on incidents
    #    if self.ground_truth:
    #        primary_service = self.ground_truth["primary_service"]
    #        failure_type = self.ground_truth.get("failure_type", "")
    #        
    #        if primary_service == "stripe-api":
    #            pages["stripe"] = StatusPageEntry(
    #                "stripe", 
    #                "degraded" if random.random() > 0.2 else "down",
    #                "Elevated error rates on payment processing API"
    #            )
    #        elif "database" in primary_service and random.random() < 0.3:
    #            pages["aws"] = StatusPageEntry(
    #                "aws", 
    #                "degraded", 
    #                "Connectivity issues with RDS in us-west-2"
    #            )
    #        elif "external_api" in failure_type and random.random() < 0.4:
    #            external_service = random.choice(["github", "datadog", "cloudflare"])
    #            pages[external_service] = StatusPageEntry(
    #                external_service,
    #                "degraded",
    #                f"API response time degradation"
    #            )
    #    
    #    return pages
    
    def _generate_deployments(self) -> List[DeploymentEntry]:
        """Generate deployment history around incident times"""
        deployments = []
        base_time = datetime.now()
        
        # Generate normal background deployments
        services = ["user-service", "payment-service", "auth-service", "api-gateway"]
        for i in range(4):
            service = random.choice(services)
            deploy_time = base_time - timedelta(hours=random.randint(2, 72))
            deployments.append(DeploymentEntry(
                service=service,
                version=f"v{random.randint(1, 3)}.{random.randint(0, 15)}.{random.randint(0, 10)}",
                deployed_at=deploy_time,
                deployed_by="ci-cd-pipeline"
            ))
        
        # Generate suspicious deployment if incident might be deployment-related
        if self.ground_truth and random.random() < 0.35:  # 35% chance incident is deployment-related
            incident_time = self.ground_truth["start_time"]
            # Deploy 5-45 minutes before incident
            deploy_time = incident_time - timedelta(minutes=random.randint(5, 45))
            deployments.append(DeploymentEntry(
                service=self.ground_truth["primary_service"],
                version=f"v{random.randint(1, 3)}.{random.randint(0, 15)}.{random.randint(0, 10)}",
                deployed_at=deploy_time,
                deployed_by=random.choice(["engineer-alice", "engineer-bob", "engineer-charlie"])
            ))
        
        return sorted(deployments, key=lambda x: x.deployed_at, reverse=True)
    
    def _generate_slack_messages(self) -> List[SlackMessage]:
        """Generate Slack messages about outages and incidents"""
        messages = []
        
        if not self.ground_truth:
            return messages
        
        incident_time = self.ground_truth["start_time"]
        primary_service = self.ground_truth["primary_service"]
        
        # External service announcements
        if primary_service == "stripe-api":
            messages.append(SlackMessage(
                channel="oncall-alerts",
                user="stripe-status-bot",
                timestamp=incident_time + timedelta(minutes=random.randint(1, 5)),
                message="🚨 Stripe API experiencing elevated error rates. Engineers investigating."
            ))
            messages.append(SlackMessage(
                channel="payments-team",
                user="team-lead",
                timestamp=incident_time + timedelta(minutes=random.randint(3, 8)),
                message="Seeing payment failures spike - likely Stripe issue, monitoring"
            ))
        
        # Customer support reports
        messages.append(SlackMessage(
            channel="customer-support",
            user="support-agent-sarah",
            timestamp=incident_time + timedelta(minutes=random.randint(5, 12)),
            message=f"Multiple customers reporting issues with {primary_service.replace('-', ' ')}"
        ))
        
        # Engineering chatter
        if "database" in primary_service:
            messages.append(SlackMessage(
                channel="backend-eng",
                user="dba-mike",
                timestamp=incident_time + timedelta(minutes=random.randint(2, 7)),
                message="Database connection pool looking stressed, investigating"
            ))
        
        # General incident awareness
        messages.append(SlackMessage(
            channel="oncall-alerts",
            user="monitoring-bot",
            timestamp=incident_time + timedelta(minutes=random.randint(1, 3)),
            message=f"Alert: High error rate in {primary_service} - oncall engineer notified"
        ))
        
        return messages

    def get_alert_message(self) -> str:
        if self.ground_truth:
            service = self.ground_truth["primary_service"]
            return f"High error rate alert: {service} - 15% errors in last 5 minutes"
        return "Service degredation alert: multiple services showing elevated error rates"
    
    #def get_alert_message(self) -> str:
    #    """Generate initial alert message for the incident"""
    #    if self.ground_truth:
    #        service = self.ground_truth["primary_service"]
    #        failure_type = self.ground_truth.get("failure_type", "")
    #        
    #        if "timeout" in failure_type:
    #            return f"High response time alert: {service} - 95th percentile >10s for 5+ minutes"
    #        elif "api_error" in failure_type:
    #            return f"High error rate alert: {service} - 15% 5xx responses in last 5 minutes"
    #        elif "memory" in failure_type:
    #            return f"Resource alert: {service} - Memory usage >90% for 10+ minutes"
    #        else:
    #            return f"Service degradation alert: {service} - multiple failure indicators"
    #    else:
    #        return "Service degradation alert: multiple services showing elevated error rates"


def generate_scenario() -> Dict[str, Any]:
    """Generate a training scenario with logs and incidents"""
    from log_simulator import LogSimulator
    
    simulator = LogSimulator()
    
    # Weight scenario types based on real-world frequency
    scenario_weights = {
        "external_api_failure": 0.35,  # Most common - external dependencies
        "database_outage": 0.25,       # Database issues
        "deployment_issue": 0.20,      # Recent deployment problems  
        "resource_exhaustion": 0.15,   # Memory/CPU issues
        "healthy": 0.05                # No real incident (false positive)
    }
    
    # Select scenario type based on weights
    rand = random.random()
    cumulative = 0
    selected_type = "random"
    
    for scenario_type, weight in scenario_weights.items():
        cumulative += weight
        if rand <= cumulative:
            selected_type = scenario_type
            break
    
    return simulator.generate_training_scenario(selected_type)
