"""
Reward calculation and evaluation rubric for RL training
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class InvestigationResult:
    """Result of an investigation attempt"""
    diagnosis: str
    total_tokens: int
    input_tokens: int
    output_tokens: int
    actions_taken: List[str]
    steps_taken: int
    investigation_time_seconds: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    accuracy_score: float
    efficiency_score: float
    final_reward: float
    
    # Detailed accuracy breakdown
    exact_match: bool
    service_match: bool
    category_match: bool
    
    # Detailed efficiency breakdown
    token_efficiency: float
    step_efficiency: float
    action_efficiency: float


class AccuracyEvaluator:
    """Evaluates diagnosis accuracy against ground truth"""
    
    def __init__(self):
        # Service category mappings for partial credit
        self.service_categories = {
            "database": ["database", "postgres", "mysql", "db", "rds"],
            "payment": ["payment", "stripe", "billing", "transaction"],
            "api": ["api", "gateway", "endpoint", "service"],
            "auth": ["auth", "authentication", "login", "session"],
            "cache": ["cache", "redis", "memcached"],
            "external": ["external", "third-party", "upstream", "dependency"]
        }
        
        # Failure type mappings
        self.failure_types = {
            "timeout": ["timeout", "slow", "latency", "response time"],
            "connection": ["connection", "connectivity", "network"],
            "memory": ["memory", "oom", "out of memory", "leak"],
            "cpu": ["cpu", "processing", "load"],
            "outage": ["outage", "down", "unavailable", "degraded"],
            "deployment": ["deployment", "deploy", "release", "code change"]
        }
    
    def evaluate_accuracy(self, diagnosis: str, ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, bool]]:
        """
        Evaluate diagnosis accuracy against ground truth
        Returns (accuracy_score, breakdown_dict)
        """
        if not ground_truth:
            return 0.5, {"exact_match": False, "service_match": False, "category_match": False}
        
        diagnosis_lower = diagnosis.lower()
        primary_service = ground_truth.get("primary_service", "").lower()
        failure_type = ground_truth.get("failure_type", "").lower()
        root_cause = ground_truth.get("root_cause", "").lower()
        
        # Check for exact/near-exact match
        exact_match = self._check_exact_match(diagnosis_lower, primary_service, failure_type, root_cause)
        if exact_match:
            return 1.0, {"exact_match": True, "service_match": True, "category_match": True}
        
        # Check for service match
        service_match = self._check_service_match(diagnosis_lower, primary_service)
        if service_match:
            return 0.8, {"exact_match": False, "service_match": True, "category_match": True}
        
        # Check for category match
        category_match = self._check_category_match(diagnosis_lower, primary_service, failure_type)
        if category_match:
            return 0.6, {"exact_match": False, "service_match": False, "category_match": True}
        
        # Check for failure type match (minimal credit)
        failure_match = self._check_failure_type_match(diagnosis_lower, failure_type)
        if failure_match:
            return 0.3, {"exact_match": False, "service_match": False, "category_match": False}
        
        # No significant match
        return 0.1, {"exact_match": False, "service_match": False, "category_match": False}
    
    def _check_exact_match(self, diagnosis: str, service: str, failure_type: str, root_cause: str) -> bool:
        """Check if diagnosis closely matches the actual root cause"""
        # Direct service name match + failure type match
        if service in diagnosis and any(ft in diagnosis for ft in failure_type.split("_")):
            return True
        
        # Root cause keywords match
        if root_cause and len(root_cause) > 5:
            root_cause_words = root_cause.split()
            if len(root_cause_words) >= 2:
                matches = sum(1 for word in root_cause_words if word in diagnosis and len(word) > 3)
                if matches >= 2:
                    return True
        
        return False
    
    def _check_service_match(self, diagnosis: str, service: str) -> bool:
        """Check if diagnosis identifies the correct service"""
        if not service:
            return False
        
        # Direct service name match
        if service in diagnosis:
            return True
        
        # Service name variations (e.g., "payment-service" vs "payment")
        service_base = service.replace("-service", "").replace("_service", "")
        if service_base in diagnosis:
            return True
        
        return False
    
    def _check_category_match(self, diagnosis: str, service: str, failure_type: str) -> bool:
        """Check if diagnosis identifies the correct service category"""
        # Check service category match
        for category, keywords in self.service_categories.items():
            if any(keyword in service for keyword in keywords):
                if any(keyword in diagnosis for keyword in keywords):
                    return True
        
        # Check failure type category match
        for category, keywords in self.failure_types.items():
            if any(keyword in failure_type for keyword in keywords):
                if any(keyword in diagnosis for keyword in keywords):
                    return True
        
        return False
    
    def _check_failure_type_match(self, diagnosis: str, failure_type: str) -> bool:
        """Check if diagnosis mentions the correct type of failure"""
        if not failure_type:
            return False
        
        failure_keywords = failure_type.replace("_", " ").split()
        return any(keyword in diagnosis for keyword in failure_keywords if len(keyword) > 3)


class EfficiencyEvaluator:
    """Evaluates investigation efficiency"""
    
    def __init__(self):
        # Optimal token ranges for different investigation types
        self.optimal_ranges = {
            "external_dependency": (50, 150),    # Status page check + confirmation
            "deployment_issue": (100, 250),      # Status + deployment check + light logs
            "infrastructure": (150, 300),        # Multiple checks + some log analysis
            "complex_issue": (300, 600)          # Deep investigation required
        }
        
        # Action efficiency weights
        self.action_weights = {
            "check_status_page": 1.0,      # Highly efficient
            "check_deployments": 1.0,      # Highly efficient  
            "search_slack": 0.9,           # Pretty efficient
            "query_logs": 0.3              # Less efficient but sometimes necessary
        }
    
    def evaluate_efficiency(self, result: InvestigationResult, ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate multiple aspects of investigation efficiency"""
        
        # Token efficiency (diminishing returns on high token count)
        token_efficiency = self._calculate_token_efficiency(result.total_tokens)
        
        # Step efficiency (fewer steps generally better)
        step_efficiency = self._calculate_step_efficiency(result.steps_taken)
        
        # Action efficiency (weighted by action types used)
        action_efficiency = self._calculate_action_efficiency(result.actions_taken)
        
        # Overall efficiency score
        efficiency_score = (token_efficiency * 0.6 + step_efficiency * 0.2 + action_efficiency * 0.2)
        
        return {
            "token_efficiency": token_efficiency,
            "step_efficiency": step_efficiency, 
            "action_efficiency": action_efficiency,
            "overall_efficiency": efficiency_score
        }
    
    def _calculate_token_efficiency(self, total_tokens: int) -> float:
        """Calculate efficiency based on token usage (diminishing returns)"""
        if total_tokens <= 0:
            return 0.0
        
        # Logarithmic efficiency - heavily penalize high token usage
        # Peak efficiency around 50-100 tokens, sharp decline after 200
        if total_tokens <= 50:
            return 1.0
        elif total_tokens <= 100:
            return 0.9
        elif total_tokens <= 200:
            return 0.7
        elif total_tokens <= 400:
            return 0.5
        elif total_tokens <= 800:
            return 0.3
        else:
            return 0.1
    
    def _calculate_step_efficiency(self, steps: int) -> float:
        """Calculate efficiency based on number of investigation steps"""
        if steps <= 0:
            return 0.0
        elif steps == 1:
            return 1.0
        elif steps == 2:
            return 0.9
        elif steps == 3:
            return 0.8
        elif steps <= 5:
            return 0.6
        elif steps <= 8:
            return 0.4
        else:
            return 0.2
    
    def _calculate_action_efficiency(self, actions: List[str]) -> float:
        """Calculate efficiency based on types of actions taken"""
        if not actions:
            return 0.0
        
        total_weight = 0.0
        max_possible_weight = len(actions) * 1.0  # If all actions were status page checks
        
        for action in actions:
            # Extract action type from action string
            action_type = self._extract_action_type(action)
            weight = self.action_weights.get(action_type, 0.5)  # Default weight for unknown actions
            total_weight += weight
        
        return total_weight / max_possible_weight
    
    def _extract_action_type(self, action: str) -> str:
        """Extract action type from action string"""
        action_lower = action.lower()
        if "status" in action_lower:
            return "check_status_page"
        elif "deployment" in action_lower:
            return "check_deployments"
        elif "slack" in action_lower:
            return "search_slack"
        elif "log" in action_lower:
            return "query_logs"
        else:
            return "unknown"


class RewardCalculator:
    """Main reward calculation for GRPO training"""
    
    def __init__(self, accuracy_weight: float = 0.7, efficiency_weight: float = 0.3):
        self.accuracy_weight = accuracy_weight
        self.efficiency_weight = efficiency_weight
        self.accuracy_evaluator = AccuracyEvaluator()
        self.efficiency_evaluator = EfficiencyEvaluator()
        
        # Minimum accuracy threshold for efficiency bonus
        self.min_accuracy_for_efficiency = 0.5
    
    def calculate_reward(self, result: InvestigationResult, ground_truth: Dict[str, Any]) -> EvaluationMetrics:
        """Calculate complete reward and metrics for an investigation"""
        
        # Calculate accuracy
        accuracy_score, accuracy_breakdown = self.accuracy_evaluator.evaluate_accuracy(
            result.diagnosis, ground_truth
        )
        
        # Calculate efficiency
        efficiency_metrics = self.efficiency_evaluator.evaluate_efficiency(result, ground_truth)
        efficiency_score = efficiency_metrics["overall_efficiency"]
        
        # Calculate final reward
        if accuracy_score < self.min_accuracy_for_efficiency:
            # No efficiency bonus for poor accuracy
            final_reward = self.accuracy_weight * accuracy_score
        else:
            # Full reward calculation
            final_reward = (self.accuracy_weight * accuracy_score + 
                          self.efficiency_weight * efficiency_score)
        
        return EvaluationMetrics(
            accuracy_score=accuracy_score,
            efficiency_score=efficiency_score,
            final_reward=final_reward,
            exact_match=accuracy_breakdown["exact_match"],
            service_match=accuracy_breakdown["service_match"],
            category_match=accuracy_breakdown["category_match"],
            token_efficiency=efficiency_metrics["token_efficiency"],
            step_efficiency=efficiency_metrics["step_efficiency"],
            action_efficiency=efficiency_metrics["action_efficiency"]
        )
    
    def calculate_group_rewards(self, results: List[InvestigationResult], 
                              ground_truth: Dict[str, Any]) -> List[EvaluationMetrics]:
        """Calculate rewards for a group of candidates (for GRPO)"""
        return [self.calculate_reward(result, ground_truth) for result in results]
    
    def get_reward_explanation(self, metrics: EvaluationMetrics) -> str:
        """Generate human-readable explanation of reward calculation"""
        explanation = f"REWARD BREAKDOWN:\n"
        explanation += f"Final Reward: {metrics.final_reward:.3f}\n\n"
        
        explanation += f"Accuracy: {metrics.accuracy_score:.3f} (weight: {self.accuracy_weight})\n"
        if metrics.exact_match:
            explanation += "  ✅ Exact match with ground truth\n"
        elif metrics.service_match:
            explanation += "  ✅ Correct service identified\n"
        elif metrics.category_match:
            explanation += "  ⚠️  Correct category identified\n"
        else:
            explanation += "  ❌ Incorrect diagnosis\n"
        
        explanation += f"\nEfficiency: {metrics.efficiency_score:.3f} (weight: {self.efficiency_weight})\n"
        explanation += f"  Token efficiency: {metrics.token_efficiency:.3f}\n"
        explanation += f"  Step efficiency: {metrics.step_efficiency:.3f}\n"
        explanation += f"  Action efficiency: {metrics.action_efficiency:.3f}\n"
        
        if metrics.accuracy_score < self.min_accuracy_for_efficiency:
            explanation += f"\n⚠️  Efficiency bonus disabled (accuracy < {self.min_accuracy_for_efficiency})"
        
        return explanation


# Convenience functions for easy usage
def evaluate_investigation(diagnosis: str, total_tokens: int, actions: List[str], 
                         ground_truth: Dict[str, Any]) -> EvaluationMetrics:
    """Convenience function to evaluate a single investigation"""
    result = InvestigationResult(
        diagnosis=diagnosis,
        total_tokens=total_tokens,
        input_tokens=int(total_tokens * 0.7),  # Estimate
        output_tokens=int(total_tokens * 0.3),  # Estimate
        actions_taken=actions,
        steps_taken=len(actions)
    )
    
    calculator = RewardCalculator()
    return calculator.calculate_reward(result, ground_truth)


def compare_investigations(investigations: List[Tuple[str, int, List[str]]], 
                         ground_truth: Dict[str, Any]) -> List[Tuple[EvaluationMetrics, str]]:
    """Compare multiple investigations and return ranked results"""
    results = []
    
    for diagnosis, tokens, actions in investigations:
        metrics = evaluate_investigation(diagnosis, tokens, actions, ground_truth)
        results.append((metrics, diagnosis))
    
    # Sort by reward (highest first)
    results.sort(key=lambda x: x[0].final_reward, reverse=True)
    
    return results
