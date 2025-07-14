"""
GRPO trainer for the on-call investigation agent
"""

import torch
import json
import random
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

from environment import InvestigationEnvironment, generate_scenario
from tools import InvestigationTools
from prompts import OnCallPrompts
from rubric import RewardCalculator, InvestigationResult, EvaluationMetrics
from utils import ResponseParser, TokenCounter


@dataclass
class TrainingConfig:
    """Configuration for training"""
    num_candidates: int = 5
    max_investigation_steps: int = 8
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_episodes: int = 1000
    
    # Reward weights
    accuracy_weight: float = 0.7
    efficiency_weight: float = 0.3
    
    # Model config
    model_name: str = "Qwen/Qwen3-1.7B"
    max_generation_length: int = 200
    temperature: float = 0.7
    
    # Logging
    log_every_n_episodes: int = 10
    save_every_n_episodes: int = 50


class OnCallAgent:
    """LLM-based on-call investigation agent"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Load model and tokenizer
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.token_counter = TokenCounter(self.tokenizer)
        self.parser = ResponseParser()
        
        print(f"Model loaded on {self.device}")

    def count_tokens(self, text: str) -> int:
        return self.token_counter.count_tokens(text)
    
    def investigate(self, alert: str, environment: InvestigationEnvironment, 
                   seed: Optional[int] = None) -> InvestigationResult:
        """Run a complete investigation"""
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        debug_file = open("inv_debug.txt", "w")
        debug_file.write(f"alert: {alert}\n\n")
        debug_file.write(f"ground truth: {environment.ground_truth}\n\n")
        
        # Initialize tools
        tools = InvestigationTools(environment)
        
        # Build initial prompt
        tool_descriptions = tools.get_tool_descriptions()
        context = OnCallPrompts.format_investigation_prompt(alert, tool_descriptions)

        debug_file.write(f"intiail prompt:\n{context}\n\n")
        
        total_reasoning_tokens = 0
        total_response_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        actions_taken = []
        
        start_time = time.time()
        
        for step in range(self.config.max_investigation_steps):
            debug_file.write(f"step {step}\n\n")
            # Generate response
            input_tokens = self.count_tokens(context)
            full_response = self.generate_response(context)

            debug_file.write(f"model ouptput {full_response}\n\n")

            response, reasoning = self.parser.extract_response_from_reasoning(full_response)
            reasoning_tokens, response_tokens = self.token_counter.count_response_tokens(full_response)

            debug_file.write(f"parsed resposne: {response}\n\n")
            if reasoning:
                debug_file.write(f"reasoning: {reasoning}\n\n")
            
            total_input_tokens += input_tokens
            total_reasoning_tokens += reasoning_tokens
            total_response_tokens += response_tokens
            
            # Try to parse response
            diagnosis = self._try_parse_diagnosis(response)
            if diagnosis:
                debug_file.write(f"final diagnosis {diagnosis}\n\n")
                debug_file.close()
                investigation_time = time.time() - start_time
                return InvestigationResult(
                    diagnosis=diagnosis,
                    total_tokens=total_input_tokens + total_response_tokens,
                    input_tokens=total_input_tokens,
                    output_tokens=total_response_tokens,
                    actions_taken=actions_taken,
                    steps_taken=step + 1,
                    investigation_time_seconds=investigation_time
                )
            
            # Try to parse tool call
            tool_call = self._try_parse_tool_call(response)
            if tool_call:
                tool_name, arguments = tool_call
                result = tools.execute_tool(tool_name, arguments)
                actions_taken.append(f"{tool_name}({arguments})")

                debug_file.write(f"tool call {tool_name} {arguments}\n\n")
                debug_file.write(f"tool result {result}\n\n")
                
                # Add result to context
                context += f"\nAssistant: {response}\n\n"
                context += OnCallPrompts.get_tool_result_prompt(tool_name, result)
            else:
                debug_file.write("invalid response format from tool stuff\n\n")
                context += f"\nAssistant: {response}\n\n"
                context += OnCallPrompts.get_error_handling_prompt("Invalid response format. Please provide valid JSON.")
        
        # Max steps reached, force diagnosis
        context += "\n" + OnCallPrompts.get_diagnosis_request_prompt()
        input_tokens = self.count_tokens(context)
        full_response = self.generate_response(context)

        response, _ = self.parser.extract_response_from_reasoning(full_response)
        reasoning_tokens, response_tokens = self.token_counter.count_response_tokens(full_response)
        
        total_input_tokens += input_tokens
        total_response_tokens += response_tokens
        
        diagnosis = self._try_parse_diagnosis(response) or "Unable to determine root cause"
        investigation_time = time.time() - start_time
        
        return InvestigationResult(
            diagnosis=diagnosis,
            total_tokens=total_input_tokens + total_response_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_response_tokens,
            actions_taken=actions_taken,
            steps_taken=self.config.max_investigation_steps,
            investigation_time_seconds=investigation_time
        )
    
    def _try_parse_diagnosis(self, response: str) -> Optional[str]:
        """Try to parse diagnosis from response"""
        return self.parser.parse_diagnosis(response)

    
    def _try_parse_tool_call(self, response: str) -> Optional[tuple]:
        """Try to parse tool call from response"""
        return self.parser.parse_tool_call(response)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text, truncation=True, max_length=4096))
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from model"""
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_propmt=True
        )
        inputs = self.tokenizer.encode(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_generation_length,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        )
        return response.strip()


class GRPOTrainer:
    """GRPO trainer for the on-call agent"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.agent = OnCallAgent(config)
        self.reward_calculator = RewardCalculator(
            accuracy_weight=config.accuracy_weight,
            efficiency_weight=config.efficiency_weight
        )
        
        # Training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "episode_accuracies": [],
            "episode_tokens": [],
            "best_reward": 0.0,
            "best_episode": 0
        }
    
    def generate_candidate_investigations(self, scenario: Dict[str, Any]) -> List[tuple]:
        """Generate multiple investigation candidates for GRPO"""
        
        # Create alert message
        environment = InvestigationEnvironment(scenario)
        alert = environment.get_alert_message()
        
        def run_candidate(seed: int) -> tuple:
            """Run single candidate investigation"""
            try:
                # Create fresh environment for each candidate
                candidate_env = InvestigationEnvironment(scenario)
                result = self.agent.investigate(alert, candidate_env, seed=seed)
                
                # Calculate metrics
                ground_truth = scenario["incidents"][0] if scenario["incidents"] else {}
                metrics = self.reward_calculator.calculate_reward(result, ground_truth)
                
                return result, metrics, None
            except Exception as e:
                # Return dummy result on error
                dummy_result = InvestigationResult(
                    diagnosis="Error during investigation",
                    total_tokens=1000,
                    input_tokens=700,
                    output_tokens=300,
                    actions_taken=[],
                    steps_taken=0
                )
                dummy_metrics = EvaluationMetrics(
                    accuracy_score=0.0,
                    efficiency_score=0.0,
                    final_reward=0.0,
                    exact_match=False,
                    service_match=False,
                    category_match=False,
                    token_efficiency=0.0,
                    step_efficiency=0.0,
                    action_efficiency=0.0
                )
                return dummy_result, dummy_metrics, str(e)
        
        # Generate candidates in parallel
        candidates = []
        with ThreadPoolExecutor(max_workers=self.config.num_candidates) as executor:
            future_to_seed = {
                executor.submit(run_candidate, seed): seed 
                for seed in range(self.config.num_candidates)
            }
            
            for future in as_completed(future_to_seed):
                result, metrics, error = future.result()
                candidates.append((result, metrics, error))
        
        return candidates
    
    def training_step(self, batch_scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
        """Execute one GRPO training step"""
        all_candidates = []
        all_rewards = []
        all_metrics = []
        
        print(f"Processing {len(batch_scenarios)} scenarios with {self.config.num_candidates} candidates each...")
        
        for i, scenario in enumerate(batch_scenarios):
            print(f"  Scenario {i+1}/{len(batch_scenarios)}: ", end="", flush=True)
            
            candidates = self.generate_candidate_investigations(scenario)
            
            # Extract results and rewards
            scenario_rewards = []
            for result, metrics, error in candidates:
                all_candidates.append(result)
                all_rewards.append(metrics.final_reward)
                all_metrics.append(metrics)
                scenario_rewards.append(metrics.final_reward)
            
            # Print scenario summary
            best_reward = max(scenario_rewards)
            avg_reward = sum(scenario_rewards) / len(scenario_rewards)
            print(f"Best: {best_reward:.3f}, Avg: {avg_reward:.3f}")
        
        # Calculate training metrics
        avg_reward = sum(all_rewards) / len(all_rewards)
        max_reward = max(all_rewards)
        min_reward = min(all_rewards)
        avg_tokens = sum(c.total_tokens for c in all_candidates) / len(all_candidates)
        avg_accuracy = sum(m.accuracy_score for m in all_metrics) / len(all_metrics)
        avg_steps = sum(c.steps_taken for c in all_candidates) / len(all_candidates)
        
        # TODO: Implement actual GRPO policy gradient update here
        # This would involve:
        # 1. Computing log probabilities of generated sequences
        # 2. Computing group-relative advantages
        # 3. Backpropagating gradients to update model weights
        
        return {
            "mean_reward": avg_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "avg_tokens": avg_tokens,
            "avg_accuracy": avg_accuracy,
            "avg_steps": avg_steps,
            "num_candidates": len(all_candidates)
        }
    
    def train(self, num_episodes: Optional[int] = None) -> Dict[str, List[float]]:
        """Main training loop"""
        if num_episodes is None:
            num_episodes = self.config.max_episodes
        
        print(f"Starting GRPO training for {num_episodes} episodes...")
        print(f"Config: {self.config.num_candidates} candidates per scenario, batch size {self.config.batch_size}")
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Generate batch of scenarios
            batch_scenarios = [generate_scenario() for _ in range(self.config.batch_size)]
            
            # Training step
            metrics = self.training_step(batch_scenarios)
            
            # Update training metrics
            self.training_metrics["episode_rewards"].append(metrics["mean_reward"])
            self.training_metrics["episode_accuracies"].append(metrics["avg_accuracy"])
            self.training_metrics["episode_tokens"].append(metrics["avg_tokens"])
            
            if metrics["max_reward"] > self.training_metrics["best_reward"]:
                self.training_metrics["best_reward"] = metrics["max_reward"]
                self.training_metrics["best_episode"] = episode + 1
            
            # Logging
            if (episode + 1) % self.config.log_every_n_episodes == 0:
                self._log_training_progress(episode + 1, metrics)
            
            # Show detailed example occasionally
            if (episode + 1) % (self.config.log_every_n_episodes * 2) == 0:
                self._show_example_investigation(batch_scenarios[0])
        
        print(f"\nTraining completed!")
        print(f"Best reward: {self.training_metrics['best_reward']:.3f} (Episode {self.training_metrics['best_episode']})")
        
        return self.training_metrics
    
    def _log_training_progress(self, episode: int, metrics: Dict[str, float]):
        """Log training progress"""
        print(f"\nTraining Progress (Episode {episode}):")
        print(f"  Mean Reward: {metrics['mean_reward']:.3f}")
        print(f"  Reward Range: {metrics['min_reward']:.3f} - {metrics['max_reward']:.3f}")
        print(f"  Avg Accuracy: {metrics['avg_accuracy']:.3f}")
        print(f"  Avg Tokens: {metrics['avg_tokens']:.1f}")
        print(f"  Avg Steps: {metrics['avg_steps']:.1f}")
        
        # Show recent trend
        if len(self.training_metrics["episode_rewards"]) >= 10:
            recent_rewards = self.training_metrics["episode_rewards"][-10:]
            trend = sum(recent_rewards) / len(recent_rewards)
            print(f"  Recent 10-episode average: {trend:.3f}")
    
    def _show_example_investigation(self, scenario: Dict[str, Any]):
        """Show detailed example of best investigation"""
        print(f"\nExample Investigation:")
        
        candidates = self.generate_candidate_investigations(scenario)
        best_result, best_metrics, _ = max(candidates, key=lambda x: x[1].final_reward)
        
        environment = InvestigationEnvironment(scenario)
        alert = environment.get_alert_message()
        
        print(f"  Alert: {alert}")
        print(f"  Ground Truth: {scenario['incidents'][0] if scenario['incidents'] else 'No incident'}")
        print(f"  Best Diagnosis: {best_result.diagnosis}")
        print(f"  Actions: {best_result.actions_taken}")
        print(f"  Tokens: {best_result.total_tokens}")
        print(f"  Reward: {best_metrics.final_reward:.3f}")
        print(f"  Breakdown: Accuracy={best_metrics.accuracy_score:.3f}, Efficiency={best_metrics.efficiency_score:.3f}")


def main():
    """Main training script"""
    config = TrainingConfig(
        num_candidates=5,
        batch_size=2,
        max_episodes=20,  # Small test run
        log_every_n_episodes=5
    )
    
    trainer = GRPOTrainer(config)
    training_metrics = trainer.train()
    
    print("\nTraining Summary:")
    print(f"Episodes completed: {len(training_metrics['episode_rewards'])}")
    print(f"Final average reward: {training_metrics['episode_rewards'][-1]:.3f}")
    print(f"Best reward achieved: {training_metrics['best_reward']:.3f}")
    print(f"Final average accuracy: {training_metrics['episode_accuracies'][-1]:.3f}")
    print(f"Final average tokens: {training_metrics['episode_tokens'][-1]:.1f}")


if __name__ == "__main__":
    main()
