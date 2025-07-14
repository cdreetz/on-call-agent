"""
GRPO trainer for the on-call investigation agent
"""

import torch
import torch.nn.functional as F
import json
import random
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from tqdm import tqdm
import wandb

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
    max_generation_length: int = 500
    temperature: float = 0.7
    
    # GRPO specific
    group_size: int = 5  # Number of candidates per group for GRPO
    beta: float = 0.1    # KL regularization coefficient
    clip_range: float = 0.2  # PPO-style clipping
    warmup_episodes: int = 5  # Skip GRPO training for first N episodes
    
    # Logging
    log_every_n_episodes: int = 10
    save_every_n_episodes: int = 50
    
    # Wandb config
    use_wandb: bool = True
    wandb_project: str = "on-call-agent-grpo"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None


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
            step_start = time.time()
            debug_file.write(f"step {step}\n\n")
            # Generate response
            input_tokens = self.count_tokens(context)
            gen_start = time.time()
            full_response = self.generate_response(context)
            gen_time = time.time() - gen_start
            debug_file.write(f"generation took {gen_time:.2f}s\n\n")

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
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_generation_length,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(input_ids[0]):], 
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
        
        # Set up optimizer for GRPO
        self.optimizer = torch.optim.AdamW(
            self.agent.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Store reference model (frozen copy) for KL regularization
        self.ref_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.ref_model.to(self.agent.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Initialize wandb if enabled
        if config.use_wandb:
            self._init_wandb()
        
        # Training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "episode_accuracies": [],
            "episode_tokens": [],
            "policy_losses": [],
            "kl_divergences": [],
            "best_reward": 0.0,
            "best_episode": 0
        }
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        # Create config dict for wandb
        wandb_config = {
            "model_name": self.config.model_name,
            "num_candidates": self.config.num_candidates,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "max_episodes": self.config.max_episodes,
            "accuracy_weight": self.config.accuracy_weight,
            "efficiency_weight": self.config.efficiency_weight,
            "max_investigation_steps": self.config.max_investigation_steps,
            "temperature": self.config.temperature,
            "group_size": self.config.group_size,
            "beta": self.config.beta,
            "clip_range": self.config.clip_range,
        }
        
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            config=wandb_config,
            tags=["grpo", "on-call-agent", "rl"]
        )
        
        # Watch the model for gradients and parameters
        wandb.watch(self.agent.model, log="all", log_freq=100)
        
        print(f"Initialized wandb project: {self.config.wandb_project}")
    
    def generate_candidate_investigations(self, scenario: Dict[str, Any]) -> List[tuple]:
        """Generate multiple investigation candidates for GRPO"""
        
        # Create alert message
        environment = InvestigationEnvironment(scenario)
        alert = environment.get_alert_message()
        
        def run_candidate(seed: int) -> tuple:
            """Run single candidate investigation"""
            try:
                start_time = time.time()
                print(f"Starting candidate {seed}", end="", flush=True)
                
                # Create fresh environment for each candidate
                candidate_env = InvestigationEnvironment(scenario)
                result = self.agent.investigate(alert, candidate_env, seed=seed)
                
                # Calculate metrics
                ground_truth = scenario["incidents"][0] if scenario["incidents"] else {}
                metrics = self.reward_calculator.calculate_reward(result, ground_truth)
                
                duration = time.time() - start_time
                print(f" -> {duration:.1f}s (reward: {metrics.final_reward:.3f})")
                
                return result, metrics, None
            except Exception as e:
                print(f"ERROR in candidate {seed}: {str(e)}")
                import traceback
                traceback.print_exc()
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
        for seed in range(self.config.num_candidates):
            result, metrics, error = run_candidate(seed)
            candidates.append((result, metrics, error))
        
        return candidates
    
    def compute_log_probs(self, prompt: str, response: str, model: torch.nn.Module) -> torch.Tensor:
        """Compute log probabilities for a response given a prompt"""
        # Tokenize prompt and response
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.agent.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        # Tokenize the full sequence (prompt + response)
        full_text = formatted_prompt + response
        prompt_tokens = self.agent.tokenizer.encode(
            formatted_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.agent.device)
        full_tokens = self.agent.tokenizer.encode(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.agent.device)
        
        # Get model predictions
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(full_tokens)
            logits = outputs.logits
        
        # Extract logits for response tokens only
        response_start = prompt_tokens.shape[1] - 1
        response_logits = logits[0, response_start:response_start + (full_tokens.shape[1] - prompt_tokens.shape[1])]
        response_tokens = full_tokens[0, prompt_tokens.shape[1]:]
        
        # Compute log probabilities
        log_probs = F.log_softmax(response_logits, dim=-1)
        response_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        
        return response_log_probs.sum()
    
    def compute_grpo_loss(self, prompts: List[str], responses: List[str], 
                         rewards: List[float], group_size: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss for a batch of responses"""
        total_loss = 0.0
        total_kl = 0.0
        num_groups = len(prompts) // group_size
        
        for group_idx in range(num_groups):
            # Get group data
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            
            group_prompts = prompts[start_idx:end_idx]
            group_responses = responses[start_idx:end_idx]
            group_rewards = torch.tensor(rewards[start_idx:end_idx], device=self.agent.device)
            
            # Compute log probabilities for current policy and reference
            current_log_probs = []
            ref_log_probs = []
            
            for prompt, response in zip(group_prompts, group_responses):
                current_lp = self.compute_log_probs(prompt, response, self.agent.model)
                ref_lp = self.compute_log_probs(prompt, response, self.ref_model)
                current_log_probs.append(current_lp)
                ref_log_probs.append(ref_lp)
            
            current_log_probs = torch.stack(current_log_probs)
            ref_log_probs = torch.stack(ref_log_probs)
            
            # Compute importance weights
            log_ratios = current_log_probs - ref_log_probs
            ratios = torch.exp(log_ratios)
            
            # Compute group-relative advantages
            baseline = group_rewards.mean()
            advantages = group_rewards - baseline
            
            # GRPO loss with KL regularization
            policy_loss = -(ratios * advantages).mean()
            kl_penalty = self.config.beta * log_ratios.mean()
            
            group_loss = policy_loss + kl_penalty
            total_loss += group_loss
            total_kl += kl_penalty.item()
        
        avg_loss = total_loss / num_groups
        avg_kl = total_kl / num_groups
        
        return avg_loss, {"policy_loss": avg_loss.item(), "kl_divergence": avg_kl}
    
    def training_step(self, batch_scenarios: List[Dict[str, Any]], episode: int) -> Dict[str, float]:
        """Execute one GRPO training step"""
        all_candidates = []
        all_rewards = []
        all_metrics = []
        all_prompts = []
        all_responses = []
        
        print(f"Processing {len(batch_scenarios)} scenarios with {self.config.num_candidates} candidates each...")
        
        for i, scenario in enumerate(batch_scenarios):
            print(f"  Scenario {i+1}/{len(batch_scenarios)}: ", end="", flush=True)
            
            # Generate environment for this scenario
            environment = InvestigationEnvironment(scenario)
            alert = environment.get_alert_message()
            
            # Build prompt for this scenario
            from tools import InvestigationTools
            tools = InvestigationTools(environment)
            tool_descriptions = tools.get_tool_descriptions()
            prompt = OnCallPrompts.format_investigation_prompt(alert, tool_descriptions)
            
            candidates = self.generate_candidate_investigations(scenario)
            
            # Extract results and rewards
            scenario_rewards = []
            for result, metrics, error in candidates:
                all_candidates.append(result)
                all_rewards.append(metrics.final_reward)
                all_metrics.append(metrics)
                all_prompts.append(prompt)
                # For simplicity, use the diagnosis as the response to train on
                all_responses.append(result.diagnosis)
                scenario_rewards.append(metrics.final_reward)
            
            # Print scenario summary
            best_reward = max(scenario_rewards)
            avg_reward = sum(scenario_rewards) / len(scenario_rewards)
            print(f"Best: {best_reward:.3f}, Avg: {avg_reward:.3f}")
        
        # Execute GRPO training step (skip during warmup)
        if episode >= self.config.warmup_episodes and len(all_prompts) >= self.config.group_size:
            grpo_start = time.time()
            print("  Executing GRPO training step...", end="", flush=True)
            self.optimizer.zero_grad()
            
            # Compute GRPO loss
            loss, loss_info = self.compute_grpo_loss(
                all_prompts, all_responses, all_rewards, self.config.group_size
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 1.0)
            
            # Update parameters
            self.optimizer.step()
            
            grpo_time = time.time() - grpo_start
            print(f" -> {grpo_time:.1f}s")
            print(f"  Policy Loss: {loss_info['policy_loss']:.4f}, KL: {loss_info['kl_divergence']:.4f}")
        else:
            if episode < self.config.warmup_episodes:
                print(f"  Skipping GRPO training (warmup: {episode+1}/{self.config.warmup_episodes})")
            loss_info = {"policy_loss": 0.0, "kl_divergence": 0.0}
        
        # Calculate training metrics
        avg_reward = sum(all_rewards) / len(all_rewards)
        max_reward = max(all_rewards)
        min_reward = min(all_rewards)
        avg_tokens = sum(c.total_tokens for c in all_candidates) / len(all_candidates)
        avg_accuracy = sum(m.accuracy_score for m in all_metrics) / len(all_metrics)
        avg_steps = sum(c.steps_taken for c in all_candidates) / len(all_candidates)
        
        return {
            "mean_reward": avg_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "avg_tokens": avg_tokens,
            "avg_accuracy": avg_accuracy,
            "avg_steps": avg_steps,
            "num_candidates": len(all_candidates),
            "policy_loss": loss_info["policy_loss"],
            "kl_divergence": loss_info["kl_divergence"]
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
            metrics = self.training_step(batch_scenarios, episode)
            
            # Update training metrics
            self.training_metrics["episode_rewards"].append(metrics["mean_reward"])
            self.training_metrics["episode_accuracies"].append(metrics["avg_accuracy"])
            self.training_metrics["episode_tokens"].append(metrics["avg_tokens"])
            self.training_metrics["policy_losses"].append(metrics["policy_loss"])
            self.training_metrics["kl_divergences"].append(metrics["kl_divergence"])
            
            if metrics["max_reward"] > self.training_metrics["best_reward"]:
                self.training_metrics["best_reward"] = metrics["max_reward"]
                self.training_metrics["best_episode"] = episode + 1
            
            # Log to wandb
            if self.config.use_wandb:
                # Create reward distribution histogram
                episode_rewards = [r for r in all_rewards] if 'all_rewards' in locals() else []
                
                log_dict = {
                    "episode": episode + 1,
                    "train/mean_reward": metrics["mean_reward"],
                    "train/max_reward": metrics["max_reward"], 
                    "train/min_reward": metrics["min_reward"],
                    "train/avg_accuracy": metrics["avg_accuracy"],
                    "train/avg_tokens": metrics["avg_tokens"],
                    "train/avg_steps": metrics["avg_steps"],
                    "train/policy_loss": metrics["policy_loss"],
                    "train/kl_divergence": metrics["kl_divergence"],
                    "train/best_reward": self.training_metrics["best_reward"],
                    "train/num_candidates": metrics["num_candidates"],
                    "train/reward_std": np.std(episode_rewards) if episode_rewards else 0.0
                }
                
                # Add learning rate (in case of scheduling)
                if hasattr(self.optimizer, 'param_groups'):
                    log_dict["train/learning_rate"] = self.optimizer.param_groups[0]['lr']
                
                # Add recent reward trend
                if len(self.training_metrics["episode_rewards"]) >= 10:
                    recent_rewards = self.training_metrics["episode_rewards"][-10:]
                    log_dict["train/recent_10_avg"] = np.mean(recent_rewards)
                
                wandb.log(log_dict)
            
            # Logging
            if (episode + 1) % self.config.log_every_n_episodes == 0:
                self._log_training_progress(episode + 1, metrics)
            
            # Show detailed example occasionally
            if (episode + 1) % (self.config.log_every_n_episodes * 2) == 0:
                self._show_example_investigation(batch_scenarios[0])
            
            # Save model checkpoint
            if (episode + 1) % self.config.save_every_n_episodes == 0:
                self._save_checkpoint(episode + 1)
        
        print(f"\nTraining completed!")
        print(f"Best reward: {self.training_metrics['best_reward']:.3f} (Episode {self.training_metrics['best_episode']})")
        
        # Finish wandb run
        if self.config.use_wandb:
            wandb.finish()
        
        return self.training_metrics
    
    def _log_training_progress(self, episode: int, metrics: Dict[str, float]):
        """Log training progress"""
        print(f"\nTraining Progress (Episode {episode}):")
        print(f"  Mean Reward: {metrics['mean_reward']:.3f}")
        print(f"  Reward Range: {metrics['min_reward']:.3f} - {metrics['max_reward']:.3f}")
        print(f"  Avg Accuracy: {metrics['avg_accuracy']:.3f}")
        print(f"  Avg Tokens: {metrics['avg_tokens']:.1f}")
        print(f"  Avg Steps: {metrics['avg_steps']:.1f}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  KL Divergence: {metrics['kl_divergence']:.4f}")
        
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
        
        # Log example to wandb
        if self.config.use_wandb:
            wandb.log({
                "examples/alert": alert,
                "examples/ground_truth": str(scenario['incidents'][0] if scenario['incidents'] else 'No incident'),
                "examples/best_diagnosis": best_result.diagnosis,
                "examples/best_reward": best_metrics.final_reward,
                "examples/best_accuracy": best_metrics.accuracy_score,
                "examples/best_efficiency": best_metrics.efficiency_score,
                "examples/tokens_used": best_result.total_tokens,
                "examples/steps_taken": best_result.steps_taken,
                "examples/actions_taken": str(best_result.actions_taken)
            })
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint"""
        import os
        os.makedirs("checkpoints", exist_ok=True)
        
        checkpoint = {
            "episode": episode,
            "model_state_dict": self.agent.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_metrics": self.training_metrics
        }
        
        checkpoint_path = f"checkpoints/checkpoint_episode_{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")


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
