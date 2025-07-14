"""
Main entry point for the RL on-call agent training
"""

import argparse
import json
import torch
from pathlib import Path

from trainer import GRPOTrainer, TrainingConfig, OnCallAgent
from environment import generate_scenario
from rubric import evaluate_investigation


def run_training(config_path: str = None, checkpoint_path: str = None, 
                 wandb_args: dict = None):
    """Run the main training loop"""
    
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        # Default configuration
        config = TrainingConfig(
            num_candidates=5,
            batch_size=3,
            max_episodes=50,
            log_every_n_episodes=10,
            model_name="Qwen/Qwen3-1.7B",
            accuracy_weight=0.7,
            efficiency_weight=0.3
        )
    
    # Override wandb settings if provided
    if wandb_args:
        config.use_wandb = not wandb_args.get('no_wandb', False)
        if wandb_args.get('wandb_project'):
            config.wandb_project = wandb_args['wandb_project']
        if wandb_args.get('wandb_entity'):
            config.wandb_entity = wandb_args['wandb_entity']
        if wandb_args.get('wandb_name'):
            config.wandb_run_name = wandb_args['wandb_name']
    
    print("=== RL On-Call Agent Training ===")
    print(f"Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Candidates per scenario: {config.num_candidates}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max episodes: {config.max_episodes}")
    print(f"  Reward weights: {config.accuracy_weight} accuracy + {config.efficiency_weight} efficiency")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Initialize trainer
    trainer = GRPOTrainer(config)
    
    # Load checkpoint if resuming
    start_episode = 0
    if checkpoint_path:
        start_episode = load_checkpoint(trainer, checkpoint_path)
        print(f"Resuming training from episode {start_episode}")
    
    # Run training
    remaining_episodes = config.max_episodes - start_episode
    if remaining_episodes > 0:
        training_metrics = trainer.train(remaining_episodes)
    else:
        print("Training already completed!")
        training_metrics = trainer.training_metrics
    
    # Save results
    results_path = "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"\nTraining results saved to {results_path}")
    
    return trainer, training_metrics


def load_checkpoint(trainer: GRPOTrainer, checkpoint_path: str):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=trainer.agent.device)
    trainer.agent.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.training_metrics = checkpoint["training_metrics"]
    print(f"Loaded checkpoint from episode {checkpoint['episode']}")
    return checkpoint["episode"]


def run_evaluation(model_path: str = None, num_scenarios: int = 10):
    """Run evaluation on test scenarios"""
    
    print("=== Evaluation Mode ===")
    
    # Load model (or use default if no path provided)
    config = TrainingConfig()
    trainer = GRPOTrainer(config)
    
    if model_path:
        load_checkpoint(trainer, model_path)
    
    agent = trainer.agent
    
    # Generate test scenarios
    test_scenarios = [generate_scenario() for _ in range(num_scenarios)]
    
    results = []
    total_reward = 0
    total_accuracy = 0
    total_tokens = 0
    
    print(f"Evaluating on {num_scenarios} test scenarios...")
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i+1}/{num_scenarios}:")
        
        # Create environment and run investigation
        from environment import InvestigationEnvironment
        environment = InvestigationEnvironment(scenario)
        alert = environment.get_alert_message()
        
        print(f"  Alert: {alert}")
        
        # Run investigation
        result = agent.investigate(alert, environment)
        
        # Evaluate
        ground_truth = scenario["incidents"][0] if scenario["incidents"] else {}
        metrics = evaluate_investigation(
            result.diagnosis, 
            result.total_tokens, 
            result.actions_taken, 
            ground_truth
        )
        
        print(f"  Diagnosis: {result.diagnosis}")
        print(f"  Ground Truth: {ground_truth.get('primary_service', 'No incident')} - {ground_truth.get('failure_type', '')}")
        print(f"  Tokens: {result.total_tokens}")
        print(f"  Reward: {metrics.final_reward:.3f} (Acc: {metrics.accuracy_score:.3f}, Eff: {metrics.efficiency_score:.3f})")
        
        results.append({
            "scenario_id": i,
            "alert": alert,
            "diagnosis": result.diagnosis,
            "ground_truth": ground_truth,
            "tokens": result.total_tokens,
            "actions": result.actions_taken,
            "reward": metrics.final_reward,
            "accuracy": metrics.accuracy_score,
            "efficiency": metrics.efficiency_score
        })
        
        total_reward += metrics.final_reward
        total_accuracy += metrics.accuracy_score
        total_tokens += result.total_tokens
    
    # Summary statistics
    avg_reward = total_reward / num_scenarios
    avg_accuracy = total_accuracy / num_scenarios
    avg_tokens = total_tokens / num_scenarios
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Average Accuracy: {avg_accuracy:.3f}")
    print(f"Average Tokens: {avg_tokens:.1f}")
    
    # Save evaluation results
    eval_results = {
        "summary": {
            "num_scenarios": num_scenarios,
            "avg_reward": avg_reward,
            "avg_accuracy": avg_accuracy,
            "avg_tokens": avg_tokens
        },
        "detailed_results": results
    }
    
    eval_path = "evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Evaluation results saved to {eval_path}")
    
    return eval_results


def run_single_investigation():
    """Run a single investigation for testing/demo"""
    
    print("=== Single Investigation Demo ===")
    
    # Generate scenario
    scenario = generate_scenario()
    
    # Setup
    config = TrainingConfig()
    agent = OnCallAgent(config)
    
    from environment import InvestigationEnvironment
    environment = InvestigationEnvironment(scenario)
    alert = environment.get_alert_message()
    
    print(f"Alert: {alert}")
    print(f"Ground Truth: {scenario['incidents'][0] if scenario['incidents'] else 'No incident'}")
    
    # Run investigation
    print("\nInvestigation in progress...")
    result = agent.investigate(alert, environment)
    
    # Show results
    print(f"\nInvestigation Complete:")
    print(f"  Diagnosis: {result.diagnosis}")
    print(f"  Total Tokens: {result.total_tokens}")
    print(f"  Steps Taken: {result.steps_taken}")
    print(f"  Actions: {result.actions_taken}")
    print(f"  Time: {result.investigation_time_seconds:.1f}s")
    
    # Evaluate
    ground_truth = scenario["incidents"][0] if scenario["incidents"] else {}
    metrics = evaluate_investigation(
        result.diagnosis,
        result.total_tokens,
        result.actions_taken,
        ground_truth
    )
    
    print(f"\nEvaluation:")
    print(f"  Final Reward: {metrics.final_reward:.3f}")
    print(f"  Accuracy: {metrics.accuracy_score:.3f}")
    print(f"  Efficiency: {metrics.efficiency_score:.3f}")
    
    return result, metrics


def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "num_candidates": 5,
        "batch_size": 4,
        "max_episodes": 100,
        "log_every_n_episodes": 10,
        "model_name": "Qwen/Qwen3-1.7B",
        "accuracy_weight": 0.7,
        "efficiency_weight": 0.3,
        "max_investigation_steps": 8,
        "temperature": 0.7,
        "learning_rate": 1e-5,
        "use_wandb": True,
        "wandb_project": "on-call-agent-grpo",
        "wandb_entity": None,
        "wandb_run_name": None
    }
    
    config_path = "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Sample configuration saved to {config_path}")
    return config_path


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description="RL On-Call Agent Training")
    parser.add_argument("--mode", choices=["train", "eval", "demo", "config"], 
                       default="train", help="Mode to run")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--scenarios", type=int, default=10, 
                       help="Number of scenarios for evaluation")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume training from checkpoint")
    parser.add_argument("--no-wandb", action="store_true", 
                       help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="on-call-agent-grpo",
                       help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, 
                       help="Wandb entity/team name")
    parser.add_argument("--wandb-name", type=str, 
                       help="Wandb run name")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        wandb_args = {
            'no_wandb': args.no_wandb,
            'wandb_project': args.wandb_project,
            'wandb_entity': args.wandb_entity,
            'wandb_name': args.wandb_name
        }
        trainer, metrics = run_training(args.config, args.checkpoint if args.resume else None, wandb_args)
        
    elif args.mode == "eval":
        results = run_evaluation(args.checkpoint, args.scenarios)
        
    elif args.mode == "demo":
        result, metrics = run_single_investigation()
        
    elif args.mode == "config":
        config_path = create_sample_config()
        print(f"Edit {config_path} to customize training parameters")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
