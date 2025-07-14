# test_grpo_training.py - Test GRPO training with file output
def test_grpo_training():
    """Test GRPO training and write investigation trace to file"""
    from trainer import TrainingConfig, GRPOTrainer, OnCallAgent
    from environment import generate_scenario, InvestigationEnvironment
    from tools import InvestigationTools
    from prompts import OnCallPrompts
    import json
    
    print("\n=== Testing GRPO Training ===\n")
    
    # Small config for testing
    config = TrainingConfig(
        num_candidates=2,
        batch_size=1,
        max_episodes=3,
        log_every_n_episodes=1,
        model_name="microsoft/DialoGPT-small"  # Smaller for testing
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(config)
    
    # Run a few training steps
    print("Running 3 training episodes...")
    metrics = trainer.train(num_episodes=3)
    
    print(f"\nFinal metrics:")
    print(f"  Rewards: {metrics['rewards']}")
    print(f"  Tokens: {metrics['tokens']}")
    print(f"  Accuracy: {metrics['accuracy']}")
    
    # Generate one investigation and save to file
    print("\nGenerating sample investigation trace...")
    
    scenario = generate_scenario()
    env = InvestigationEnvironment(scenario)
    alert = env.get_alert_message()
    
    # Run investigation
    agent = OnCallAgent(config)
    result = agent.investigate(alert, env)
    
    # Create investigation trace
    trace = {
        "alert": alert,
        "ground_truth": env.ground_truth,
        "investigation": {
            "diagnosis": result.diagnosis,
            "total_tokens": result.total_tokens,
            "steps": result.steps_taken,
            "actions": result.actions_taken,
            "time_seconds": result.investigation_time_seconds
        },
        "reward_calculation": {
            "accuracy": "1.0 if service match, 0.3 if category match, 0.1 otherwise",
            "efficiency": f"1.0 if <100 tokens, 0.7 if <200, 0.4 if <400, 0.1 otherwise",
            "final": f"0.7 * accuracy + 0.3 * efficiency"
        }
    }
    
    # Write to file
    with open("investigation_trace.json", "w") as f:
        json.dump(trace, f, indent=2, default=str)
    
    print(f"Investigation trace written to investigation_trace.json")
    print(f"\nInvestigation summary:")
    print(f"  Diagnosis: {result.diagnosis}")
    print(f"  Tokens used: {result.total_tokens}")
    print(f"  Actions taken: {result.actions_taken}")
