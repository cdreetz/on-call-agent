#!/usr/bin/env python3
"""
Test scripts for the on-call agent components
"""

# test_environment.py - Test environment and log generation
def test_environment():
    """Test environment generation with sample error logs"""
    from environment import generate_scenario, InvestigationEnvironment
    from log_entry import LogLevel
    
    print("=== Testing Environment Generation ===\n")
    
    # Generate a scenario
    scenario = generate_scenario()
    env = InvestigationEnvironment(scenario)
    
    # Show alert
    print(f"Alert: {env.get_alert_message()}\n")
    
    # Show ground truth
    if env.ground_truth:
        print(f"Ground Truth:")
        print(f"  Service: {env.ground_truth['primary_service']}")
        print(f"  Type: {env.ground_truth.get('failure_type', 'unknown')}")
        print(f"  Root Cause: {env.ground_truth.get('root_cause', 'unknown')}\n")
    
    # Show some error logs
    error_logs = [log for log in env.logs if log.level == LogLevel.ERROR][:5]
    print(f"Sample Error Logs ({len(error_logs)} shown):")
    for log in error_logs:
        print(f"  [{log.timestamp.strftime('%H:%M:%S')}] {log.service}: {log.message}")
    
    # Show status pages
    print(f"\nStatus Pages:")
    for service, status in env.status_pages.items():
        if status.status != "operational":
            print(f"  {service}: {status.status} - {status.message}")
    
    # Show recent deployments
    print(f"\nRecent Deployments:")
    for dep in env.deployments[:3]:
        print(f"  {dep.service} {dep.version} by {dep.deployed_by}")


# test_tools.py - Test tool execution
def test_tools():
    """Test investigation tools"""
    from environment import generate_scenario, InvestigationEnvironment
    from tools import InvestigationTools
    
    print("\n=== Testing Investigation Tools ===\n")
    
    scenario = generate_scenario()
    env = InvestigationEnvironment(scenario)
    tools = InvestigationTools(env)
    
    # Test each tool
    print("1. Status Page Check:")
    print(tools.execute_tool("check_status_page", {"service": "stripe"}))
    
    print("\n2. Deployment Check:")
    print(tools.execute_tool("check_deployments", {"hours": 6}))
    
    print("\n3. Slack Search:")
    print(tools.execute_tool("search_slack", {"channel": "oncall-alerts"}))
    
    print("\n4. Log Query:")
    print(tools.execute_tool("query_logs", {"service": "payment-service", "time_range": "30m", "filters": "error"}))


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


# main test runner
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "env":
            test_environment()
        elif sys.argv[1] == "tools":
            test_tools()
        elif sys.argv[1] == "grpo":
            test_grpo_training()
        else:
            print("Usage: python test_scripts.py [env|tools|grpo]")
    else:
        # Run all tests
        test_environment()
        test_tools()
        test_grpo_training()
