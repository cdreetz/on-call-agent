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

