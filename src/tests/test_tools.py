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
    print(tools.execute("check_status_page", {"service": "stripe"}))
    
    print("\n2. Deployment Check:")
    print(tools.execute("check_deployments", {"hours": 6}))
    
    print("\n3. Slack Search:")
    print(tools.execute("search_slack", {"channel": "oncall-alerts"}))
    
    print("\n4. Log Query:")
    print(tools.execute("query_logs", {"service": "payment-service", "time_range": "30m", "filters": "error"}))


