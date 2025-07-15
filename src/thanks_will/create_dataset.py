import random
import copy
from datetime import datetime, timedelta
from datasets import Dataset

def generate_random_datetime(base_date="2024-01-15", hours_back=24):
    """Generate random datetime within specified range."""
    base = datetime.fromisoformat(base_date)
    offset = timedelta(hours=random.randint(0, hours_back))
    return (base - offset).isoformat() + 'Z'

def generate_base_state():
    """Generate randomized base state with 5 items in each category."""
    
    # 5 random status pages - all healthy
    services = ['stripe', 'aws-s3', 'database', 'redis', 'cdn', 'api-gateway', 'auth-service', 'notification-service', 'load-balancer', 'monitoring']
    status_pages = []
    for service in random.sample(services, 5):
        status_pages.append({
            'service': service,
            'status': 'healthy', 
            'message': ''
        })
    
    # 5 random slack messages - all normal chatter
    channels = ['general', 'engineering', 'support', 'infra-eng', 'alerts', 'ops', 'backend-team']
    users = ['alice.dev', 'bob.ops', 'charlie.support', 'monitoring_bot', 'deploy_bot', 'sarah.backend', 'mike.frontend']
    normal_messages = [
        'Daily standup in 10 minutes',
        'Code review ready for merge', 
        'System health check complete',
        'Planning meeting moved to 3pm',
        'New monitoring dashboard deployed'
    ]
    
    slack_messages = []
    for i in range(5):
        slack_messages.append({
            'channel': random.choice(channels),
            'datetime': generate_random_datetime(hours_back=48),
            'user': random.choice(users),
            'message': random.choice(normal_messages)
        })
    
    # 5 random deployments - all successful
    deploy_services = ['user-service', 'payment-service', 'api-gateway', 'notification-service', 'auth-service', 'search-service', 'analytics-service']
    deployments = []
    for i in range(5):
        deployments.append({
            'datetime': generate_random_datetime(hours_back=72),
            'service': random.choice(deploy_services),
            'version': f'{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}',
            'succeeded': True
        })
    
    # 5 random logs - all normal INFO logs
    info_messages = [
        'Request processed successfully',
        'Database connection established', 
        'Cache hit ratio: 85%',
        'User authentication successful',
        'Health check passed'
    ]
    
    logs = []
    for i in range(5):
        logs.append({
            'datetime': generate_random_datetime(hours_back=6),
            'message': f'INFO: {random.choice(info_messages)}',
            'response_time': random.randint(50, 200)
        })
    
    return {
        'status_pages': status_pages,
        'slack': slack_messages, 
        'deployments': deployments,
        'logs': logs
    }

def inject_issue(state, issue_type):
    """Inject a single issue into the randomized base state."""
    
    if issue_type == 'status_page':
        # Pick random status page and make it unhealthy
        idx = random.randint(0, 4)
        service = state['status_pages'][idx]['service']
        status = random.choice(['degraded', 'down', 'critical'])
        
        state['status_pages'][idx]['status'] = status
        state['status_pages'][idx]['message'] = f'{service} experiencing {status} performance'
        
        title = f'{service.title()} Service Issues'
        answer = f'{service} service is experiencing {status} performance'
        
    elif issue_type == 'slack':
        # Add problematic slack message
        problem_messages = [
            'Users cant log in - getting timeout errors',
            'Checkout process hanging for customers', 
            'API responses are super slow',
            'Email notifications not going out',
            'Search taking forever to return results'
        ]
        
        state['slack'].append({
            'channel': random.choice(['support', 'alerts', 'engineering']),
            'datetime': generate_random_datetime(hours_back=2),
            'user': random.choice(['support_team', 'alerts_bot', 'user_reports']),
            'message': random.choice(problem_messages)
        })
        
        title = 'User Reported Service Issues'
        answer = 'Service issues reported by users through support channels'
        
    elif issue_type == 'deployment':
        # Add failed deployment
        failed_services = ['payment-service', 'user-service', 'api-gateway', 'auth-service', 'notification-service']
        
        state['deployments'].append({
            'datetime': generate_random_datetime(hours_back=4),
            'service': random.choice(failed_services),
            'version': f'{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}',
            'succeeded': False
        })
        
        title = 'Service Deployment Failure'
        answer = 'Recent deployment failed and may have introduced service issues'
        
    elif issue_type == 'logs':
        # Add error logs
        error_types = [
            'OutOfMemoryError: Java heap space exceeded',
            'ConcurrentModificationException in order processing',
            'SQLException: Connection timeout after 30s',
            'ThreadPoolExecutor: Queue capacity exceeded',
            'NullPointerException in payment validation'
        ]
        
        error_msg = random.choice(error_types)
        state['logs'].append({
            'datetime': generate_random_datetime(hours_back=1),
            'message': f'ERROR: {error_msg}',
            'response_time': random.randint(5000, 30000)
        })
        
        title = 'Application Error Detected'
        answer = f'Application error requiring investigation: {error_msg}'
    
    return title, answer

def generate_incident():
    """Generate a single incident with randomized base state + injected issue."""
    
    # Generate random base state
    state = generate_base_state()
    
    # Sample issue type
    issue_types = [('status_page', 0.5), ('slack', 0.2), ('deployment', 0.2), ('logs', 0.1)]
    issue_type = random.choices([t[0] for t in issue_types], weights=[t[1] for t in issue_types])[0]
    
    # Inject the issue
    title, answer = inject_issue(state, issue_type)
    
    return {
        'question': f"INCIDENT ALERT - INC-{random.randint(10000, 99999)}\nTitle: {title}\nUsers reporting service disruptions.",
        'answer': answer,
        'state': state,
        'issue_type': issue_type
    }

def generate_dataset(n=1000):
    return Dataset.from_list([generate_incident() for _ in range(n)])

# Generate dataset
dataset = generate_dataset(1000)
print("Generated dataset with fully randomized base states")
#dataset.save_to_disk("randomized_incidents")
dataset.push_to_hub("cdreetz/on-call-agent-grpo-dataset")