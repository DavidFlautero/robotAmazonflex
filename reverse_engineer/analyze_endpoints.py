import json
from urllib.parse import urlparse, parse_qs

def analyze_captured_data():
    with open('amazon_flex_requests.json', 'r') as f:
        requests = json.load(f)
    
    # Filtrar endpoints importantes
    api_endpoints = []
    for req in requests:
        if 'api.amazon.com' in req['url'] or 'flex.amazon' in req['url']:
            endpoint = {
                'url': req['url'],
                'method': req['method'],
                'headers': req['headers'],
                'likely_function': identify_function(req['url'], req['body'])
            }
            api_endpoints.append(endpoint)
    
    # Guardar análisis
    with open('api_analysis.json', 'w') as f:
        json.dump(api_endpoints, f, indent=2)
    
    print("✅ Análisis completado. Revisa api_analysis.json")

def identify_function(url, body):
    """Identificar función del endpoint basado en URL y body"""
    if 'offers' in url and 'POST' in method:
        return 'get_available_blocks'
    elif 'accept' in url and 'POST' in method:
        return 'accept_block'
    elif 'schedule' in url:
        return 'get_schedule'
    elif 'auth' in url:
        return 'authentication'
    return 'unknown'

if __name__ == "__main__":
    analyze_captured_data()