from seleniumwire import webdriver
from selenium.webdriver.chrome.options import Options
import json
import time

def capture_amazon_flex_traffic():
    chrome_options = Options()
    chrome_options.add_argument("--user-data-dir=./user_data")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navegar a Amazon Flex
        driver.get("https://flex.amazon.com")
        
        # Esperar login manual (hacerlo manualmente)
        input("👉 Haz login manualmente y presiona Enter...")
        
        # Capturar requests
        requests_data = []
        for request in driver.requests:
            if request.response:
                requests_data.append({
                    'url': request.url,
                    'method': request.method,
                    'headers': dict(request.headers),
                    'body': request.body,
                    'response_status': request.response.status_code,
                    'response_headers': dict(request.response.headers),
                    'response_body': request.response.body.decode() if request.response.body else None
                })
        
        # Guardar captura
        with open('amazon_flex_requests.json', 'w') as f:
            json.dump(requests_data, f, indent=2)
            
        print("✅ Captura completada. Revisa amazon_flex_requests.json")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    capture_amazon_flex_traffic()