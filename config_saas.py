import os
from dotenv import load_dotenv

load_dotenv()

class SaaSConfig:
    # Configuración de FelxEasy
    SAAS_BASE_URL = os.getenv('FELXEASY_URL', 'https://autoflexeasy.com')
    USER_ID = os.getenv('FELXEASY_USER_ID')
    
    @classmethod
    def validate_config(cls):
        """Valida que la configuración esté completa"""
        missing = []
        if not cls.SAAS_BASE_URL:
            missing.append('FELXEASY_URL')
        if not cls.USER_ID:
            missing.append('FELXEASY_USER_ID')
            
        if missing:
            raise ValueError(f"Configuración faltante: {', '.join(missing)}")