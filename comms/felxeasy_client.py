import requests
import json
import time
from typing import Dict, Any

class FelxEasyClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def register_robot(self, user_id: str, driver_data: Dict) -> bool:
        """Registra esta instancia robot en FelxEasy"""
        try:
            payload = {
                "userId": user_id,
                "driverData": driver_data,
                "status": "online"
            }
            response = self.session.post(
                f"{self.base_url}/api/robots/register",
                json=payload
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error registrando robot: {e}")
            return False
    
    def get_commands(self, user_id: str) -> Dict[str, Any]:
        """Obtiene comandos pendientes desde FelxEasy"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/robots/commands",
                params={"userId": user_id}
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"❌ Error obteniendo comandos: {e}")
            return {}
    
    def send_status(self, user_id: str, status_data: Dict) -> bool:
        """Envía estado y métricas a FelxEasy"""
        try:
            payload = {
                "userId": user_id,
                "status": status_data.get("status", "running"),
                "metrics": status_data.get("metrics", {}),
                "lastPing": int(time.time()),
                "capturedBlocks": status_data.get("captured_blocks", [])
            }
            
            response = self.session.post(
                f"{self.base_url}/api/robots/status",
                json=payload
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error enviando estado: {e}")
            return False
    
    def report_block(self, user_id: str, block_data: Dict) -> bool:
        """Reporta un bloque capturado exitosamente"""
        try:
            payload = {
                "userId": user_id,
                "blockData": block_data
            }
            response = self.session.post(
                f"{self.base_url}/api/robots/blocks",
                json=payload
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error reportando bloque: {e}")
            return False