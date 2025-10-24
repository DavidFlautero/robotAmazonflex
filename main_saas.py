import time
import threading
from comms.felxeasy_client import FelxEasyClient
from typing import Dict, Any
from config_saas import SaaSConfig

class FlexRobotWithSaaS:
    def __init__(self, user_id: str, saas_url: str):
        self.user_id = user_id
        self.saas_client = FelxEasyClient(saas_url)
        self.is_running = False
        self.status = {
            "status": "offline",
            "blocks_captured": 0,
            "last_activity": None,
            "errors": 0
        }
    
    def start(self):
        """Inicia el robot integrado con FelxEasy"""
        print(f"🚀 Iniciando robot para usuario: {self.user_id}")
        
        # Registrar robot en FelxEasy
        driver_data = {
            "platform": "amazon_flex",
            "version": "2.0",
            "capabilities": ["block_scanning", "auto_accept"]
        }
        
        if self.saas_client.register_robot(self.user_id, driver_data):
            print("✅ Robot registrado en FelxEasy")
            self.is_running = True
            
            # Iniciar hilos de trabajo
            threading.Thread(target=self._command_listener, daemon=True).start()
            threading.Thread(target=self._status_reporter, daemon=True).start()
            threading.Thread(target=self._main_worker, daemon=True).start()
            
        else:
            print("❌ No se pudo registrar el robot")
    
    def _command_listener(self):
        """Escucha comandos desde FelxEasy"""
        while self.is_running:
            try:
                commands = self.saas_client.get_commands(self.user_id)
                if commands:
                    self._execute_commands(commands)
                time.sleep(10)  # Consultar cada 10 segundos
            except Exception as e:
                print(f"Error en command listener: {e}")
                time.sleep(30)
    
    def _status_reporter(self):
        """Reporta estado periódicamente a FelxEasy"""
        while self.is_running:
            try:
                self.saas_client.send_status(self.user_id, self.status)
                time.sleep(30)  # Reportar cada 30 segundos
            except Exception as e:
                print(f"Error en status reporter: {e}")
                time.sleep(60)
    
    def _execute_commands(self, commands: Dict):
        """Ejecuta comandos recibidos desde FelxEasy"""
        for command in commands.get("commands", []):
            action = command.get("action")
            
            if action == "start_scanning":
                print("▶️ Comando: Iniciar escaneo")
                # Aquí integrarías tu lógica actual de escaneo
                
            elif action == "stop_scanning":
                print("⏹️ Comando: Detener escaneo")
                self.is_running = False
                
            elif action == "update_preferences":
                print("⚙️ Comando: Actualizar preferencias")
                # Actualizar configuración según comandos
    
    def _main_worker(self):
        """Tu lógica principal de escaneo de bloques (adaptar de tu código actual)"""
        while self.is_running:
            try:
                # TODO: Integrar aquí tu lógica actual de:
                # - Login Amazon Flex
                # - Escaneo de bloques
                # - Aceptación automática
                
                print("🔍 Escaneando bloques...")
                time.sleep(5)  # Simulación
                
                # Ejemplo: cuando captures un bloque
                block_data = {
                    "block_id": "B12345",
                    "amount": 18.50,
                    "location": "Madrid Centro",
                    "schedule": "15:00-19:00",
                    "captured_at": int(time.time())
                }
                self.saas_client.report_block(self.user_id, block_data)
                self.status["blocks_captured"] += 1
                
            except Exception as e:
                print(f"Error en main worker: {e}")
                self.status["errors"] += 1
                time.sleep(10)

# Uso del nuevo sistema
if __name__ == "__main__":
    # Configuración desde variables de entorno
    SaaSConfig.validate_config()
    
    robot = FlexRobotWithSaaS(
        SaaSConfig.USER_ID, 
        SaaSConfig.SAAS_BASE_URL
    )
    robot.start()
    
    # Mantener el programa corriendo
    try:
        while robot.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Deteniendo robot...")
        robot.is_running = False