import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import json
from typing import Optional, Dict, Any
import asyncio
import logging
from enum import Enum
import ipaddress
import re

# Configuración de logging
logger = logging.getLogger(__name__)

class AuthContext(Enum):
    LOGIN = "login"
    API_ACCESS = "api_access"
    SENSITIVE_OPERATION = "sensitive_operation"
    PAYMENT = "payment"

class ZeroTrustSecurity:
    def __init__(self, secret_key: str, pepper: Optional[str] = None):
        self.secret_key = secret_key.encode()
        self.pepper = pepper.encode() if pepper else os.urandom(16)
        self.cipher_suite = Fernet(Fernet.generate_key())
        
        # Para derivación de claves más segura
        self.kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=32,
            salt=self.pepper,
            iterations=100000,
        )
    
    def generate_secure_token(self, user_id: str, action: str, 
                            context: AuthContext = AuthContext.API_ACCESS,
                            additional_data: Optional[Dict] = None) -> str:
        """
        Genera un token seguro con contexto y timestamp
        """
        timestamp = datetime.utcnow().timestamp()
        nonce = os.urandom(8).hex()
        
        # Estructura de datos del token
        token_data = {
            "user_id": user_id,
            "action": action,
            "context": context.value,
            "timestamp": timestamp,
            "nonce": nonce
        }
        
        if additional_data:
            token_data["additional"] = additional_data
        
        # Convertir a JSON y cifrar
        json_data = json.dumps(token_data, sort_keys=True)
        data_to_sign = f"{json_data}:{self.pepper.hex()}"
        
        # Crear HMAC
        signature = hmac.new(
            self.secret_key, 
            data_to_sign.encode(), 
            hashlib.sha3_256
        ).digest()
        
        # Combinar datos y firma
        combined = f"{json_data}:{base64.b64encode(signature).decode()}"
        
        # Cifrar todo
        encrypted = self.cipher_suite.encrypt(combined.encode())
        return base64.b64encode(encrypted).decode()
    
    def verify_token(self, token: str, user_id: str, action: str, 
                    context: AuthContext = AuthContext.API_ACCESS,
                    max_age_seconds: int = 300) -> bool:
        """
        Verifica un token con validación de contexto y tiempo
        """
        try:
            # Descifrar token
            decrypted = self.cipher_suite.decrypt(base64.b64decode(token))
            parts = decrypted.decode().split(':')
            
            if len(parts) < 2:
                return False
            
            json_data = ':'.join(parts[:-1])
            received_signature = base64.b64decode(parts[-1])
            
            # Verificar firma
            expected_signature = hmac.new(
                self.secret_key, 
                f"{json_data}:{self.pepper.hex()}".encode(), 
                hashlib.sha3_256
            ).digest()
            
            if not hmac.compare_digest(received_signature, expected_signature):
                return False
            
            # Parsear datos
            token_data = json.loads(json_data)
            
            # Verificar coincidencia
            if (token_data.get('user_id') != user_id or 
                token_data.get('action') != action or
                token_data.get('context') != context.value):
                return False
            
            # Verificar timestamp
            token_time = token_data.get('timestamp', 0)
            current_time = datetime.utcnow().timestamp()
            
            if current_time - token_time > max_age_seconds:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return False
    
    def extract_token_data(self, token: str) -> Optional[Dict]:
        """
        Extrae datos de un token sin verificar (solo para uso interno)
        """
        try:
            decrypted = self.cipher_suite.decrypt(base64.b64decode(token))
            parts = decrypted.decode().split(':')
            json_data = ':'.join(parts[:-1])
            return json.loads(json_data)
        except:
            return None
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Cifra datos sensibles con salting adicional"""
        # Añadir salt único para cada cifrado
        salt = os.urandom(8)
        data_to_encrypt = f"{salt.hex()}:{data}"
        
        encrypted = self.cipher_suite.encrypt(data_to_encrypt.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Optional[str]:
        """Descifra datos sensibles"""
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher_suite.decrypt(decoded).decode()
            # Extraer y eliminar el salt
            return decrypted.split(':', 1)[1]
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    async def continuous_authentication(self, user_session: dict) -> bool:
        """
        Autenticación continua con múltiples factores de verificación
        """
        verification_tasks = [
            self.analyze_behavioral_biometrics(user_session),
            self.analyze_network_patterns(user_session),
            self.verify_contextual_integrity(user_session),
            self.check_device_fingerprint(user_session),
            self.verify_geolocation(user_session)
        ]
        
        results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        
        # Ponderar resultados
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        weighted_sum = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Verification task failed: {result}")
                result = 0  # Fallo seguro
            weighted_sum += result * weights[i]
        
        logger.info(f"Continuous authentication score: {weighted_sum:.2f}")
        return weighted_sum > 0.75  # Threshold ajustable
    
    async def analyze_behavioral_biometrics(self, user_session: dict) -> float:
        """Analiza biométricos conductuales avanzados"""
        try:
            # Análisis de patrones de tecleo (timing entre teclas)
            typing_patterns = user_session.get('typing_patterns', {})
            typing_score = self._analyze_typing_patterns(typing_patterns)
            
            # Análisis de patrones de mouse
            mouse_patterns = user_session.get('mouse_patterns', {})
            mouse_score = self._analyze_mouse_patterns(mouse_patterns)
            
            # Análisis de patrones de navegación
            navigation_patterns = user_session.get('navigation_patterns', {})
            navigation_score = self._analyze_navigation_patterns(navigation_patterns)
            
            # Combinar scores con pesos
            return (typing_score * 0.4 + mouse_score * 0.3 + navigation_score * 0.3)
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed: {e}")
            return 0.5  # Valor neutral en caso de error
    
    def _analyze_typing_patterns(self, patterns: Dict) -> float:
        """Analiza patrones de tecleo"""
        # Implementación real usaría machine learning
        # Esta es una implementación simplificada
        avg_speed = patterns.get('avg_typing_speed', 0)
        consistency = patterns.get('consistency', 0)
        
        if 200 <= avg_speed <= 400:  # Caracteres por minuto típicos
            speed_score = 0.8
        else:
            speed_score = 0.3
        
        return (speed_score * 0.6 + consistency * 0.4)
    
    def _analyze_mouse_patterns(self, patterns: Dict) -> float:
        """Analiza patrones de mouse"""
        velocity = patterns.get('velocity', 0)
        acceleration = patterns.get('acceleration', 0)
        
        # Valores típicos para comportamiento humano
        if 0.1 <= velocity <= 5.0 and -2.0 <= acceleration <= 2.0:
            return 0.8
        return 0.3
    
    def _analyze_navigation_patterns(self, patterns: Dict) -> float:
        """Analiza patrones de navegación"""
        page_transition_times = patterns.get('page_transition_times', [])
        if not page_transition_times:
            return 0.5
        
        # Tiempos de transición típicos (segundos)
        typical_times = [1.5, 2.0, 3.0, 2.5, 1.8]
        avg_time = sum(page_transition_times) / len(page_transition_times)
        
        if 1.0 <= avg_time <= 4.0:
            return 0.7
        return 0.3
    
    def analyze_network_patterns(self, user_session: dict) -> float:
        """Analiza patrones de red avanzados"""
        try:
            ip_address = user_session.get('ip_address', '')
            user_agent = user_session.get('user_agent', '')
            asn = user_session.get('asn', '')
            
            score = 0.5  # Puntuación base
            
            # Verificar IP (pública vs privada, rangos sospechosos)
            if ip_address:
                try:
                    ip = ipaddress.ip_address(ip_address)
                    if ip.is_private:
                        score += 0.2  # IP privada es más confiable
                    elif str(ip).startswith(('10.', '192.168.', '172.')):
                        score += 0.1
                except ValueError:
                    score -= 0.2
            
            # Verificar User-Agent
            if user_agent:
                if any(browser in user_agent.lower() for browser in 
                      ['chrome', 'firefox', 'safari', 'edge']):
                    score += 0.1
                if 'headless' in user_agent.lower() or 'bot' in user_agent.lower():
                    score -= 0.3
            
            # Verificar ASN (si está disponible)
            if asn and 'google' in asn.lower():
                score += 0.1  # Google ASN es generalmente confiable
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            return 0.5
    
    def verify_contextual_integrity(self, user_session: dict) -> float:
        """Verificación de integridad contextual avanzada"""
        try:
            current_time = datetime.utcnow()
            login_time = user_session.get('login_time')
            last_activity = user_session.get('last_activity')
            requested_resource = user_session.get('requested_resource', '')
            user_role = user_session.get('user_role', 'user')
            
            score = 0.5
            
            # Verificar tiempo desde el login
            if login_time:
                login_delta = (current_time - login_time).total_seconds()
                if login_delta > 86400:  # Más de 24 horas
                    score -= 0.2
                elif login_delta < 300:  # Menos de 5 minutos
                    score += 0.1
            
            # Verificar inactividad
            if last_activity:
                inactivity = (current_time - last_activity).total_seconds()
                if inactivity > 3600:  # Más de 1 hora inactivo
                    score -= 0.3
                elif inactivity < 30:  # Muy reciente
                    score += 0.1
            
            # Verificar acceso a recursos sensibles
            sensitive_resources = ['admin', 'payment', 'config', 'database']
            if any(resource in requested_resource for resource in sensitive_resources):
                if user_role != 'admin':
                    score -= 0.4
                else:
                    score += 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Context verification failed: {e}")
            return 0.5
    
    async def check_device_fingerprint(self, user_session: dict) -> float:
        """Verifica la huella digital del dispositivo"""
        # Implementación simplificada
        expected_fingerprint = user_session.get('expected_device_fingerprint')
        current_fingerprint = user_session.get('current_device_fingerprint')
        
        if expected_fingerprint and current_fingerprint:
            if expected_fingerprint == current_fingerprint:
                return 0.9
            else:
                logger.warning("Device fingerprint mismatch")
                return 0.2
        
        return 0.5  # Valor neutral cuando no hay datos
    
    async def verify_geolocation(self, user_session: dict) -> float:
        """Verifica la geolocalización"""
        expected_location = user_session.get('expected_location')
        current_location = user_session.get('current_location')
        
        if expected_location and current_location:
            # Calcular distancia entre ubicaciones
            distance = self._calculate_distance(expected_location, current_location)
            
            if distance < 50:  # Menos de 50 km
                return 0.9
            elif distance < 500:  # Menos de 500 km
                return 0.7
            else:
                logger.warning(f"Large geographical distance: {distance} km")
                return 0.3
        
        return 0.5  # Valor neutral cuando no hay datos
    
    def _calculate_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calcula distancia entre dos puntos geográficos (simplificado)"""
        # Implementación simplificada - usaría haversine en producción
        lat1, lon1 = loc1.get('lat', 0), loc1.get('lon', 0)
        lat2, lon2 = loc2.get('lat', 0), loc2.get('lon', 0)
        
        # Fórmula simplificada para demostración
        return abs(lat1 - lat2) * 110 + abs(lon1 - lon2) * 110  # Aprox km por grado

# Ejemplo de uso
async def main():
    # Inicializar con una clave secreta (debería estar en variables de entorno)
    zt = ZeroTrustSecurity("my-super-secret-key-12345")
    
    # Generar token
    token = zt.generate_secure_token(
        "user123", 
        "access_resource", 
        AuthContext.API_ACCESS,
        {"resource": "admin_panel"}
    )
    print(f"Generated token: {token}")
    
    # Verificar token
    is_valid = zt.verify_token(
        token, 
        "user123", 
        "access_resource", 
        AuthContext.API_ACCESS
    )
    print(f"Token valid: {is_valid}")
    
    # Cifrar datos
    encrypted = zt.encrypt_sensitive_data("my-sensitive-data")
    print(f"Encrypted: {encrypted}")
    
    # Descifrar datos
    decrypted = zt.decrypt_sensitive_data(encrypted)
    print(f"Decrypted: {decrypted}")
    
    # Autenticación continua
    session_data = {
        "typing_patterns": {"avg_typing_speed": 280, "consistency": 0.8},
        "mouse_patterns": {"velocity": 2.5, "acceleration": 0.5},
        "navigation_patterns": {"page_transition_times": [1.8, 2.2, 1.5]},
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "login_time": datetime.utcnow() - timedelta(hours=2),
        "last_activity": datetime.utcnow() - timedelta(minutes=5),
        "requested_resource": "user_profile",
        "user_role": "user"
    }
    
    is_authenticated = await zt.continuous_authentication(session_data)
    print(f"Continuous authentication: {is_authenticated}")

if __name__ == "__main__":
    asyncio.run(main())