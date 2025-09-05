import os
import base64
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag

# Configuración de logging
logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    AES_GCM = "aes-gcm"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    RSA_OAEP = "rsa-oaep"
    HYBRID = "hybrid"

class KeyManagerType(Enum):
    LOCAL = "local"
    AWS_KMS = "aws-kms"
    HASHICORP_VAULT = "hashicorp-vault"
    AZURE_KEY_VAULT = "azure-key-vault"

@dataclass
class EncryptionResult:
    ciphertext: bytes
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    encrypted_key: Optional[bytes] = None
    algorithm: Optional[EncryptionAlgorithm] = None
    timestamp: Optional[datetime] = None

@dataclass
class KeyMetadata:
    key_id: str
    version: int
    creation_date: datetime
    expiration_date: Optional[datetime] = None
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_GCM
    enabled: bool = True

class AdvancedEncryptionManager:
    """
    Sistema avanzado de cifrado con soporte para múltiples algoritmos,
    gestión de claves y rotación automática.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.keys = {}
        self.current_key_id = None
        self.key_manager_type = KeyManagerType(config.get("key_manager_type", "local"))
        self.backend = default_backend()
        
        # Inicializar el gestor de claves según la configuración
        self.key_manager = self._initialize_key_manager()
        
        # Cargar claves existentes o generar nuevas
        self._load_or_generate_keys()
        
        # Iniciar tarea de rotación automática de claves
        self.rotation_task = asyncio.create_task(self._key_rotation_loop())
    
    def _initialize_key_manager(self):
        """Inicializar el gestor de claves según el tipo configurado"""
        key_manager_type = self.key_manager_type
        
        if key_manager_type == KeyManagerType.LOCAL:
            return LocalKeyManager(self.config)
        elif key_manager_type == KeyManagerType.AWS_KMS:
            return AWSKMSManager(self.config)
        elif key_manager_type == KeyManagerType.HASHICORP_VAULT:
            return HashiCorpVaultManager(self.config)
        elif key_manager_type == KeyManagerType.AZURE_KEY_VAULT:
            return AzureKeyVaultManager(self.config)
        else:
            raise ValueError(f"Tipo de gestor de claves no soportado: {key_manager_type}")
    
    def _load_or_generate_keys(self):
        """Cargar claves existentes o generar nuevas si es necesario"""
        try:
            loaded_keys = self.key_manager.load_keys()
            if loaded_keys:
                self.keys = loaded_keys
                # Usar la clave más reciente
                self.current_key_id = max(self.keys.keys(), key=lambda k: self.keys[k].creation_date)
                logger.info(f"Claves cargadas exitosamente. Clave actual: {self.current_key_id}")
            else:
                self._generate_new_key()
        except Exception as e:
            logger.error(f"Error al cargar claves: {e}")
            self._generate_new_key()
    
    def _generate_new_key(self):
        """Generar una nueva clave maestra"""
        key_id = f"key_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if self.key_manager_type == KeyManagerType.LOCAL:
            # Generar clave AES de 256 bits
            key = os.urandom(32)
            key_metadata = KeyMetadata(
                key_id=key_id,
                version=1,
                creation_date=datetime.utcnow(),
                expiration_date=datetime.utcnow() + timedelta(days=90),
                algorithm=EncryptionAlgorithm.AES_GCM
            )
            
            self.keys[key_id] = (key, key_metadata)
            self.current_key_id = key_id
            
            # Guardar la nueva clave
            self.key_manager.save_keys(self.keys)
            logger.info(f"Nueva clave generada: {key_id}")
        else:
            # Para gestores externos, solicitar una nueva clave
            key_metadata = self.key_manager.generate_key(key_id)
            self.keys[key_id] = (None, key_metadata)  # La clave se maneja externamente
            self.current_key_id = key_id
            logger.info(f"Nueva clave solicitada al gestor externo: {key_id}")
    
    async def _key_rotation_loop(self):
        """Bucle para rotación automática de claves"""
        rotation_interval = self.config.get("key_rotation_interval", 86400)  # Por defecto 24 horas
        
        while True:
            try:
                await asyncio.sleep(rotation_interval)
                self.rotate_keys()
            except Exception as e:
                logger.error(f"Error en el bucle de rotación de claves: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos antes de reintentar
    
    def rotate_keys(self):
        """Rotar las claves según la política configurada"""
        current_time = datetime.utcnow()
        
        # Verificar si la clave actual está cerca de expirar
        current_key_metadata = self.keys[self.current_key_id][1]
        if current_key_metadata.expiration_date and current_key_metadata.expiration_date <= current_time + timedelta(days=7):
            logger.info(f"La clave actual {self.current_key_id} expira pronto. Generando nueva clave.")
            self._generate_new_key()
        
        # Eliminar claves expiradas
        expired_keys = [
            key_id for key_id, (key, metadata) in self.keys.items()
            if metadata.expiration_date and metadata.expiration_date <= current_time
        ]
        
        for key_id in expired_keys:
            del self.keys[key_id]
            logger.info(f"Clave expirada eliminada: {key_id}")
        
        # Guardar cambios
        self.key_manager.save_keys(self.keys)
    
    def encrypt(self, plaintext: Union[str, bytes], algorithm: EncryptionAlgorithm = None) -> EncryptionResult:
        """
        Cifrar datos usando el algoritmo especificado
        """
        if not plaintext:
            raise ValueError("El texto plano no puede estar vacío")
        
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        algorithm = algorithm or EncryptionAlgorithm.AES_GCM
        
        try:
            if algorithm == EncryptionAlgorithm.AES_GCM:
                return self._encrypt_aes_gcm(plaintext)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return self._encrypt_chacha20(plaintext)
            elif algorithm == EncryptionAlgorithm.RSA_OAEP:
                return self._encrypt_rsa(plaintext)
            elif algorithm == EncryptionAlgorithm.HYBRID:
                return self._encrypt_hybrid(plaintext)
            else:
                raise ValueError(f"Algoritmo no soportado: {algorithm}")
        except Exception as e:
            logger.error(f"Error al cifrar datos: {e}")
            raise
    
    def decrypt(self, encryption_result: EncryptionResult) -> bytes:
        """
        Descifrar datos usando la información del resultado de cifrado
        """
        try:
            if encryption_result.algorithm == EncryptionAlgorithm.AES_GCM:
                return self._decrypt_aes_gcm(encryption_result)
            elif encryption_result.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return self._decrypt_chacha20(encryption_result)
            elif encryption_result.algorithm == EncryptionAlgorithm.RSA_OAEP:
                return self._decrypt_rsa(encryption_result)
            elif encryption_result.algorithm == EncryptionAlgorithm.HYBRID:
                return self._decrypt_hybrid(encryption_result)
            else:
                raise ValueError(f"Algoritmo no soportado: {encryption_result.algorithm}")
        except Exception as e:
            logger.error(f"Error al descifrar datos: {e}")
            raise
    
    def _encrypt_aes_gcm(self, plaintext: bytes) -> EncryptionResult:
        """Cifrado AES-GCM"""
        key, metadata = self.keys[self.current_key_id]
        iv = os.urandom(12)
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return EncryptionResult(
            ciphertext=ciphertext,
            iv=iv,
            tag=encryptor.tag,
            algorithm=EncryptionAlgorithm.AES_GCM,
            timestamp=datetime.utcnow()
        )
    
    def _decrypt_aes_gcm(self, encryption_result: EncryptionResult) -> bytes:
        """Descifrado AES-GCM"""
        key, metadata = self.keys[self.current_key_id]
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(encryption_result.iv, encryption_result.tag), backend=self.backend)
        decryptor = cipher.decryptor()
        
        return decryptor.update(encryption_result.ciphertext) + decryptor.finalize()
    
    def _encrypt_chacha20(self, plaintext: bytes) -> EncryptionResult:
        """Cifrado ChaCha20-Poly1305"""
        key, metadata = self.keys[self.current_key_id]
        nonce = os.urandom(12)
        
        cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=self.backend)
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(plaintext)
        
        return EncryptionResult(
            ciphertext=ciphertext,
            iv=nonce,
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
            timestamp=datetime.utcnow()
        )
    
    def _decrypt_chacha20(self, encryption_result: EncryptionResult) -> bytes:
        """Descifrado ChaCha20-Poly1305"""
        key, metadata = self.keys[self.current_key_id]
        
        cipher = Cipher(algorithms.ChaCha20(key, encryption_result.iv), mode=None, backend=self.backend)
        decryptor = cipher.decryptor()
        
        return decryptor.update(encryption_result.ciphertext)
    
    def _encrypt_rsa(self, plaintext: bytes) -> EncryptionResult:
        """Cifrado RSA-OAEP"""
        # Para RSA, necesitamos la clave pública
        public_key = self.key_manager.get_public_key()
        
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return EncryptionResult(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.RSA_OAEP,
            timestamp=datetime.utcnow()
        )
    
    def _decrypt_rsa(self, encryption_result: EncryptionResult) -> bytes:
        """Descifrado RSA-OAEP"""
        # Para RSA, necesitamos la clave privada
        private_key = self.key_manager.get_private_key()
        
        return private_key.decrypt(
            encryption_result.ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def _encrypt_hybrid(self, plaintext: bytes) -> EncryptionResult:
        """
        Cifrado híbrido: usa AES para los datos y RSA para la clave AES
        """
        # Generar una clave AES aleatoria para esta operación
        session_key = os.urandom(32)
        iv = os.urandom(12)
        
        # Cifrar los datos con AES
        cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Cifrar la clave de sesión con RSA
        public_key = self.key_manager.get_public_key()
        encrypted_key = public_key.encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return EncryptionResult(
            ciphertext=ciphertext,
            iv=iv,
            tag=encryptor.tag,
            encrypted_key=encrypted_key,
            algorithm=EncryptionAlgorithm.HYBRID,
            timestamp=datetime.utcnow()
        )
    
    def _decrypt_hybrid(self, encryption_result: EncryptionResult) -> bytes:
        """Descifrado híbrido"""
        # Descifrar la clave de sesión con RSA
        private_key = self.key_manager.get_private_key()
        session_key = private_key.decrypt(
            encryption_result.encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Descifrar los datos con AES
        cipher = Cipher(algorithms.AES(session_key), modes.GCM(encryption_result.iv, encryption_result.tag), backend=self.backend)
        decryptor = cipher.decryptor()
        
        return decryptor.update(encryption_result.ciphertext) + decryptor.finalize()
    
    def encrypt_sensitive_data(self, data: Dict) -> Dict:
        """
        Cifrar datos sensibles en un diccionario
        """
        encrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, (str, bytes)):
                # Cifrar valores sensibles
                encryption_result = self.encrypt(value)
                encrypted_data[key] = {
                    'ciphertext': base64.b64encode(encryption_result.ciphertext).decode('utf-8'),
                    'iv': base64.b64encode(encryption_result.iv).decode('utf-8') if encryption_result.iv else None,
                    'tag': base64.b64encode(encryption_result.tag).decode('utf-8') if encryption_result.tag else None,
                    'encrypted_key': base64.b64encode(encryption_result.encrypted_key).decode('utf-8') if encryption_result.encrypted_key else None,
                    'algorithm': encryption_result.algorithm.value,
                    'key_id': self.current_key_id,
                    'timestamp': encryption_result.timestamp.isoformat() if encryption_result.timestamp else None
                }
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    def decrypt_sensitive_data(self, encrypted_data: Dict) -> Dict:
        """
        Descifrar datos sensibles en un diccionario
        """
        decrypted_data = {}
        
        for key, value in encrypted_data.items():
            if isinstance(value, dict) and 'ciphertext' in value:
                # Descifrar valores cifrados
                encryption_result = EncryptionResult(
                    ciphertext=base64.b64decode(value['ciphertext']),
                    iv=base64.b64decode(value['iv']) if value['iv'] else None,
                    tag=base64.b64decode(value['tag']) if value['tag'] else None,
                    encrypted_key=base64.b64decode(value['encrypted_key']) if value['encrypted_key'] else None,
                    algorithm=EncryptionAlgorithm(value['algorithm']),
                    timestamp=datetime.fromisoformat(value['timestamp']) if value['timestamp'] else None
                )
                
                decrypted_data[key] = self.decrypt(encryption_result).decode('utf-8')
            else:
                decrypted_data[key] = value
        
        return decrypted_data
    
    async def close(self):
        """Cerrar recursos"""
        if self.rotation_task:
            self.rotation_task.cancel()
            try:
                await self.rotation_task
            except asyncio.CancelledError:
                pass
        
        self.key_manager.close()

# Implementaciones de gestores de claves
class LocalKeyManager:
    """Gestor de claves local"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.keys_file = config.get("keys_file", "encryption_keys.json")
        self.private_key_file = config.get("private_key_file", "private_key.pem")
        self.public_key_file = config.get("public_key_file", "public_key.pem")
        self._ensure_key_files()
    
    def _ensure_key_files(self):
        """Asegurarse de que los archivos de claves existan"""
        if not os.path.exists(self.private_key_file) or not os.path.exists(self.public_key_file):
            self._generate_rsa_keys()
    
    def _generate_rsa_keys(self):
        """Generar par de claves RSA"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        # Guardar clave privada
        with open(self.private_key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Guardar clave pública
        with open(self.public_key_file, "wb") as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
    
    def load_keys(self) -> Dict:
        """Cargar claves desde el archivo"""
        if not os.path.exists(self.keys_file):
            return {}
        
        try:
            with open(self.keys_file, 'r') as f:
                keys_data = json.load(f)
            
            keys = {}
            for key_id, key_data in keys_data.items():
                key = base64.b64decode(key_data['key'])
                metadata = KeyMetadata(
                    key_id=key_data['metadata']['key_id'],
                    version=key_data['metadata']['version'],
                    creation_date=datetime.fromisoformat(key_data['metadata']['creation_date']),
                    expiration_date=datetime.fromisoformat(key_data['metadata']['expiration_date']) if key_data['metadata']['expiration_date'] else None,
                    algorithm=EncryptionAlgorithm(key_data['metadata']['algorithm']),
                    enabled=key_data['metadata']['enabled']
                )
                keys[key_id] = (key, metadata)
            
            return keys
        except Exception as e:
            logger.error(f"Error al cargar claves: {e}")
            return {}
    
    def save_keys(self, keys: Dict):
        """Guardar claves en el archivo"""
        keys_data = {}
        for key_id, (key, metadata) in keys.items():
            keys_data[key_id] = {
                'key': base64.b64encode(key).decode('utf-8'),
                'metadata': {
                    'key_id': metadata.key_id,
                    'version': metadata.version,
                    'creation_date': metadata.creation_date.isoformat(),
                    'expiration_date': metadata.expiration_date.isoformat() if metadata.expiration_date else None,
                    'algorithm': metadata.algorithm.value,
                    'enabled': metadata.enabled
                }
            }
        
        with open(self.keys_file, 'w') as f:
            json.dump(keys_data, f, indent=2)
    
    def get_public_key(self):
        """Obtener la clave pública RSA"""
        with open(self.public_key_file, "rb") as f:
            return serialization.load_pem_public_key(f.read(), backend=default_backend())
    
    def get_private_key(self):
        """Obtener la clave privada RSA"""
        with open(self.private_key_file, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())
    
    def close(self):
        """Cerrar recursos"""
        pass

# Implementaciones de gestores de claves externos (simplificadas)
class AWSKMSManager:
    """Gestor de claves AWS KMS (simplificado)"""
    def __init__(self, config: Dict):
        self.config = config
        # Aquí iría la configuración real del cliente AWS KMS
        logger.info("Inicializando gestor de claves AWS KMS")
    
    def load_keys(self):
        """Cargar claves desde AWS KMS"""
        # Implementación real usaría el SDK de AWS
        return {}
    
    def save_keys(self, keys: Dict):
        """Guardar claves en AWS KMS"""
        # Implementación real usaría el SDK de AWS
        pass
    
    def generate_key(self, key_id: str) -> KeyMetadata:
        """Generar una nueva clave en AWS KMS"""
        # Implementación real usaría el SDK de AWS
        return KeyMetadata(
            key_id=key_id,
            version=1,
            creation_date=datetime.utcnow(),
            expiration_date=datetime.utcnow() + timedelta(days=90),
            algorithm=EncryptionAlgorithm.AES_GCM
        )
    
    def get_public_key(self):
        """Obtener clave pública desde AWS KMS"""
        # Implementación real usaría el SDK de AWS
        return None
    
    def get_private_key(self):
        """Obtener clave privada desde AWS KMS"""
        # Implementación real usaría el SDK de AWS
        return None
    
    def close(self):
        """Cerrar cliente AWS"""
        pass

class HashiCorpVaultManager:
    """Gestor de claves HashiCorp Vault (simplificado)"""
    def __init__(self, config: Dict):
        self.config = config
        # Aquí iría la configuración real del cliente de Vault
        logger.info("Inicializando gestor de claves HashiCorp Vault")
    
    def load_keys(self):
        """Cargar claves desde Vault"""
        # Implementación real usaría el cliente de Vault
        return {}
    
    def save_keys(self, keys: Dict):
        """Guardar claves en Vault"""
        # Implementación real usaría el cliente de Vault
        pass
    
    def generate_key(self, key_id: str) -> KeyMetadata:
        """Generar una nueva clave en Vault"""
        # Implementación real usaría el cliente de Vault
        return KeyMetadata(
            key_id=key_id,
            version=1,
            creation_date=datetime.utcnow(),
            expiration_date=datetime.utcnow() + timedelta(days=90),
            algorithm=EncryptionAlgorithm.AES_GCM
        )
    
    def get_public_key(self):
        """Obtener clave pública desde Vault"""
        # Implementación real usaría el cliente de Vault
        return None
    
    def get_private_key(self):
        """Obtener clave privada desde Vault"""
        # Implementación real usaría el cliente de Vault
        return None
    
    def close(self):
        """Cerrar cliente Vault"""
        pass

class AzureKeyVaultManager:
    """Gestor de claves Azure Key Vault (simplificado)"""
    def __init__(self, config: Dict):
        self.config = config
        # Aquí iría la configuración real del cliente de Azure
        logger.info("Inicializando gestor de claves Azure Key Vault")
    
    def load_keys(self):
        """Cargar claves desde Azure Key Vault"""
        # Implementación real usaría el SDK de Azure
        return {}
    
    def save_keys(self, keys: Dict):
        """Guardar claves en Azure Key Vault"""
        # Implementación real usaría el SDK de Azure
        pass
    
    def generate_key(self, key_id: str) -> KeyMetadata:
        """Generar una nueva clave en Azure Key Vault"""
        # Implementación real usaría el SDK de Azure
        return KeyMetadata(
            key_id=key_id,
            version=1,
            creation_date=datetime.utcnow(),
            expiration_date=datetime.utcnow() + timedelta(days=90),
            algorithm=EncryptionAlgorithm.AES_GCM
        )
    
    def get_public_key(self):
        """Obtener clave pública desde Azure Key Vault"""
        # Implementación real usaría el SDK de Azure
        return None
    
    def get_private_key(self):
        """Obtener clave privada desde Azure Key Vault"""
        # Implementación real usaría el SDK de Azure
        return None
    
    def close(self):
        """Cerrar cliente Azure"""
        pass

# Ejemplo de uso
async def main():
    # Configuración
    config = {
        "key_manager_type": "local",
        "keys_file": "encryption_keys.json",
        "private_key_file": "private_key.pem",
        "public_key_file": "public_key.pem",
        "key_rotation_interval": 86400  # 24 horas
    }
    
    # Inicializar el gestor de cifrado
    encryption_manager = AdvancedEncryptionManager(config)
    
    try:
        # Datos sensibles a cifrar
        sensitive_data = {
            "api_key": "tu_api_key_secreta",
            "database_password": "tu_contraseña_de_bd",
            "proxy_credentials": "usuario:contraseña@proxy:puerto"
        }
        
        # Cifrar datos sensibles
        encrypted_data = encryption_manager.encrypt_sensitive_data(sensitive_data)
        print("Datos cifrados:")
        print(json.dumps(encrypted_data, indent=2))
        
        # Descifrar datos
        decrypted_data = encryption_manager.decrypt_sensitive_data(encrypted_data)
        print("\nDatos descifrados:")
        print(json.dumps(decrypted_data, indent=2))
        
    finally:
        # Cerrar el gestor
        await encryption_manager.close()

if __name__ == "__main__":
    asyncio.run(main())