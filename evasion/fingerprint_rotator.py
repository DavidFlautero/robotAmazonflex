import random
import hashlib
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from cryptography.fernet import Fernet
import secrets
import string
from prometheus_client import Counter, Gauge, Histogram
from app.database.database_manager import db_manager
from app.config import settings
from app.evasion.behavioral_patterns import behavior_analyzer

# Configuración de logging
logger = logging.getLogger(__name__)

# Métricas Prometheus
FINGERPRINT_ROTATIONS = Counter('fingerprint_rotations_total', 'Total fingerprint rotations')
FINGERPRINT_QUALITY = Gauge('fingerprint_quality_score', 'Fingerprint quality score')
BROWSER_CONSISTENCY = Gauge('browser_consistency_score', 'Browser consistency score')
FINGERPRINT_DETECTION_RISK = Gauge('fingerprint_detection_risk', 'Fingerprint detection risk')

@dataclass
class FingerprintConfig:
    rotation_interval: int = 3600  # segundos
    min_quality_threshold: float = 0.8
    max_history_size: int = 100
    entropy_level: str = "high"  # low, medium, high, extreme
    canvas_noise_level: float = 0.1
    webgl_variation: float = 0.3
    font_diversity: int = 5

class FingerprintComponent(Enum):
    USER_AGENT = "user_agent"
    SCREEN = "screen"
    TIMEZONE = "timezone"
    LANGUAGE = "language"
    PLATFORM = "platform"
    HARDWARE = "hardware"
    FONTS = "fonts"
    CANVAS = "canvas"
    WEBGL = "webgl"
    AUDIO = "audio"
    CPU = "cpu"
    MEMORY = "memory"
    CONNECTION = "connection"

@dataclass
class DigitalFingerprint:
    user_id: int
    components: Dict[FingerprintComponent, Any]
    fingerprint_hash: str
    quality_score: float
    created_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

class AdvancedFingerprintRotator:
    def __init__(self):
        self.config = FingerprintConfig()
        self.fingerprint_history: Dict[int, List[DigitalFingerprint]] = {}
        self.component_generators = self._initialize_component_generators()
        self.quality_validators = self._initialize_quality_validators()
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self._setup_metrics()
        
    def _initialize_component_generators(self) -> Dict[FingerprintComponent, Any]:
        """Inicializar generadores de componentes de fingerprint"""
        return {
            FingerprintComponent.USER_AGENT: self._generate_user_agent,
            FingerprintComponent.SCREEN: self._generate_screen_properties,
            FingerprintComponent.TIMEZONE: self._generate_timezone,
            FingerprintComponent.LANGUAGE: self._generate_language,
            FingerprintComponent.PLATFORM: self._generate_platform,
            FingerprintComponent.HARDWARE: self._generate_hardware_info,
            FingerprintComponent.FONTS: self._generate_font_list,
            FingerprintComponent.CANVAS: self._generate_canvas_fingerprint,
            FingerprintComponent.WEBGL: self._generate_webgl_info,
            FingerprintComponent.AUDIO: self._generate_audio_context,
            FingerprintComponent.CPU: self._generate_cpu_info,
            FingerprintComponent.MEMORY: self._generate_memory_info,
            FingerprintComponent.CONNECTION: self._generate_connection_info
        }
    
    def _initialize_quality_validators(self) -> Dict[str, Any]:
        """Inicializar validadores de calidad"""
        return {
            "consistency": self._validate_consistency,
            "realism": self._validate_realism,
            "uniqueness": self._validate_uniqueness,
            "persistence": self._validate_persistence,
            "stealth": self._validate_stealth
        }
    
    def _setup_metrics(self):
        """Configurar métricas adicionales"""
        self.rotation_times = Histogram('fingerprint_rotation_seconds', 'Fingerprint rotation time')
        self.component_quality = Gauge('fingerprint_component_quality', 'Individual component quality', ['component'])
    
    async def rotate_fingerprint(self, user_id: int, current_session: Dict[str, Any]) -> DigitalFingerprint:
        """Rotar fingerprint digital del usuario"""
        start_time = time.time()
        
        try:
            # Obtener comportamiento actual para personalizar el fingerprint
            behavior_data = await self._extract_behavioral_context(current_session)
            
            # Generar nuevos componentes
            new_components = await self._generate_new_components(user_id, behavior_data)
            
            # Calcular hash del fingerprint
            fingerprint_hash = self._calculate_fingerprint_hash(new_components)
            
            # Validar calidad
            quality_score = await self._validate_fingerprint_quality(new_components, user_id)
            
            # Crear objeto fingerprint
            fingerprint = DigitalFingerprint(
                user_id=user_id,
                components=new_components,
                fingerprint_hash=fingerprint_hash,
                quality_score=quality_score,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=self.config.rotation_interval),
                metadata={
                    "behavior_context": behavior_data,
                    "entropy_level": self.config.entropy_level,
                    "generation_strategy": "adaptive"
                }
            )
            
            # Guardar en historial
            await self._store_fingerprint(user_id, fingerprint)
            
            # Actualizar métricas
            rotation_time = time.time() - start_time
            self.rotation_times.observe(rotation_time)
            FINGERPRINT_ROTATIONS.inc()
            FINGERPRINT_QUALITY.set(quality_score)
            
            logger.info(f"Fingerprint rotated for user {user_id}. Quality: {quality_score:.2f}")
            
            return fingerprint
            
        except Exception as e:
            logger.error(f"Error rotating fingerprint: {e}")
            raise
    
    async def _extract_behavioral_context(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer contexto comportamental para personalización del fingerprint"""
        try:
            # Analizar comportamiento actual
            behavior_analysis = await behavior_analyzer.analyze_behavior(
                session_data.get("user_id", 0),
                session_data
            )
            
            return {
                "anomaly_score": behavior_analysis.get("anomaly_score", 0.0),
                "human_likelihood": behavior_analysis.get("human_likelihood", 0.0),
                "pattern_consistency": behavior_analysis.get("pattern_consistency", 0.0),
                "recommended_action": behavior_analysis.get("recommended_action", "allow"),
                "risk_level": self._calculate_risk_level(behavior_analysis)
            }
            
        except Exception as e:
            logger.warning(f"Error extracting behavioral context: {e}")
            return {"risk_level": "medium"}
    
    def _calculate_risk_level(self, behavior_analysis: Dict[str, Any]) -> str:
        """Calcular nivel de riesgo basado en análisis comportamental"""
        anomaly_score = behavior_analysis.get("anomaly_score", 0.0)
        
        if anomaly_score > 0.8:
            return "critical"
        elif anomaly_score > 0.6:
            return "high"
        elif anomaly_score > 0.4:
            return "medium"
        else:
            return "low"
    
    async def _generate_new_components(self, user_id: int, behavior_context: Dict[str, Any]) -> Dict[FingerprintComponent, Any]:
        """Generar nuevos componentes de fingerprint"""
        components = {}
        risk_level = behavior_context.get("risk_level", "medium")
        
        # Generar cada componente con estrategia adaptativa
        for component_type, generator in self.component_generators.items():
            try:
                component_value = await generator(user_id, risk_level)
                components[component_type] = component_value
                
                # Validar calidad individual del componente
                comp_quality = self._validate_component_quality(component_type, component_value)
                self.component_quality.labels(component=component_type.value).set(comp_quality)
                
            except Exception as e:
                logger.error(f"Error generating {component_type}: {e}")
                # Usar valor por defecto en caso de error
                components[component_type] = self._get_fallback_component(component_type)
        
        return components
    
    def _calculate_fingerprint_hash(self, components: Dict[FingerprintComponent, Any]) -> str:
        """Calcular hash único del fingerprint"""
        # Serializar componentes de manera consistente
        serialized_data = json.dumps(
            {k.value: v for k, v in sorted(components.items())},
            sort_keys=True,
            default=str
        )
        
        # Calcular hash con múltiples algoritmos para mayor unicidad
        md5_hash = hashlib.md5(serialized_data.encode()).hexdigest()
        sha256_hash = hashlib.sha256(serialized_data.encode()).hexdigest()
        
        # Combinar hashes para mayor entropía
        combined_hash = hashlib.sha3_512(f"{md5_hash}{sha256_hash}".encode()).hexdigest()
        
        return combined_hash
    
    async def _validate_fingerprint_quality(self, components: Dict[FingerprintComponent, Any], user_id: int) -> float:
        """Validar calidad general del fingerprint"""
        quality_scores = []
        
        for validator_name, validator_func in self.quality_validators.items():
            try:
                score = await validator_func(components, user_id)
                quality_scores.append(score)
            except Exception as e:
                logger.warning(f"Quality validator {validator_name} failed: {e}")
                quality_scores.append(0.5)  # Score neutral en caso de error
        
        # Ponderar scores según importancia
        weights = {
            "consistency": 0.25,
            "realism": 0.30,
            "uniqueness": 0.20,
            "persistence": 0.15,
            "stealth": 0.10
        }
        
        final_score = sum(
            quality_scores[i] * weight
            for i, weight in enumerate(weights.values())
        )
        
        return max(0.0, min(1.0, final_score))
    
    async def _validate_consistency(self, components: Dict[FingerprintComponent, Any], user_id: int) -> float:
        """Validar consistencia interna del fingerprint"""
        consistency_checks = []
        
        # Verificar compatibilidad entre componentes
        consistency_checks.append(self._check_platform_consistency(components))
        consistency_checks.append(self._check_hardware_consistency(components))
        consistency_checks.append(self._check_browser_consistency(components))
        
        # Verificar con historial del usuario
        historical_consistency = await self._check_historical_consistency(user_id, components)
        consistency_checks.append(historical_consistency)
        
        return np.mean(consistency_checks)
    
    async def _validate_realism(self, components: Dict[FingerprintComponent, Any], user_id: int) -> float:
        """Validar realismo del fingerprint"""
        realism_checks = []
        
        # Verificar valores realistas
        realism_checks.append(self._check_realistic_values(components))
        realism_checks.append(self._check_common_configurations(components))
        realism_checks.append(self._check_technical_plausibility(components))
        
        return np.mean(realism_checks)
    
    async def _validate_uniqueness(self, components: Dict[FingerprintComponent, Any], user_id: int) -> float:
        """Validar unicidad del fingerprint"""
        try:
            # Verificar contra la base de datos global
            fingerprint_hash = self._calculate_fingerprint_hash(components)
            
            with db_manager.get_db() as db:
                # Buscar fingerprints similares (esto sería una consulta real a la base de datos)
                similar_count = 0  # Placeholder para consulta real
                
                uniqueness_score = 1.0 - min(similar_count / 1000, 1.0)
                return uniqueness_score
                
        except Exception as e:
            logger.warning(f"Uniqueness validation failed: {e}")
            return 0.7  # Score por defecto
    
    async def _validate_persistence(self, components: Dict[FingerprintComponent, Any], user_id: int) -> float:
        """Validar persistencia del fingerprint"""
        # Verificar componentes que deberían persistir entre sesiones
        persistence_checks = []
        
        persistence_checks.append(self._check_persistent_components(components))
        persistence_checks.append(self._check_storage_capabilities(components))
        
        return np.mean(persistence_checks)
    
    async def _validate_stealth(self, components: Dict[FingerprintComponent, Any], user_id: int) -> float:
        """Validar stealth del fingerprint"""
        stealth_checks = []
        
        # Verificar características de evasión
        stealth_checks.append(self._check_automation_detection(components))
        stealth_checks.append(self._check_fingerprint_uniqueness(components))
        stealth_checks.append(self._check_common_automation_patterns(components))
        
        return np.mean(stealth_checks)
    
    async def _store_fingerprint(self, user_id: int, fingerprint: DigitalFingerprint):
        """Almacenar fingerprint en historial"""
        if user_id not in self.fingerprint_history:
            self.fingerprint_history[user_id] = []
        
        # Agregar al historial
        self.fingerprint_history[user_id].append(fingerprint)
        
        # Mantener tamaño máximo del historial
        if len(self.fingerprint_history[user_id]) > self.config.max_history_size:
            self.fingerprint_history[user_id] = self.fingerprint_history[user_id][-self.config.max_history_size:]
        
        # Guardar en base de datos (implementación real)
        try:
            await self._save_to_database(user_id, fingerprint)
        except Exception as e:
            logger.error(f"Error saving fingerprint to database: {e}")
    
    async def _save_to_database(self, user_id: int, fingerprint: DigitalFingerprint):
        """Guardar fingerprint en base de datos"""
        # Esta sería la implementación real con tu database_manager
        pass
    
    # Generadores de componentes individuales
    async def _generate_user_agent(self, user_id: int, risk_level: str) -> str:
        """Generar User-Agent realista"""
        browsers = {
            "chrome": {
                "versions": ["90.0.4430.212", "91.0.4472.124", "92.0.4515.107", "93.0.4577.63"],
                "platforms": ["Windows NT 10.0", "Macintosh; Intel Mac OS X 10_15_7", "X11; Linux x86_64"]
            },
            "firefox": {
                "versions": ["89.0", "90.0", "91.0", "92.0"],
                "platforms": ["Windows NT 10.0", "Macintosh; Intel Mac OS X 10.15", "X11; Linux x86_64"]
            },
            "safari": {
                "versions": ["14.1.1", "14.1.2", "15.0", "15.1"],
                "platforms": ["Macintosh; Intel Mac OS X 10_15_7", "Macintosh; Intel Mac OS X 11_5_2"]
            }
        }
        
        # Seleccionar navegador basado en riesgo
        if risk_level in ["critical", "high"]:
            browser = random.choice(["chrome", "firefox"])  # Navegadores más comunes
        else:
            browser = random.choice(list(browsers.keys()))
        
        browser_info = browsers[browser]
        version = random.choice(browser_info["versions"])
        platform = random.choice(browser_info["platforms"])
        
        if browser == "chrome":
            return f"Mozilla/5.0 ({platform}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"
        elif browser == "firefox":
            return f"Mozilla/5.0 ({platform}; rv:{version}) Gecko/20100101 Firefox/{version}"
        else:  # safari
            return f"Mozilla/5.0 ({platform}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{version} Safari/605.1.15"
    
    async def _generate_screen_properties(self, user_id: int, risk_level: str) -> Dict[str, Any]:
        """Generar propiedades de pantalla realistas"""
        common_resolutions = [
            {"width": 1920, "height": 1080, "depth": 24},
            {"width": 1366, "height": 768, "depth": 24},
            {"width": 1536, "height": 864, "depth": 24},
            {"width": 1440, "height": 900, "depth": 24},
            {"width": 1280, "height": 720, "depth": 24},
            {"width": 2560, "height": 1440, "depth": 24},
            {"width": 3840, "height": 2160, "depth": 30}
        ]
        
        # Para alto riesgo, usar resoluciones más comunes
        if risk_level in ["critical", "high"]:
            resolution = random.choice(common_resolutions[:3])
        else:
            resolution = random.choice(common_resolutions)
        
        return {
            "width": resolution["width"],
            "height": resolution["height"],
            "colorDepth": resolution["depth"],
            "pixelDepth": resolution["depth"],
            "availWidth": resolution["width"] - random.randint(0, 100),
            "availHeight": resolution["height"] - random.randint(0, 100),
            "devicePixelRatio": random.choice([1, 1.5, 2, 2.5, 3])
        }
    
    async def _generate_timezone(self, user_id: int, risk_level: str) -> Dict[str, Any]:
        """Generar información de timezone"""
        timezones = [
            "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
            "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Moscow",
            "Asia/Tokyo", "Asia/Shanghai", "Asia/Seoul", "Asia/Kolkata",
            "Australia/Sydney", "Australia/Melbourne"
        ]
        
        offset_minutes = random.randint(-720, 720)  # -12h to +12h
        offset_hours = offset_minutes // 60
        offset_minutes %= 60
        
        return {
            "timezone": random.choice(timezones),
            "offset": f"{offset_hours:+03d}:{offset_minutes:02d}",
            "dst": random.choice([True, False]),
            "locale": "en-US"
        }
    
    async def _generate_language(self, user_id: int, risk_level: str) -> List[str]:
        """Generar lista de idiomas"""
        primary_languages = ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "pt-BR", "ru-RU", "zh-CN", "ja-JP"]
        secondary_languages = ["en", "es", "fr", "de", "pt", "ru", "zh", "ja", "ko", "ar"]
        
        languages = [random.choice(primary_languages)]
        
        # Agregar idiomas secundarios
        num_secondary = random.randint(0, 3)
        for _ in range(num_secondary):
            lang = random.choice(secondary_languages)
            if lang not in languages:
                languages.append(lang)
        
        return languages
    
    async def _generate_platform(self, user_id: int, risk_level: str) -> str:
        """Generar plataforma"""
        platforms = [
            "Win32", "Win64", "MacIntel", "Linux x86_64", "Linux i686",
            "iPhone", "iPad", "Android", "CrOS"
        ]
        
        # Para alto riesgo, usar plataformas más comunes
        if risk_level in ["critical", "high"]:
            return random.choice(["Win32", "MacIntel", "Linux x86_64"])
        
        return random.choice(platforms)
    
    async def _generate_hardware_info(self, user_id: int, risk_level: str) -> Dict[str, Any]:
        """Generar información de hardware"""
        concurrency = random.choice([4, 8, 16, 32, 64])
        memory = random.choice([4, 8, 16, 32, 64])
        
        return {
            "hardwareConcurrency": concurrency,
            "deviceMemory": memory,
            "maxTouchPoints": random.randint(0, 10),
            "cpuClass": random.choice(["unknown", "amd64", "ia64", "x86"]),
            "architecture": random.choice(["amd64", "x86", "arm64"])
        }
    
    async def _generate_font_list(self, user_id: int, risk_level: str) -> List[str]:
        """Generar lista de fuentes instaladas"""
        common_fonts = [
            "Arial", "Helvetica", "Times New Roman", "Times", "Courier New",
            "Courier", "Verdana", "Georgia", "Palatino", "Garamond",
            "Comic Sans MS", "Trebuchet MS", "Arial Black", "Impact"
        ]
        
        system_specific_fonts = {
            "Windows": ["Segoe UI", "Tahoma", "Microsoft Sans Serif", "Calibri"],
            "Mac": ["Helvetica Neue", "Lucida Grande", "Geneva", "Menlo"],
            "Linux": ["Ubuntu", "DejaVu Sans", "Liberation Sans", "FreeSans"]
        }
        
        fonts = common_fonts.copy()
        
        # Agregar fuentes específicas del sistema
        system = random.choice(list(system_specific_fonts.keys()))
        fonts.extend(system_specific_fonts[system])
        
        # Mezclar y limitar el número de fuentes
        random.shuffle(fonts)
        return fonts[:random.randint(15, 30)]
    
    async def _generate_canvas_fingerprint(self, user_id: int, risk_level: str) -> str:
        """Generar fingerprint de canvas con ruido controlado"""
        base_data = secrets.token_hex(32)
        
        # Agregar ruido basado en el nivel de riesgo
        if risk_level == "critical":
            noise_level = 0.05  # Mínimo ruido
        elif risk_level == "high":
            noise_level = 0.1
        elif risk_level == "medium":
            noise_level = 0.2
        else:
            noise_level = self.config.canvas_noise_level
        
        # Aplicar ruido
        noisy_data = self._apply_noise_to_canvas(base_data, noise_level)
        
        return hashlib.sha256(noisy_data.encode()).hexdigest()
    
    async def _generate_webgl_info(self, user_id: int, risk_level: str) -> Dict[str, Any]:
        """Generar información WebGL"""
        vendors = ["Google Inc.", "Intel Inc.", "NVIDIA Corporation", "AMD", "Apple Inc."]
        renderers = [
            "ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0)",
            "Intel Iris OpenGL Engine",
            "NVIDIA GeForce GTX 1060 OpenGL Engine",
            "AMD Radeon Pro 5600M OpenGL Engine",
            "Apple GPU"
        ]
        
        return {
            "vendor": random.choice(vendors),
            "renderer": random.choice(renderers),
            "version": random.choice(["WebGL 1.0", "WebGL 2.0"]),
            "maxAnisotropy": random.randint(1, 16),
            "aliasedLineWidthRange": [1, random.randint(5, 10)]
        }
    
    async def _generate_audio_context(self, user_id: int, risk_level: str) -> Dict[str, Any]:
        """Generar contexto de audio"""
        return {
            "sampleRate": random.choice([44100, 48000, 96000]),
            "channelCount": random.randint(1, 8),
            "bufferSize": random.choice([256, 512, 1024, 2048]),
            "numberOfInputs": random.randint(1, 8),
            "numberOfOutputs": random.randint(1, 8)
        }
    
    async def _generate_cpu_info(self, user_id: int, risk_level: str) -> Dict[str, Any]:
        """Generar información de CPU"""
        return {
            "cores": random.choice([4, 8, 16, 32]),
            "threads": random.choice([8, 16, 32, 64]),
            "architecture": random.choice(["x86_64", "arm64", "amd64"]),
            "vendor": random.choice(["Intel", "AMD", "Apple"]),
            "model": random.choice(["i7-10700K", "Ryzen 7 5800X", "M1 Pro", "Xeon Gold 6254"])
        }
    
    async def _generate_memory_info(self, user_id: int, risk_level: str) -> Dict[str, Any]:
        """Generar información de memoria"""
        return {
            "total": random.choice([8, 16, 32, 64, 128]) * 1024 * 1024 * 1024,
            "available": random.randint(4, 32) * 1024 * 1024 * 1024,
            "used": random.randint(4, 16) * 1024 * 1024 * 1024
        }
    
    async def _generate_connection_info(self, user_id: int, risk_level: str) -> Dict[str, Any]:
        """Generar información de conexión"""
        connection_types = ["4g", "wifi", "ethernet", "5g", "3g", "2g"]
        effective_types = ["4g", "3g", "2g", "slow-2g"]
        
        return {
            "type": random.choice(connection_types),
            "effectiveType": random.choice(effective_types),
            "downlink": random.uniform(1.0, 100.0),
            "rtt": random.randint(50, 500),
            "saveData": random.choice([True, False])
        }
    
    # Métodos de utilidad
    def _apply_noise_to_canvas(self, data: str, noise_level: float) -> str:
        """Aplicar ruido controlado a los datos del canvas"""
        data_bytes = bytearray(data.encode())
        
        for i in range(len(data_bytes)):
            if random.random() < noise_level:
                data_bytes[i] = random.randint(0, 255)
        
        return data_bytes.decode('latin-1')
    
    def _get_fallback_component(self, component_type: FingerprintComponent) -> Any:
        """Obtener componente de respaldo en caso de error"""
        fallbacks = {
            FingerprintComponent.USER_AGENT: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            FingerprintComponent.SCREEN: {"width": 1920, "height": 1080, "colorDepth": 24},
            FingerprintComponent.TIMEZONE: {"timezone": "UTC", "offset": "+00:00"},
            FingerprintComponent.LANGUAGE: ["en-US", "en"],
            FingerprintComponent.PLATFORM: "Win32",
            FingerprintComponent.HARDWARE: {"hardwareConcurrency": 8, "deviceMemory": 8},
            FingerprintComponent.FONTS: ["Arial", "Helvetica", "Times New Roman"],
            FingerprintComponent.CANVAS: "fallback_canvas_hash",
            FingerprintComponent.WEBGL: {"vendor": "Google Inc.", "renderer": "ANGLE"},
            FingerprintComponent.AUDIO: {"sampleRate": 44100, "channelCount": 2},
            FingerprintComponent.CPU: {"cores": 8, "vendor": "Intel"},
            FingerprintComponent.MEMORY: {"total": 17179869184},
            FingerprintComponent.CONNECTION: {"type": "wifi", "effectiveType": "4g"}
        }
        
        return fallbacks.get(component_type, "")
    
    def _validate_component_quality(self, component_type: FingerprintComponent, value: Any) -> float:
        """Validar calidad individual de componente"""
        # Implementar validaciones específicas por componente
        validators = {
            FingerprintComponent.USER_AGENT: self._validate_user_agent,
            FingerprintComponent.SCREEN: self._validate_screen_properties,
            FingerprintComponent.TIMEZONE: self._validate_timezone,
            # ... otros validadores
        }
        
        validator = validators.get(component_type, lambda x: 0.8)
        return validator(value)
    
    # Validadores de componentes individuales
    def _validate_user_agent(self, user_agent: str) -> float:
        """Validar User-Agent"""
        if not user_agent or len(user_agent) < 10:
            return 0.1
        
        # Verificar estructura básica
        if "Mozilla/5.0" not in user_agent:
            return 0.3
        
        # Verificar componentes esperados
        checks = [
            "AppleWebKit" in user_agent,
            "Safari" in user_agent or "Chrome" in user_agent or "Firefox" in user_agent,
            "(" in user_agent and ")" in user_agent
        ]
        
        return sum(checks) / len(checks)
    
    def _validate_screen_properties(self, screen: Dict[str, Any]) -> float:
        """Validar propiedades de pantalla"""
        required_keys = ["width", "height", "colorDepth"]
        if not all(key in screen for key in required_keys):
            return 0.1
        
        # Verificar valores realistas
        checks = [
            screen["width"] > 0,
            screen["height"] > 0,
            screen["colorDepth"] in [16, 24, 30, 32],
            screen.get("availWidth", 0) <= screen["width"],
            screen.get("availHeight", 0) <= screen["height"]
        ]
        
        return sum(checks) / len(checks)
    
    # Métodos de validación de calidad
    def _check_platform_consistency(self, components: Dict[FingerprintComponent, Any]) -> float:
        """Verificar consistencia de plataforma"""
        platform = components.get(FingerprintComponent.PLATFORM, "")
        user_agent = components.get(FingerprintComponent.USER_AGENT, "")
        
        if "Win" in platform and "Windows" not in user_agent:
            return 0.3
        if "Mac" in platform and "Mac" not in user_agent:
            return 0.3
        if "Linux" in platform and "Linux" not in user_agent:
            return 0.3
        
        return 0.9
    
    def _check_hardware_consistency(self, components: Dict[FingerprintComponent, Any]) -> float:
        """Verificar consistencia de hardware"""
        hardware = components.get(FingerprintComponent.HARDWARE, {})
        platform = components.get(FingerprintComponent.PLATFORM, "")
        
        # Verificar concurrencia realista
        concurrency = hardware.get("hardwareConcurrency", 0)
        if concurrency < 2 or concurrency > 64:
            return 0.4
        
        # Verificar memoria realista
        memory = hardware.get("deviceMemory", 0)
        if memory < 4 or memory > 64:
            return 0.4
        
        return 0.8
    
    def _check_browser_consistency(self, components: Dict[FingerprintComponent, Any]) -> float:
        """Verificar consistencia del navegador"""
        user_agent = components.get(FingerprintComponent.USER_AGENT, "")
        webgl = components.get(FingerprintComponent.WEBGL, {})
        
        checks = []
        
        # Verificar que el vendor de WebGL coincida con el navegador
        if "Chrome" in user_agent and "Google" not in webgl.get("vendor", ""):
            checks.append(0.5)
        if "Firefox" in user_agent and "Mozilla" not in webgl.get("vendor", ""):
            checks.append(0.5)
        
        return 0.8 if not checks else np.mean(checks)
    
    async def _check_historical_consistency(self, user_id: int, components: Dict[FingerprintComponent, Any]) -> float:
        """Verificar consistencia con historial del usuario"""
        if user_id not in self.fingerprint_history or not self.fingerprint_history[user_id]:
            return 0.7  # Score neutral para nuevo usuario
        
        historical_components = self.fingerprint_history[user_id][-1].components
        
        similarities = []
        for comp_type in components:
            if comp_type in historical_components:
                similarity = self._calculate_component_similarity(
                    components[comp_type],
                    historical_components[comp_type]
                )
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.6
    
    def _calculate_component_similarity(self, current: Any, previous: Any) -> float:
        """Calcular similitud entre componentes"""
        if isinstance(current, dict) and isinstance(previous, dict):
            return self._calculate_dict_similarity(current, previous)
        elif isinstance(current, list) and isinstance(previous, list):
            return self._calculate_list_similarity(current, previous)
        elif current == previous:
            return 0.9
        else:
            return 0.4
    
    def _calculate_dict_similarity(self, current: Dict, previous: Dict) -> float:
        """Calcular similitud entre diccionarios"""
        common_keys = set(current.keys()) & set(previous.keys())
        if not common_keys:
            return 0.3
        
        similarities = []
        for key in common_keys:
            if current[key] == previous[key]:
                similarities.append(0.9)
            else:
                similarities.append(0.4)
        
        return np.mean(similarities)
    
    def _calculate_list_similarity(self, current: List, previous: List) -> float:
        """Calcular similitud entre listas"""
        if not current or not previous:
            return 0.5
        
        common_elements = set(current) & set(previous)
        similarity = len(common_elements) / max(len(current), len(previous))
        
        return similarity
    
    def _check_realistic_values(self, components: Dict[FingerprintComponent, Any]) -> float:
        """Verificar valores realistas"""
        checks = []
        
        # Verificar screen
        screen = components.get(FingerprintComponent.SCREEN, {})
        if screen:
            checks.append(300 <= screen.get("width", 0) <= 7680)
            checks.append(300 <= screen.get("height", 0) <= 4320)
        
        # Verificar hardware
        hardware = components.get(FingerprintComponent.HARDWARE, {})
        if hardware:
            checks.append(2 <= hardware.get("hardwareConcurrency", 0) <= 64)
            checks.append(4 <= hardware.get("deviceMemory", 