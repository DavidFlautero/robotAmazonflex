from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from typing import Dict, List, Optional, Callable
import json
from prometheus_client import Counter, Histogram, Gauge
import asyncio
from security.zero_trust import ZeroTrustSecurity
import os

logger = logging.getLogger(__name__)

# Métricas Prometheus para middleware
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
REQUEST_SIZE = Histogram('http_request_size_bytes', 'HTTP request size', ['method', 'endpoint'])
RESPONSE_SIZE = Histogram('http_response_size_bytes', 'HTTP response size', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('http_requests_active', 'Active HTTP requests')

# Rate limiting metrics
RATE_LIMIT_HITS = Counter('rate_limit_hits_total', 'Total rate limit hits', ['endpoint', 'user_id'])
CONCURRENT_REQUESTS = Gauge('concurrent_requests_total', 'Current concurrent requests')

class AdvancedMiddleware:
    def __init__(self, app):
        self.app = app
        self.rate_limiter = RateLimiter()
        self.security_analyzer = SecurityAnalyzer()
        self.zt_security = ZeroTrustSecurity(os.getenv('SECRET_KEY', 'default-secret-key'))
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        request = Request(scope, receive)
        start_time = time.time()
        
        try:
            # Incrementar contador de requests activos
            ACTIVE_REQUESTS.inc()
            CONCURRENT_REQUESTS.inc()
            
            # Verificar rate limiting
            if await self.rate_limiter.is_rate_limited(request):
                RATE_LIMIT_HITS.labels(
                    endpoint=request.url.path, 
                    user_id=self._extract_user_id(request)
                ).inc()
                return await self._rate_limited_response(send)
            
            # Análisis de seguridad
            security_check = await self.security_analyzer.analyze_request(request)
            if not security_check["allowed"]:
                return await self._security_blocked_response(security_check, send)
            
            # Continuar con el request
            response = await self.app(scope, receive, send)
            
            # Registrar métricas
            duration = time.time() - start_time
            self._record_metrics(request, response, duration)
            
            return response
            
        except Exception as e:
            # Manejar errores y registrar métricas incluso en fallos
            duration = time.time() - start_time
            self._record_error_metrics(request, e, duration)
            raise e
            
        finally:
            # Decrementar contadores
            ACTIVE_REQUESTS.dec()
            CONCURRENT_REQUESTS.dec()

    def _extract_user_id(self, request: Request) -> str:
        """Extraer user_id del token JWT para rate limiting"""
        try:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                user_id = self.zt_security.verify_token(token)
                return user_id if user_id else "anonymous"
        except:
            pass
        return "anonymous"

    async def _rate_limited_response(self, send):
        """Respuesta para requests rate limited"""
        response = JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests, please try again later",
                "retry_after": 60
            }
        )
        await response(scope, receive, send)

    async def _security_blocked_response(self, security_check: Dict, send):
        """Respuesta para requests bloqueados por seguridad"""
        response = JSONResponse(
            status_code=403,
            content={
                "error": "Security block",
                "message": security_check["reason"],
                "threat_level": security_check["threat_level"]
            }
        )
        await response(scope, receive, send)

    def _record_metrics(self, request: Request, response, duration: float):
        """Registrar métricas Prometheus para requests exitosos"""
        endpoint = request.url.path
        method = request.method
        
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Estimación de tamaños (simplificado)
        content_length = request.headers.get("content-length", 0)
        if content_length:
            REQUEST_SIZE.labels(
                method=method,
                endpoint=endpoint
            ).observe(int(content_length))

    def _record_error_metrics(self, request: Request, error: Exception, duration: float):
        """Registrar métricas para requests con error"""
        endpoint = request.url.path
        method = request.method
        status_code = 500 if not hasattr(error, 'status_code') else error.status_code
        
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.cleanup_interval = 60  # seconds
        self._start_cleanup_task()
        
    def _start_cleanup_task(self):
        """Iniciar tarea de limpieza periódica"""
        async def cleanup():
            while True:
                await asyncio.sleep(self.cleanup_interval)
                self._cleanup_old_entries()
        
        asyncio.create_task(cleanup())
    
    def _cleanup_old_entries(self):
        """Limpiar entradas antiguas del rate limiter"""
        current_time = time.time()
        for key in list(self.requests.keys()):
            if current_time - self.requests[key]["last_reset"] > 3600:  # 1 hour
                del self.requests[key]
    
    async def is_rate_limited(self, request: Request) -> bool:
        """Verificar si el request está rate limited"""
        client_id = self._get_client_identifier(request)
        endpoint = request.url.path
        current_time = time.time()
        
        # Configuración de rate limits por endpoint
        limits = self._get_endpoint_limits(endpoint)
        
        if client_id not in self.requests:
            self.requests[client_id] = {
                "count": 1,
                "last_reset": current_time,
                "endpoints": {endpoint: 1}
            }
            return False
        
        # Resetear contador si ha pasado el período
        if current_time - self.requests[client_id]["last_reset"] > limits["window"]:
            self.requests[client_id] = {
                "count": 1,
                "last_reset": current_time,
                "endpoints": {endpoint: 1}
            }
            return False
        
        # Verificar límites globales por cliente
        if self.requests[client_id]["count"] >= limits["global_max"]:
            return True
        
        # Verificar límites por endpoint
        endpoint_count = self.requests[client_id]["endpoints"].get(endpoint, 0)
        if endpoint_count >= limits["endpoint_max"]:
            return True
        
        # Incrementar contadores
        self.requests[client_id]["count"] += 1
        self.requests[client_id]["endpoints"][endpoint] = endpoint_count + 1
        
        return False
    
    def _get_client_identifier(self, request: Request) -> str:
        """Obtener identificador único del cliente"""
        # Prioridad: user_id > IP > session
        try:
            # Intentar obtener user_id del token
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                user_id = ZeroTrustSecurity(os.getenv('SECRET_KEY')).verify_token(token)
                if user_id:
                    return f"user_{user_id}"
        except:
            pass
        
        # Fallback a IP address
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0]
        else:
            client_ip = request.client.host
            
        return f"ip_{client_ip}"
    
    def _get_endpoint_limits(self, endpoint: str) -> Dict:
        """Obtener configuración de rate limits por endpoint"""
        endpoint_limits = {
            "/api/capture-blocks": {"window": 60, "global_max": 30, "endpoint_max": 10},
            "/api/user/stats": {"window": 60, "global_max": 60, "endpoint_max": 20},
            "/healthz": {"window": 60, "global_max": 120, "endpoint_max": 60},
            "/metrics": {"window": 60, "global_max": 60, "endpoint_max": 30},
            "default": {"window": 60, "global_max": 100, "endpoint_max": 50}
        }
        
        return endpoint_limits.get(endpoint, endpoint_limits["default"])

class SecurityAnalyzer:
    def __init__(self):
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.geo_blocklist = self._load_geo_blocklist()
        
    def _load_suspicious_patterns(self) -> List[Dict]:
        """Cargar patrones sospechosos desde configuración"""
        return [
            {"pattern": r"(?i)(select|insert|update|delete|drop|union)", "type": "sql_injection"},
            {"pattern": r"(?i)(<script|javascript:|onload=)", "type": "xss"},
            {"pattern": r"(\.\./|\.\.\\|/etc/passwd)", "type": "path_traversal"},
            {"pattern": r"(?i)(eval\(|system\(|exec\()", "type": "code_injection"}
        ]
    
    def _load_geo_blocklist(self) -> List[str]:
        """Cargar lista de países bloqueados"""
        return ["RU", "CN", "KP", "IR", "SY"]  # Ejemplo: bloquear estos países
    
    async def analyze_request(self, request: Request) -> Dict:
        """Analizar request para detectar amenazas de seguridad"""
        analysis = {
            "allowed": True,
            "threat_level": "low",
            "reasons": [],
            "threats_detected": []
        }
        
        # 1. Verificar User-Agent sospechoso
        user_agent = request.headers.get("user-agent", "")
        if not user_agent or "python" in user_agent.lower():
            analysis["reasons"].append("Suspicious User-Agent")
            analysis["threat_level"] = "medium"
        
        # 2. Verificar contenido malicioso en body
        if await self._has_malicious_content(request):
            analysis["threats_detected"].append("malicious_content")
            analysis["threat_level"] = "high"
        
        # 3. Verificar geolocalización (si está disponible)
        if await self._is_blocked_geo(request):
            analysis["threats_detected"].append("blocked_region")
            analysis["threat_level"] = "high"
        
        # 4. Verificar headers sospechosos
        if self._has_suspicious_headers(request):
            analysis["threats_detected"].append("suspicious_headers")
            analysis["threat_level"] = "medium"
        
        # 5. Verificar tamaño de request excesivo
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
            analysis["threats_detected"].append("large_request")
            analysis["threat_level"] = "medium"
        
        # Decidir si bloquear el request
        if analysis["threat_level"] == "high" or len(analysis["threats_detected"]) > 2:
            analysis["allowed"] = False
            analysis["reason"] = "High threat level detected"
        
        return analysis
    
    async def _has_malicious_content(self, request: Request) -> bool:
        """Verificar contenido malicioso en el request"""
        try:
            # Leer el body una vez para múltiples verificaciones
            body = await request.body()
            body_str = body.decode('utf-8', errors='ignore')
            
            # Verificar patrones de inyección SQL
            for pattern in self.suspicious_patterns:
                import re
                if re.search(pattern["pattern"], body_str):
                    return True
                    
            # Verificar JSON malformado (podría ser un ataque)
            if request.headers.get("content-type") == "application/json":
                try:
                    json.loads(body_str)
                except json.JSONDecodeError:
                    return True
                    
        except Exception as e:
            logger.warning(f"Error analyzing request content: {e}")
            
        return False
    
    async def _is_blocked_geo(self, request: Request) -> bool:
        """Verificar si la request viene de una región bloqueada"""
        # En producción, integrar con servicio de geolocalización
        geo_header = request.headers.get("cf-ipcountry")  # Cloudflare
        if geo_header and geo_header in self.geo_blocklist:
            return True
            
        # Verificar IPs sospechosas (ejemplo simplificado)
        client_ip = self._get_client_ip(request)
        if client_ip and self._is_suspicious_ip(client_ip):
            return True
            
        return False
    
    def _has_suspicious_headers(self, request: Request) -> bool:
        """Verificar headers sospechosos"""
        suspicious_headers = {
            "x-forwarded-host": r"(localhost|127\.0\.0\.1|0\.0\.0\.0)",
            "via": r".*",
            "x-real-ip": r".*",
        }
        
        for header, pattern in suspicious_headers.items():
            if header in request.headers:
                import re
                if re.match(pattern, request.headers[header]):
                    return True
                    
        return False
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Obtener la IP real del cliente"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else None
    
    def _is_suspicious_ip(self, ip: str) -> bool:
        """Verificar si la IP es sospechosa (lista básica)"""
        suspicious_ips = [
            "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",  # Private networks
            "127.0.0.0/8", "0.0.0.0", "255.255.255.255"       # Localhost/broadcast
        ]
        
        # Verificación simplificada - en producción usar librería de IP
        for suspicious_ip in suspicious_ips:
            if ip.startswith(suspicious_ip.split("/")[0]):
                return True
                
        return False

# Middleware específico para logging estructurado
class StructuredLoggingMiddleware:
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        request = Request(scope, receive)
        start_time = time.time()
        
        try:
            response = await self.app(scope, receive, send)
            duration = time.time() - start_time
            
            self.log_request(request, response, duration, None)
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_request(request, None, duration, e)
            raise e
    
    def log_request(self, request: Request, response, duration: float, error: Optional[Exception]):
        """Log estructurado para requests HTTP"""
        log_data = {
            "timestamp": time.time(),
            "method": request.method,
            "url": str(request.url),
            "ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "duration_ms": round(duration * 1000, 2),
            "status_code": response.status if response else 500,
            "error": str(error) if error else None,
            "request_size": request.headers.get("content-length", 0),
            "response_size": response.headers.get("content-length", 0) if response else 0
        }
        
        if error:
            logger.error("HTTP Request Error", extra=log_data)
        else:
            logger.info("HTTP Request", extra=log_data)
    
    def _get_client_ip(self, request: Request) -> str:
        """Obtener IP del cliente con soporte para proxies"""
        if request.headers.get("x-forwarded-for"):
            return request.headers["x-forwarded-for"].split(",")[0]
        return request.client.host if request.client else "unknown"

# Configuración de CORS
def setup_cors_middleware(app):
    """Configurar middleware CORS"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )
    return app

# Middleware de compresión (GZip)
class GZipMiddleware:
    def __init__(self, app, minimum_size=500, compress_level=6):
        self.app = app
        self.minimum_size = minimum_size
        self.compress_level = compress_level
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Solo comprimir respuestas grandes
        original_send = send
        
        async def compressed_send(message):
            if message.get("type") == "http.response.start":
                # Agregar header de compresión
                headers = dict(message.get("headers", []))
                headers[b"content-encoding"] = b"gzip"
                message["headers"] = list(headers.items())
            
            await original_send(message)
        
        await self.app(scope, receive, compressed_send)

# Middleware de timeout
class TimeoutMiddleware:
    def __init__(self, app, timeout=30):
        self.app = app
        self.timeout = timeout
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        try:
            # Ejecutar con timeout
            await asyncio.wait_for(
                self.app(scope, receive, send),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            # Responder con error de timeout
            response = JSONResponse(
                status_code=504,
                content={"error": "Request timeout", "message": "The request took too long to process"}
            )
            await response(scope, receive, send)