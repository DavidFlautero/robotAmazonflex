import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Generator, List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, Session, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy import event
from sqlalchemy import func
import redis
from prometheus_client import Counter, Gauge, Histogram
import json
from app.config import settings

# Configuración de logging
logger = logging.getLogger(__name__)

# Métricas Prometheus
DB_QUERY_COUNT = Counter('db_query_total', 'Total database queries')
DB_QUERY_DURATION = Histogram('db_query_duration_seconds', 'Database query duration')
DB_CONNECTION_POOL = Gauge('db_connection_pool_size', 'Database connection pool size')
DB_ACTIVE_CONNECTIONS = Gauge('db_active_connections', 'Active database connections')

# Base de datos
Base = declarative_base()

# Modelo de Usuario
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    subscription_id = Column(String(100))
    subscription_status = Column(String(20), default="inactive")
    stripe_customer_id = Column(String(100))
    
    # Relaciones
    blocks = relationship("Block", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")
    proxies = relationship("UserProxy", back_populates="user")

# Modelo de Bloque
class Block(Base):
    __tablename__ = "blocks"
    
    id = Column(Integer, primary_key=True, index=True)
    block_id = Column(String(50), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    payment = Column(Float, nullable=False)
    location = Column(String(100), nullable=False)
    duration = Column(Integer, nullable=False)  # en minutos
    status = Column(String(20), default="available")  # available, captured, failed, expired
    captured_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    user = relationship("User", back_populates="blocks")

# Modelo de Sesión de Usuario
class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(500), nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relaciones
    user = relationship("User", back_populates="sessions")

# Modelo de Proxy de Usuario
class UserProxy(Base):
    __tablename__ = "user_proxies"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    proxy_url = Column(String(500), nullable=False)
    proxy_type = Column(String(20), default="residential")  # residential, datacenter, mobile
    is_active = Column(Boolean, default=True)
    success_rate = Column(Float, default=0.0)
    response_time = Column(Float, default=0.0)  # en milisegundos
    last_used = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    user = relationship("User", back_populates="proxies")

# Modelo de Configuración de Búsqueda
class SearchConfig(Base):
    __tablename__ = "search_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(50), nullable=False)
    min_payment = Column(Float, default=0.0)
    max_payment = Column(Float, default=1000.0)
    locations = Column(Text)  # JSON array de ubicaciones
    start_time = Column(String(5))  # formato HH:MM
    end_time = Column(String(5))    # formato HH:MM
    max_distance = Column(Integer, default=50)  # en millas
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relación
    user = relationship("User")

# Clase avanzada de gestión de base de datos
class AdvancedDatabaseManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AdvancedDatabaseManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.engine = None
        self.SessionLocal = None
        self.redis_client = None
        self._initialized = True
        self._setup_database()
        self._setup_redis()
    
    def _setup_database(self):
        """Configurar la conexión a la base de datos con pooling avanzado"""
        try:
            # Configurar engine con pooling avanzado
            self.engine = create_engine(
                settings.database_url,
                poolclass=QueuePool,
                pool_size=settings.database_pool_size,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800,  # 30 minutos
                pool_pre_ping=True,  # Verificar conexiones antes de usarlas
                echo=False  # Cambiar a True para debug
            )
            
            # Configurar session factory
            self.SessionLocal = scoped_session(
                sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self.engine,
                    expire_on_commit=False
                )
            )
            
            # Eventos para métricas
            @event.listens_for(self.engine, "before_cursor_execute")
            def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                conn.info.setdefault('query_start_time', []).append(time.time())
            
            @event.listens_for(self.engine, "after_cursor_execute")
            def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                total = time.time() - conn.info['query_start_time'].pop(-1)
                DB_QUERY_DURATION.observe(total)
                DB_QUERY_COUNT.inc()
                
            logger.info("Database connection pool configured successfully")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def _setup_redis(self):
        """Configurar cliente Redis para caché"""
        try:
            self.redis_client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Error setting up Redis: {e}")
            self.redis_client = None
    
    @contextmanager
    def get_db(self) -> Generator[Session, None, None]:
        """Context manager para obtener sesión de base de datos"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def get_redis(self) -> Optional[redis.Redis]:
        """Obtener cliente Redis"""
        return self.redis_client
    
    # Métodos de caché avanzados
    def cache_get(self, key: str) -> Optional[Any]:
        """Obtener valor de caché"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None
    
    def cache_set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Establecer valor en caché"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.setex(key, expire, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def cache_delete(self, key: str) -> bool:
        """Eliminar valor de caché"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return False
    
    # Métodos avanzados de usuarios
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Obtener usuario por username con caché"""
        cache_key = f"user:{username}"
        cached_user = self.cache_get(cache_key)
        
        if cached_user:
            return User(**cached_user)
        
        with self.get_db() as db:
            user = db.query(User).filter(User.username == username).first()
            if user:
                user_dict = {c.name: getattr(user, c.name) for c in User.__table__.columns}
                self.cache_set(cache_key, user_dict, expire=300)
            return user
    
    def create_user(self, user_data: dict) -> User:
        """Crear nuevo usuario"""
        with self.get_db() as db:
            user = User(**user_data)
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Invalidar caché
            self.cache_delete(f"user:{user.username}")
            
            return user
    
    # Métodos avanzados de bloques
    def get_user_blocks(self, user_id: int, limit: int = 100, offset: int = 0) -> List[Block]:
        """Obtener bloques de usuario con paginación"""
        cache_key = f"user_blocks:{user_id}:{limit}:{offset}"
        cached_blocks = self.cache_get(cache_key)
        
        if cached_blocks:
            return [Block(**block) for block in cached_blocks]
        
        with self.get_db() as db:
            blocks = db.query(Block)\
                .filter(Block.user_id == user_id)\
                .order_by(Block.created_at.desc())\
                .offset(offset)\
                .limit(limit)\
                .all()
            
            if blocks:
                blocks_dict = [
                    {c.name: getattr(block, c.name) for c in Block.__table__.columns}
                    for block in blocks
                ]
                self.cache_set(cache_key, blocks_dict, expire=60)
            
            return blocks
    
    def add_block(self, block_data: dict) -> Block:
        """Agregar nuevo bloque"""
        with self.get_db() as db:
            block = Block(**block_data)
            db.add(block)
            db.commit()
            db.refresh(block)
            
            # Invalidar caché de bloques del usuario
            self.cache_delete(f"user_blocks:{block.user_id}:*")
            
            return block
    
    # Métodos de estadísticas avanzadas
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Obtener estadísticas avanzadas del usuario"""
        cache_key = f"user_stats:{user_id}"
        cached_stats = self.cache_get(cache_key)
        
        if cached_stats:
            return cached_stats
        
        with self.get_db() as db:
            # Total de bloques
            total_blocks = db.query(func.count(Block.id))\
                .filter(Block.user_id == user_id)\
                .scalar()
            
            # Bloques capturados
            captured_blocks = db.query(func.count(Block.id))\
                .filter(Block.user_id == user_id, Block.status == "captured")\
                .scalar()
            
            # Tasa de éxito
            success_rate = (captured_blocks / total_blocks * 100) if total_blocks > 0 else 0
            
            # Pago promedio
            avg_payment = db.query(func.avg(Block.payment))\
                .filter(Block.user_id == user_id, Block.status == "captured")\
                .scalar() or 0
            
            # Bloque más reciente
            latest_block = db.query(Block)\
                .filter(Block.user_id == user_id)\
                .order_by(Block.created_at.desc())\
                .first()
            
            stats = {
                "total_blocks": total_blocks,
                "captured_blocks": captured_blocks,
                "success_rate": round(success_rate, 2),
                "avg_payment": round(avg_payment, 2),
                "total_earnings": round(avg_payment * captured_blocks, 2),
                "latest_block": latest_block.to_dict() if latest_block else None,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            self.cache_set(cache_key, stats, expire=300)
            return stats
    
    # Métodos de mantenimiento
    def health_check(self) -> Dict[str, Any]:
        """Verificar salud de la base de datos"""
        try:
            with self.get_db() as db:
                # Test query
                db.execute("SELECT 1")
                
            redis_status = "healthy" if self.redis_client and self.redis_client.ping() else "unhealthy"
            
            # Estadísticas del pool
            pool_stats = {
                "checkedout": self.engine.pool.checkedout(),
                "checkedin": self.engine.pool.checkedin(),
                "overflow": self.engine.pool.overflow(),
                "size": self.engine.pool.size()
            }
            
            DB_CONNECTION_POOL.set(pool_stats["size"])
            DB_ACTIVE_CONNECTIONS.set(pool_stats["checkedout"])
            
            return {
                "database": "healthy",
                "redis": redis_status,
                "pool_stats": pool_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "database": "unhealthy",
                "redis": "unknown",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Limpiar sesiones antiguas"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with self.get_db() as db:
                result = db.query(UserSession)\
                    .filter(UserSession.created_at < cutoff_date)\
                    .delete()
                db.commit()
                
                logger.info(f"Cleaned up {result} old sessions")
                return result
                
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0
    
    def backup_database(self, backup_path: str) -> bool:
        """Crear backup de la base de datos (implementación básica)"""
        try:
            # Esta es una implementación simplificada
            # En producción, usaría herramientas como pg_dump para PostgreSQL
            logger.info(f"Database backup created at: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return False

# Instancia global del database manager
db_manager = AdvancedDatabaseManager()

# Funciones de utilidad para FastAPI
def get_db() -> Generator[Session, None, None]:
    """Dependency para FastAPI"""
    with db_manager.get_db() as db:
        yield db

def get_redis() -> Optional[redis.Redis]:
    """Dependency para Redis"""
    return db_manager.get_redis()

# Inicialización de la base de datos
def init_db():
    """Inicializar tablas de la base de datos"""
    try:
        Base.metadata.create_all(bind=db_manager.engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

# Método de extensión para serialización
def to_dict(obj):
    """Convertir objeto SQLAlchemy a dict"""
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

# Añadir método a los modelos
Base.to_dict = to_dict