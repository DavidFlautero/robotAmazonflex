from datetime import datetime
from enum import Enum
import asyncpg
from typing import Optional

class UserStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"

class SubscriptionPlan(Enum):
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

async def init_db(pool: asyncpg.Pool):
    async with pool.acquire() as connection:
        # Tabla de usuarios
        await connection.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                status VARCHAR(20) DEFAULT 'trial',
                subscription_plan VARCHAR(20) DEFAULT 'basic',
                location VARCHAR(100) DEFAULT 'US',
                created_at TIMESTAMP DEFAULT NOW(),
                last_login TIMESTAMP,
                trial_ends_at TIMESTAMP,
                stripe_customer_id VARCHAR(255),
                CONSTRAINT valid_status CHECK (status IN ('active', 'suspended', 'trial')),
                CONSTRAINT valid_plan CHECK (subscription_plan IN ('basic', 'pro', 'enterprise'))
            )
        ''')
        
        # Tabla de sesiones
        await connection.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                session_token VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                expires_at TIMESTAMP NOT NULL,
                ip_address VARCHAR(45),
                user_agent TEXT,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # Tabla de capturas de bloques
        await connection.execute('''
            CREATE TABLE IF NOT EXISTS block_captures (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                blocks_captured INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT NOW(),
                success BOOLEAN NOT NULL,
                response_time INTEGER,
                proxy_used VARCHAR(255),
                anomaly_score FLOAT,
                evasion_strategies TEXT[]
            )
        ''')
        
        # Tabla de configuración de usuarios
        await connection.execute('''
            CREATE TABLE IF NOT EXISTS user_configurations (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                min_payment INTEGER DEFAULT 18,
                max_distance INTEGER DEFAULT 20,
                preferred_locations TEXT[],
                working_hours JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        
        # Índices para mejor performance
        await connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_block_captures_user_id 
            ON block_captures(user_id, timestamp)
        ''')
        
        await connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_users_status 
            ON users(status, subscription_plan)
        ''')