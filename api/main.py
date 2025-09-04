from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncpg
from typing import Dict, List
import os
from datetime import datetime
import logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from database.models import User, Session, BlockCapture
from security.zero_trust import ZeroTrustSecurity
from evasion.advanced_ml_evasion import AdvancedBehavioralMLDetector
from network.ai_proxy_manager import AdvancedAIProxyManager

logger = logging.getLogger(__name__)

app = FastAPI(title="Amazon Flex Bot API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
zt_security = ZeroTrustSecurity(os.getenv('SECRET_KEY', 'default-secret'))
evasion_system = AdvancedBehavioralMLDetector()
proxy_manager = AdvancedAIProxyManager()

@app.on_event("startup")
async def startup():
    app.state.db = await asyncpg.create_pool(os.getenv('DATABASE_URL'))
    await proxy_manager.start_health_checks()
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()
    await proxy_manager.close()
    logger.info("Application shutdown")

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/capture-blocks")
async def capture_blocks(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_data: Dict = None
):
    try:
        user_id = zt_security.verify_token(credentials.credentials)
        
        async with app.state.db.acquire() as connection:
            user = await connection.fetchrow(
                "SELECT * FROM users WHERE id = $1", user_id
            )
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            session_data = {
                'user_id': user_id,
                'action': 'capture_blocks',
                'timestamp': datetime.utcnow()
            }
            
            anomaly_score, strategies = await evasion_system.analyze_behavior_pattern(session_data)
            
            proxy = await proxy_manager.get_optimal_proxy(
                "https://flex.amazon.com",
                {"user_id": user_id, "location": user['location']}
            )
            
            if not proxy:
                raise HTTPException(status_code=503, detail="No proxies available")
            
            result = await execute_amazon_flex_request(proxy, user)
            
            await connection.execute(
                """INSERT INTO block_captures 
                (user_id, blocks_captured, timestamp, success) 
                VALUES ($1, $2, $3, $4)""",
                user_id, result['blocks_captured'], datetime.utcnow(), result['success']
            )
            
            return {
                "success": True,
                "blocks_captured": result['blocks_captured'],
                "anomaly_score": anomaly_score,
                "evasion_strategies": [s.name for s in strategies]
            }
            
    except Exception as e:
        logger.error(f"Error capturing blocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_amazon_flex_request(proxy, user):
    # Implementación de la lógica específica de Amazon Flex
    # Esta parte requiere reverse engineering actualizado
    return {"success": True, "blocks_captured": 3}

@app.get("/api/user/stats")
async def get_user_stats(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    user_id = zt_security.verify_token(credentials.credentials)
    
    async with app.state.db.acquire() as connection:
        stats = await connection.fetchrow(
            """SELECT 
                COUNT(*) as total_captures,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_captures,
                AVG(CASE WHEN success THEN 1 ELSE 0 END) as success_rate
            FROM block_captures 
            WHERE user_id = $1""",
            user_id
        )
        
        return {
            "total_captures": stats['total_captures'],
            "successful_captures": stats['successful_captures'],
            "success_rate": float(stats['success_rate']) if stats['success_rate'] else 0
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)