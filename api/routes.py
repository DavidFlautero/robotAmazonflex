from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import List, Optional
import asyncio

from app.models.user import User, UserCreate, Token
from app.models.block import Block, BlockSearch
from app.services.amazon_flex import AmazonFlexService
from app.services.payment import PaymentService
from app.utils.security import verify_password, create_access_token, get_password_hash
from app.database.database_manager import get_db, User as DBUser

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
amazon_service = AmazonFlexService()
payment_service = PaymentService()

# Endpoints de Autenticación
@router.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/auth/register", response_model=User)
async def register_user(user: UserCreate):
    db = get_db()
    # Verificar si el usuario ya existe
    existing_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Crear nuevo usuario
    hashed_password = get_password_hash(user.password)
    db_user = DBUser(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        created_at=datetime.utcnow(),
        is_active=True
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

# Endpoints de Usuarios
@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.put("/users/me", response_model=User)
async def update_user_me(user_update: User, current_user: User = Depends(get_current_user)):
    db = get_db()
    db_user = db.query(DBUser).filter(DBUser.id == current_user.id).first()
    
    if user_update.email:
        db_user.email = user_update.email
    if user_update.full_name:
        db_user.full_name = user_update.full_name
    
    db.commit()
    db.refresh(db_user)
    return db_user

# Endpoints de Bloques
@router.get("/blocks", response_model=List[Block])
async def get_blocks(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user)):
    db = get_db()
    blocks = db.query(Block).offset(skip).limit(limit).all()
    return blocks

@router.post("/blocks/search", response_model=List[Block])
async def search_blocks(search: BlockSearch, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    try:
        # Agregar user_id al contexto de búsqueda
        search_dict = search.dict()
        search_dict["user_id"] = current_user.id
        
        blocks = await amazon_service.search_blocks(search_dict)
        
        # Iniciar búsqueda en segundo plano si hay criterios de búsqueda
        if any([search.min_payment, search.location, search.start_time]):
            background_tasks.add_task(amazon_service.continuous_search, search_dict)
        
        return blocks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching blocks: {str(e)}")

@router.post("/blocks/{block_id}/capture")
async def capture_block(block_id: str, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    try:
        result = await amazon_service.capture_block(block_id, current_user.id)
        
        if result["status"] == "success":
            # Registrar bloque capturado en la base de datos
            db = get_db()
            captured_block = Block(
                block_id=block_id,
                user_id=current_user.id,
                captured_at=datetime.utcnow(),
                status="captured"
            )
            db.add(captured_block)
            db.commit()
            
            return {"status": "success", "block_id": block_id, "message": "Block captured successfully"}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error capturing block: {str(e)}")

@router.get("/blocks/stats")
async def get_block_stats(current_user: User = Depends(get_current_user)):
    db = get_db()
    
    # Obtener estadísticas de bloques
    total_blocks = db.query(Block).filter(Block.user_id == current_user.id).count()
    captured_blocks = db.query(Block).filter(Block.user_id == current_user.id, Block.status == "captured").count()
    failed_blocks = db.query(Block).filter(Block.user_id == current_user.id, Block.status == "failed").count()
    
    success_rate = (captured_blocks / total_blocks * 100) if total_blocks > 0 else 0
    
    return {
        "total_blocks": total_blocks,
        "captured_blocks": captured_blocks,
        "failed_blocks": failed_blocks,
        "success_rate": round(success_rate, 2),
        "last_updated": datetime.utcnow()
    }

# Endpoints de Pagos
@router.post("/payments/create-checkout-session")
async def create_checkout_session(price_id: str, current_user: User = Depends(get_current_user)):
    try:
        success_url = f"https://yourapp.com/success?session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = "https://yourapp.com/cancel"
        
        session = await payment_service.create_checkout_session(
            price_id, current_user.stripe_customer_id, success_url, cancel_url
        )
        
        return {"session_id": session.id, "url": session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating checkout session: {str(e)}")

@router.post("/payments/webhook")
async def handle_webhook(payload: dict, background_tasks: BackgroundTasks):
    try:
        # Verificar firma del webhook
        sig_header = request.headers.get('stripe-signature')
        result = await payment_service.handle_webhook(payload, sig_header)
        
        if result["status"] == "success":
            # Procesar eventos en segundo plano
            background_tasks.add_task(process_payment_webhook, payload)
            return {"status": "success"}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")

# Funciones auxiliares
async def authenticate_user(username: str, password: str):
    db = get_db()
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = verify_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    
    db = get_db()
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if user is None:
        raise credentials_exception
    return user

async def process_payment_webhook(payload: dict):
    # Procesar eventos de pago en segundo plano
    event_type = payload["type"]
    
    if event_type == "checkout.session.completed":
        # Actualizar estado de suscripción del usuario
        session = payload["data"]["object"]
        customer_id = session["customer"]
        subscription_id = session["subscription"]
        
        db = get_db()
        user = db.query(DBUser).filter(DBUser.stripe_customer_id == customer_id).first()
        if user:
            user.subscription_id = subscription_id
            user.subscription_status = "active"
            db.commit()