#!/bin/bash

set -e

echo "🚀 Iniciando despliegue del Amazon Flex Bot SaaS"
echo "📅 Fecha: $(date)"
echo "----------------------------------------"

# Configuración
APP_NAME="amazon-flex-bot"
DEPLOY_DIR="/opt/$APP_NAME"
BACKUP_DIR="/opt/backups/$APP_NAME"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Crear directorios si no existen
sudo mkdir -p $DEPLOY_DIR $BACKUP_DIR
sudo chown -R $USER:$USER $DEPLOY_DIR $BACKUP_DIR

echo "📦 Creando backup de la versión actual..."
if [ -d "$DEPLOY_DIR" ]; then
    tar -czf "$BACKUP_DIR/backup_$TIMESTAMP.tar.gz" -C $DEPLOY_DIR .
fi

echo "🔄 Actualizando código desde Git..."
cd $DEPLOY_DIR
git pull origin main

echo "🐳 Reconstruyendo contenedores Docker..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d

echo "📊 Esperando a que los servicios estén ready..."
sleep 30

echo "✅ Ejecutando migraciones de base de datos..."
docker-compose exec app python -c "
from database.models import init_db
from api.main import app
import asyncio
import asyncpg

async def run_migrations():
    pool = await asyncpg.create_pool('postgresql://user:password@postgres:5432/flexbot')
    await init_db(pool)
    await pool.close()

asyncio.run(run_migrations())
"

echo "🎯 Verificando el estado del servicio..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/healthz || true)

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Despliegue completado exitosamente"
    echo "🌐 La aplicación está disponible en: http://localhost:8080"
else
    echo "❌ Error en el despliegue. Status: $HTTP_STATUS"
    echo "📋 Revertiendo a la versión anterior..."
    tar -xzf "$BACKUP_DIR/backup_$TIMESTAMP.tar.gz" -C $DEPLOY_DIR
    docker-compose up -d
    exit 1
fi

echo "----------------------------------------"
echo "🎉 Despliegue finalizado correctamente!"