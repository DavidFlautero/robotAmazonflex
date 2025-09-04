#!/bin/bash

set -e

echo "🛠️ Configurando entorno de Amazon Flex Bot SaaS"
echo "📅 Fecha: $(date)"
echo "----------------------------------------"

# Verificar que Docker esté instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Instalando..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Verificar que Docker Compose esté instalado
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado. Instalando..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Crear directorio de trabajo
mkdir -p ~/amazon-flex-bot
cd ~/amazon-flex-bot

# Clonar repositorio si no existe
if [ ! -d ".git" ]; then
    echo "📦 Clonando repositorio..."
    git clone https://github.com/tu-repo/amazon-flex-bot.git .
else
    echo "🔄 Actualizando repositorio..."
    git pull origin main
fi

# Configurar variables de entorno
if [ ! -f ".env" ]; then
    echo "⚙️ Configurando variables de entorno..."
    cp .env.example .env
    echo "📝 Por favor edita el archivo .env con tus configuraciones"
fi

# Dar permisos de ejecución a los scripts
chmod +x scripts/*.sh

echo "✅ Configuración completada"
echo "🚀 Para iniciar la aplicación ejecuta: docker-compose up -d --build"