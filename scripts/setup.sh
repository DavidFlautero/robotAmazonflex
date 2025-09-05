#!/bin/bash

# Script de configuración avanzado para Amazon Flex Bot
set -euo pipefail

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly CONFIG_DIR="${PROJECT_ROOT}/kubernetes"
readonly SETUP_LOG="${PROJECT_ROOT}/setup.log"

# Funciones de logging
log() {
    local level=$1
    shift
    local message=$*
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO") echo -e "${BLUE}[INFO]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "WARNING") echo -e "${YELLOW}[WARNING]${NC} $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
    esac
    
    echo "[$timestamp] [$level] $message" >> "$SETUP_LOG"
}

# Funciones de verificación
check_os() {
    log "INFO" "Verificando sistema operativo..."
    
    if [[ "$(uname)" != "Linux" ]] && [[ "$(uname)" != "Darwin" ]]; then
        log "ERROR" "Sistema operativo no soportado: $(uname)"
        exit 1
    fi
    
    log "SUCCESS" "Sistema operativo compatible: $(uname)"
}

check_dependencies() {
    log "INFO" "Verificando dependencias del sistema..."
    
    local deps=("docker" "kubectl" "python3" "pip3" "jq" "curl" "git")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log "WARNING" "Dependencias faltantes: ${missing_deps[*]}"
        return 1
    fi
    
    log "SUCCESS" "Todas las dependencias están instaladas"
    return 0
}

install_dependencies() {
    log "INFO" "Instalando dependencias faltantes..."
    
    local os_type
    os_type=$(uname)
    
    if [[ "$os_type" == "Linux" ]]; then
        # Detectar distribución Linux
        if [ -f /etc/debian_version ]; then
            # Debian/Ubuntu
            log "INFO" "Instalando dependencias para Debian/Ubuntu"
            sudo apt update
            sudo apt install -y docker.io kubectl python3 python3-pip jq curl git
        elif [ -f /etc/redhat-release ]; then
            # RHEL/CentOS
            log "INFO" "Instalando dependencias para RHEL/CentOS"
            sudo yum install -y docker kubectl python3 python3-pip jq curl git
        else
            log "ERROR" "Distribución Linux no soportada"
            exit 1
        fi
    elif [[ "$os_type" == "Darwin" ]]; then
        # macOS
        log "INFO" "Instalando dependencias para macOS"
        if ! command -v brew &> /dev/null; then
            log "ERROR" "Homebrew no instalado. Por favor instala Homebrew primero"
            exit 1
        fi
        brew install docker kubectl python3 jq curl git
    fi
    
    log "SUCCESS" "Dependencias instaladas correctamente"
}

setup_python_environment() {
    log "INFO" "Configurando entorno Python..."
    
    # Crear virtual environment si no existe
    if [ ! -d "${PROJECT_ROOT}/venv" ]; then
        python3 -m venv "${PROJECT_ROOT}/venv"
        log "SUCCESS" "Entorno virtual creado"
    fi
    
    # Activar virtual environment
    source "${PROJECT_ROOT}/venv/bin/activate"
    
    # Instalar dependencias Python
    if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
        pip install -r "${PROJECT_ROOT}/requirements.txt"
        log "SUCCESS" "Dependencias Python instaladas"
    else
        log "WARNING" "Archivo requirements.txt no encontrado"
    fi
}

setup_docker() {
    log "INFO" "Configurando Docker..."
    
    # Iniciar servicio Docker si no está corriendo
    if ! docker info &> /dev/null; then
        log "INFO" "Iniciando servicio Docker..."
        if [[ "$(uname)" == "Linux" ]]; then
            sudo systemctl start docker
            sudo systemctl enable docker
        elif [[ "$(uname)" == "Darwin" ]]; then
            open -a Docker
            # Esperar a que Docker inicie
            while ! docker info &> /dev/null; do
                sleep 2
            done
        fi
    fi
    
    # Agregar usuario al grupo docker (Linux)
    if [[ "$(uname)" == "Linux" ]] && ! groups | grep -q docker; then
        sudo usermod -aG docker "$USER"
        log "INFO" "Usuario agregado al grupo docker. Por favor cierra sesión y vuelve a iniciar"
    fi
    
    log "SUCCESS" "Docker configurado correctamente"
}

setup_kubernetes() {
    log "INFO" "Configurando Kubernetes..."
    
    # Verificar si kubectl está configurado
    if ! kubectl cluster-info &> /dev/null; then
        log "WARNING" "Kubernetes no está configurado o no se puede conectar al cluster"
        
        # Intentar configurar Minikube si no hay cluster
        if command -v minikube &> /dev/null; then
            log "INFO" "Iniciando Minikube..."
            minikube start --driver=docker
            minikube addons enable metrics-server
        else
            log "ERROR" "Minikube no instalado. Por favor instala Minikube o configura un cluster de Kubernetes"
            exit 1
        fi
    fi
    
    log "SUCCESS" "Kubernetes configurado correctamente"
    log "INFO" "Cluster info: $(kubectl cluster-info | head -n 1)"
}

setup_environment() {
    log "INFO" "Configurando variables de entorno..."
    
    # Crear archivo .env si no existe
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
        log "INFO" "Archivo .env creado. Por favor configura las variables de entorno"
    else
        log "INFO" "Archivo .env ya existe"
    fi
    
    # Cargar variables de entorno
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
    fi
}

generate_secrets() {
    log "INFO" "Generando secrets para Kubernetes..."
    
    # Verificar si el archivo secret.yaml ya existe
    if [ -f "${CONFIG_DIR}/secret.yaml" ]; then
        log "INFO" "Archivo secret.yaml ya existe"
        return 0
    fi
    
    # Crear directorio si no existe
    mkdir -p "$CONFIG_DIR"
    
    # Generar secrets básicos
    cat > "${CONFIG_DIR}/secret.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: flex-bot-secrets
  namespace: flex-production
type: Opaque
data:
  # Generar valores con: echo -n "valor" | base64
  secret-key: $(echo -n "$(openssl rand -hex 32)" | base64)
  database-url: $(echo -n "postgresql://user:pass@postgres:5432/flexbot" | base64)
  redis-url: $(echo -n "redis://redis:6379/0" | base64)
EOF
    
    log "SUCCESS" "Archivo secret.yaml generado. Por favor completa los valores faltantes"
}

run_health_check() {
    log "INFO" "Ejecutando health check inicial..."
    
    if "${SCRIPT_DIR}/health_check.sh" --quick; then
        log "SUCCESS" "Health check pasado exitosamente"
    else
        log "WARNING" "Health check falló. Por favor verifica la configuración"
    fi
}

display_next_steps() {
    log "INFO" "================================================"
    log "INFO" "CONFIGURACIÓN COMPLETADA - PRÓXIMOS PASOS"
    log "INFO" "================================================"
    log "INFO" "1. Configura las variables de entorno en .env"
    log "INFO" "2. Completa los valores en kubernetes/secret.yaml"
    log "INFO" "3. Ejecuta el script de despliegue: ./scripts/deploy.sh"
    log "INFO" "4. Verifica el estado: ./scripts/health_check.sh"
    log "INFO" "================================================"
}

# Función principal
main() {
    log "INFO" "Iniciando configuración de Amazon Flex Bot Advanced..."
    
    # Verificar sistema operativo
    check_os
    
    # Verificar e instalar dependencias
    if ! check_dependencies; then
        install_dependencies
    fi
    
    # Configurar entorno
    setup_python_environment
    setup_docker
    setup_kubernetes
    setup_environment
    generate_secrets
    
    # Health check inicial
    run_health_check
    
    # Mostrar próximos pasos
    display_next_steps
    
    log "SUCCESS" "Configuración completada exitosamente!"
}

# Ejecutar función principal
main "$@"