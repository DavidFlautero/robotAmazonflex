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
#!/bin/bash

# Script de despliegue avanzado para Amazon Flex Bot
set -euo pipefail

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
readonly NAMESPACE="flex-production"
readonly CONFIG_DIR="./kubernetes"
readonly SCRIPTS_DIR="./scripts"
readonly BACKUP_DIR="./backups"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly LOG_FILE="${BACKUP_DIR}/deploy_${TIMESTAMP}.log"

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
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Funciones de validación
check_dependencies() {
    log "INFO" "Verificando dependencias..."
    
    local deps=("kubectl" "docker" "jq" "curl")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log "ERROR" "Dependencias faltantes: ${missing_deps[*]}"
        exit 1
    fi
}

validate_kubernetes() {
    log "INFO" "Validando configuración de Kubernetes..."
    
    if ! kubectl cluster-info &> /dev/null; then
        log "ERROR" "No se puede conectar al cluster de Kubernetes"
        exit 1
    fi
    
    # Verificar que el contexto sea el correcto (opcional)
    local current_context
    current_context=$(kubectl config current-context)
    log "INFO" "Contexto de Kubernetes: $current_context"
}

validate_configs() {
    log "INFO" "Validando archivos de configuración..."
    
    local config_files=(
        "${CONFIG_DIR}/advanced-deployment.yaml"
        "${CONFIG_DIR}/service.yaml"
        "${CONFIG_DIR}/configmap.yaml"
        "${CONFIG_DIR}/secret.yaml"
        "${CONFIG_DIR}/hpa.yaml"
        "${CONFIG_DIR}/prometheus-rules.yaml"
    )
    
    for config_file in "${config_files[@]}"; do
        if [ ! -f "$config_file" ]; then
            log "ERROR" "Archivo de configuración faltante: $config_file"
            exit 1
        fi
        
        # Validación básica de YAML
        if ! yq eval '.' "$config_file" &> /dev/null; then
            log "ERROR" "Archivo YAML inválido: $config_file"
            exit 1
        fi
    done
}

# Funciones de backup
create_backup() {
    log "INFO" "Creando backup de la configuración actual..."
    
    mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"
    
    # Backup de recursos de Kubernetes
    kubectl get all -n "$NAMESPACE" -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/k8s_resources.yaml" 2>/dev/null || true
    kubectl get configmap -n "$NAMESPACE" -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/configmaps.yaml" 2>/dev/null || true
    kubectl get secret -n "$NAMESPACE" -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/secrets.yaml" 2>/dev/null || true
    
    # Backup de configuraciones actuales
    cp -r "$CONFIG_DIR" "${BACKUP_DIR}/${TIMESTAMP}/kubernetes_configs/"
    
    log "SUCCESS" "Backup creado en ${BACKUP_DIR}/${TIMESTAMP}"
}

# Funciones de despliegue
deploy_infrastructure() {
    log "INFO" "Desplegando infraestructura..."
    
    # Crear namespace si no existe
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Aplicar configuraciones
    kubectl apply -f "${CONFIG_DIR}/configmap.yaml" -n "$NAMESPACE"
    kubectl apply -f "${CONFIG_DIR}/secret.yaml" -n "$NAMESPACE"
    
    # Aplicar reglas de Prometheus
    kubectl apply -f "${CONFIG_DIR}/prometheus-rules.yaml" -n "$NAMESPACE"
}

deploy_application() {
    log "INFO" "Desplegando aplicación..."
    
    # Aplicar deployment y servicios
    kubectl apply -f "${CONFIG_DIR}/advanced-deployment.yaml" -n "$NAMESPACE"
    kubectl apply -f "${CONFIG_DIR}/service.yaml" -n "$NAMESPACE"
    kubectl apply -f "${CONFIG_DIR}/hpa.yaml" -n "$NAMESPACE"
}

wait_for_rollout() {
    log "INFO" "Esperando a que el despliegue se complete..."
    
    local timeout=300
    local start_time=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            log "ERROR" "Timeout esperando por el despliegue"
            exit 1
        fi
        
        if kubectl rollout status deployment/amazon-flex-bot -n "$NAMESPACE" --timeout=60s &> /dev/null; then
            log "SUCCESS" "Despliegue completado exitosamente"
            break
        else
            log "WARNING" "Despliegue aún en progreso, esperando..."
            sleep 10
        fi
    done
}

# Funciones de verificación
verify_deployment() {
    log "INFO" "Verificando el despliegue..."
    
    # Verificar pods
    local pods_ready
    pods_ready=$(kubectl get pods -n "$NAMESPACE" -l app=amazon-flex-bot -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | grep -c "True")
    
    if [ "$pods_ready" -eq 0 ]; then
        log "ERROR" "Ningún pod está listo"
        exit 1
    fi
    
    log "SUCCESS" "$pods_ready pods están listos"
    
    # Verificar servicios
    if ! kubectl get service/amazon-flex-bot-service -n "$NAMESPACE" &> /dev/null; then
        log "ERROR" "Servicio no encontrado"
        exit 1
    fi
    
    log "SUCCESS" "Servicio desplegado correctamente"
}

run_tests() {
    log "INFO" "Ejecutando tests post-despliegue..."
    
    # Ejecutar tests de health check
    if ! "${SCRIPTS_DIR}/health_check.sh"; then
        log "ERROR" "Los tests de health check fallaron"
        exit 1
    fi
    
    log "SUCCESS" "Todos los tests pasaron"
}

# Función principal
main() {
    log "INFO" "Iniciando despliegue de Amazon Flex Bot Advanced..."
    log "INFO" "Timestamp: $TIMESTAMP"
    
    # Crear directorio de backups si no existe
    mkdir -p "$BACKUP_DIR"
    
    # Validaciones
    check_dependencies
    validate_kubernetes
    validate_configs
    
    # Backup
    create_backup
    
    # Despliegue
    deploy_infrastructure
    deploy_application
    wait_for_rollout
    verify_deployment
    
    # Tests
    run_tests
    
    # Mostrar información del despliegue
    log "SUCCESS" "Despliegue completado exitosamente!"
    log "INFO" "Resumen:"
    kubectl get all -n "$NAMESPACE"
    
    # Mostrar URL del servicio si está disponible
    local service_ip
    service_ip=$(kubectl get service amazon-flex-bot-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -n "$service_ip" ]; then
        log "INFO" "Servicio disponible en: http://${service_ip}:8080"
    fi
    
    log "INFO" "Logs del despliegue: $LOG_FILE"
}

# Manejo de señales
trap 'log "ERROR" "Script interrumpido por el usuario"; exit 1' INT TERM

# Ejecutar función principal
main "$@"