#!/bin/bash
# Script de build Docker pour Kibali IA v2.0
# Compatible PyTorch 2.10 + CUDA 13.0

set -e  # Arrêter en cas d'erreur

echo "=================================================="
echo "🐋 Build Docker - Kibali IA v2.0 Streaming"
echo "=================================================="
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables
IMAGE_NAME="kibali-ia"
IMAGE_TAG="2.0-streaming"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Fonction de log
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Vérifications préalables
log_info "Vérification de l'environnement..."

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker n'est pas installé !"
    exit 1
fi
log_success "Docker installé"

# Vérifier NVIDIA Docker (optionnel)
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU détecté"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log_warning "Aucun GPU NVIDIA détecté - L'image utilisera le CPU"
fi

# Vérifier les fichiers requis
log_info "Vérification des fichiers..."
required_files=("Dockerfile" "requirements.txt" "app.py" ".env")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        log_error "Fichier manquant: $file"
        exit 1
    fi
done
log_success "Tous les fichiers requis sont présents"

# Vérifier le fichier .env
if [ ! -f ".env" ]; then
    log_warning "Fichier .env manquant"
    log_info "Création d'un fichier .env template..."
    cat > .env << EOF
HF_TOKEN=hf_your_token_here
TAVILY_API_KEY=your_tavily_api_key_here
EOF
    log_warning "⚠️  Veuillez éditer .env avec vos vraies clés API avant de démarrer !"
fi

# Nettoyer les anciennes images (optionnel)
read -p "Nettoyer les anciennes images Docker? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Nettoyage des anciennes images..."
    docker image prune -f
    log_success "Images nettoyées"
fi

# Build de l'image
log_info "Début du build de l'image Docker..."
log_info "Image: ${FULL_IMAGE}"
echo ""

# Options de build
BUILD_OPTS=(
    --tag "${FULL_IMAGE}"
    --tag "${IMAGE_NAME}:latest"
    --file Dockerfile
    --progress=plain
)

# Ajouter le build multi-plateforme si buildx est disponible
if docker buildx version &> /dev/null; then
    log_info "BuildKit détecté - Utilisation de docker buildx"
    BUILD_CMD="docker buildx build"
else
    BUILD_CMD="docker build"
fi

# Lancer le build
log_info "Commande: ${BUILD_CMD} ${BUILD_OPTS[*]} ."
echo ""

if ${BUILD_CMD} "${BUILD_OPTS[@]}" .; then
    log_success "✅ Build réussi !"
else
    log_error "❌ Échec du build"
    exit 1
fi

echo ""
log_info "Informations sur l'image:"
docker images | grep "${IMAGE_NAME}"

echo ""
log_success "=================================================="
log_success "🎉 Build terminé avec succès !"
log_success "=================================================="
echo ""
log_info "Pour lancer l'application:"
echo ""
echo "  1. Avec docker-compose:"
echo "     ${GREEN}docker-compose up -d${NC}"
echo ""
echo "  2. Avec docker run (GPU):"
echo "     ${GREEN}docker run -d --gpus all -p 8501:8501 --env-file .env ${FULL_IMAGE}${NC}"
echo ""
echo "  3. Avec docker run (CPU):"
echo "     ${GREEN}docker run -d -p 8501:8501 --env-file .env ${FULL_IMAGE}${NC}"
echo ""
log_info "Accès à l'application: http://localhost:8501"
echo ""
