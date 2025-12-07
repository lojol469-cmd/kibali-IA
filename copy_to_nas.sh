#!/bin/bash

# Script de copie de Kibali-IA vers NAS Synology
# Usage: ./copy_to_nas.sh /mnt/c/Users/YourUser/Desktop/kibali-nas

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  📦 COPIE KIBALI-IA VERS NAS SYNOLOGY                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Vérifier le dossier de destination
if [ -z "$1" ]; then
    echo "❌ Erreur: Veuillez spécifier le dossier de destination"
    echo "Usage: ./copy_to_nas.sh /mnt/c/Users/YourUser/Desktop/kibali-nas"
    exit 1
fi

DEST_DIR="$1"
SOURCE_DIR="/home/belikan/kibali-IA"

# Créer le dossier de destination
echo "📁 Création du dossier de destination: $DEST_DIR"
mkdir -p "$DEST_DIR"

# Liste des fichiers/dossiers à exclure
echo "⚙️  Préparation de la copie (exclusion des fichiers temporaires)..."
echo ""

# Copie avec rsync pour exclure certains fichiers
rsync -av --progress \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='lifemodo.pid' \
  --exclude='*.log' \
  --exclude='outputs/' \
  --exclude='kibali_data/chat_history.db' \
  --exclude='kibali_data/vectordb/' \
  --exclude='kibali_data/graphs/' \
  --exclude='kibali_data/pdfs/' \
  "$SOURCE_DIR/" "$DEST_DIR/"

echo ""
echo "✅ Copie terminée!"
echo ""
echo "📊 RÉSUMÉ:"
du -sh "$DEST_DIR"
echo ""
echo "📋 MODÈLES INCLUS DANS LA COPIE:"
echo "   ✓ EasyOCR (latin_g2.pth, craft_mlt_25k.pth)"
echo "   ✓ Sentence Transformers (all-MiniLM-L6-v2)"
echo "   ✓ CLIP Vision (openai/clip-vit-base-patch32)"
echo "   ✓ ERT Analyzer"
echo "   ✓ Binary File Handler"
echo ""
echo "📦 TAILLE TOTALE DES MODÈLES: ~1.4GB"
echo ""
echo "🐳 PROCHAINES ÉTAPES POUR NAS SYNOLOGY:"
echo "   1. Uploader le dossier '$DEST_DIR' vers votre NAS"
echo "   2. Se connecter en SSH au NAS"
echo "   3. Naviguer vers le dossier uploadé"
echo "   4. Créer le fichier .env avec vos tokens API"
echo "   5. Lancer: docker-compose up -d --build"
echo ""
echo "🔑 VARIABLES D'ENVIRONNEMENT REQUISES (.env):"
echo "   HUGGINGFACE_TOKEN=your_token_here"
echo "   MISTRAL_API_KEY=your_key_here"
echo ""
