# 🚀 DÉPLOIEMENT KIBALI-IA SUR NAS SYNOLOGY

## 📦 Modèles IA Inclus (1.4GB)

### 🔤 OCR (Reconnaissance de texte)
- **EasyOCR** 
  - `latin_g2.pth` - Modèle de reconnaissance Latin
  - `craft_mlt_25k.pth` - Détection de texte CRAFT
  - Taille: ~150MB

### 🔍 Embeddings & RAG
- **Sentence Transformers**
  - `all-MiniLM-L6-v2` - Embeddings pour recherche sémantique
  - Utilisé pour: RAG vectoriel, mémoire conversationnelle
  - Taille: ~90MB

### 👁️ Vision IA
- **CLIP (OpenAI)**
  - `clip-vit-base-patch32` - Vision-Language model
  - Utilisé pour: Analyse d'images, description automatique
  - Taille: ~600MB

### 🧠 Analyseurs Spécialisés
- **ERT Analyzer** - Analyse géophysique (Electrical Resistivity Tomography)
- **Binary File Handler** - Analyse de fichiers binaires
- **Universal File Analyzer** - Détection automatique de formats

## 🚀 DÉPLOIEMENT SUR NAS

### Étape 1: Copie vers Windows/NAS

```bash
# Copier le projet vers un dossier Windows accessible par le NAS
./copy_to_nas.sh /mnt/c/Users/VotreNom/Desktop/kibali-nas
```

### Étape 2: Upload vers NAS

1. Ouvrir **File Station** sur Synology
2. Créer un dossier `/docker/kibali-ia/`
3. Uploader tout le contenu de `kibali-nas/` vers ce dossier

### Étape 3: Configuration SSH sur NAS

```bash
# Se connecter au NAS
ssh admin@votre-nas.local

# Naviguer vers le dossier
cd /volume1/docker/kibali-ia/

# Créer le fichier .env avec vos clés API
nano .env
```

Contenu du `.env`:
```env
HUGGINGFACE_TOKEN=hf_votre_token_ici
MISTRAL_API_KEY=votre_cle_mistral_ici
```

### Étape 4: Build Docker sur NAS

```bash
# Construire l'image Docker (avec tous les modèles)
sudo docker-compose build

# Lancer le conteneur
sudo docker-compose up -d

# Vérifier les logs
sudo docker-compose logs -f
```

### Étape 5: Accéder à l'application

URL: `http://votre-nas.local:8501`

## 📊 SPÉCIFICATIONS TECHNIQUES

### Ressources NAS Recommandées
- **RAM**: Minimum 8GB (16GB recommandé)
- **CPU**: 4 cores minimum
- **Stockage**: 5GB pour l'application + modèles
- **Docker**: Version 20.10+

### Ports Exposés
- **8501**: Interface Streamlit principale
- **8502**: Interface secondaire (si multi-instances)

### Volumes Docker
```yaml
volumes:
  - ./kibali_data:/app/kibali_data          # Données persistantes
  - ./outils:/app/outils                    # Outils
  - ./tools:/app/tools                      # Tools
```

## 🔧 MODÈLES DANS LE BUILD DOCKER

### ✅ Modèles Locaux (Inclus dans l'image)

Les modèles suivants seront **intégrés dans l'image Docker** lors du build:

1. **EasyOCR** (`kibali_data/models/easyocr/`)
   - latin_g2.pth
   - craft_mlt_25k.pth

2. **Sentence Transformers** (`kibali_data/models/sentence-transformers/`)
   - all-MiniLM-L6-v2 (complet)

3. **CLIP** (`kibali_data/models/clip/`)
   - clip-vit-base-patch32 (complet)

4. **Analyseurs** (`kibali_data/models/`)
   - ert_analyzer.py
   - master_orchestrator.py
   - binary_file_handler.py

### 📥 Modèles API (Téléchargés à la demande)

Ces modèles utilisent des API externes et ne sont **PAS** inclus dans l'image:

- **Qwen 2.5** (via API Mistral/HuggingFace)
- **Mistral** (via API Mistral)
- **YOLO** (téléchargé au premier lancement si nécessaire)

## 🐳 DOCKERFILE - COPIE DES MODÈLES

Le Dockerfile copie automatiquement les modèles:

```dockerfile
# Copier les modèles locaux
COPY kibali_data/models/ /app/kibali_data/models/

# Les modèles seront disponibles dans le conteneur
RUN ls -lh /app/kibali_data/models/
```

## ⚡ OPTIMISATIONS NAS

### Cache des Modèles
Les modèles sont stockés dans `/app/kibali_data/models/` dans le conteneur, ce qui évite de les re-télécharger à chaque redémarrage.

### Persistance des Données
- Base vectorielle RAG: `/app/kibali_data/vectordb/`
- Historique chat: `/app/kibali_data/chat_history.db`
- PDFs uploadés: `/app/kibali_data/pdfs/`

### Performance
Pour améliorer les performances sur NAS:
```bash
# Utiliser plus de workers
docker-compose up -d --scale kibali-ia=2
```

## 🔍 VÉRIFICATION POST-DÉPLOIEMENT

```bash
# Vérifier que les modèles sont présents
docker exec kibali-ia-app-1 ls -lh /app/kibali_data/models/

# Vérifier l'utilisation mémoire
docker stats kibali-ia-app-1

# Tester l'OCR
docker exec kibali-ia-app-1 python3 -c "from vision_utils import load_ocr_reader; print('OCR OK')"

# Tester CLIP
docker exec kibali-ia-app-1 python3 -c "from vision_utils import load_vision_models; print('CLIP OK')"
```

## 📝 NOTES IMPORTANTES

1. **Tous les modèles locaux** (1.4GB) seront **inclus dans l'image Docker**
2. L'image Docker finale fera environ **3-4GB** (base + modèles + dépendances)
3. Le premier build prendra **10-20 minutes** selon les performances du NAS
4. Les modèles **ne seront PAS re-téléchargés** à chaque redémarrage

## 🆘 DÉPANNAGE

### Erreur: Modèle manquant
```bash
# Re-copier les modèles
docker cp kibali_data/models/. kibali-ia-app-1:/app/kibali_data/models/
```

### Erreur: Mémoire insuffisante
```bash
# Augmenter la mémoire Docker dans Synology DSM
# Container Manager > Settings > Docker > Memory Limit
```

### Erreur: Build échoue
```bash
# Nettoyer et rebuilder
docker-compose down
docker system prune -a
docker-compose build --no-cache
```
