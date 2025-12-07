# 🐋 Guide Docker - Kibali IA v2.0

## 📋 Structure Docker

```
kibali-IA/
├── Dockerfile              # Image Docker avec PyTorch 2.10 + CUDA 13.0
├── docker-compose.yml      # Orchestration Docker Compose
├── docker-build.sh         # Script de build automatique
├── .dockerignore          # Fichiers à exclure du build
└── .env                   # Variables d'environnement (API keys)
```

## 🚀 Utilisation

### Option 1: Script automatique (Recommandé)

```bash
# Build de l'image
./docker-build.sh

# Lancer avec docker-compose
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter
docker-compose down
```

### Option 2: Docker Compose manuel

```bash
# Build et démarrage
docker-compose up -d --build

# Arrêt
docker-compose down

# Voir les logs
docker-compose logs -f kibali-ia
```

### Option 3: Docker run direct

**Avec GPU (NVIDIA):**
```bash
docker run -d \
  --name kibali-ia \
  --gpus all \
  -p 8501:8501 \
  -v $(pwd)/kibali_data:/app/kibali_data \
  -v $(pwd)/.env:/app/.env:ro \
  kibali-ia:2.0-streaming
```

**Avec CPU (fallback automatique):**
```bash
docker run -d \
  --name kibali-ia \
  -p 8501:8501 \
  -v $(pwd)/kibali_data:/app/kibali_data \
  -v $(pwd)/.env:/app/.env:ro \
  kibali-ia:2.0-streaming
```

## 📦 Configuration

### Variables d'environnement (.env)

```env
HF_TOKEN=hf_your_huggingface_token
TAVILY_API_KEY=your_tavily_api_key
```

### Volumes Docker

| Volume hôte | Volume container | Description |
|-------------|------------------|-------------|
| `./kibali_data/pdfs` | `/app/kibali_data/pdfs` | Documents PDF |
| `./kibali_data/graphs` | `/app/kibali_data/graphs` | Graphes OSM |
| `./kibali_data/vectordb` | `/app/kibali_data/vectordb` | Base vectorielle |
| `./kibali_data/models` | `/app/kibali_data/models` | Modèles locaux |
| `./outputs` | `/app/outputs` | Fichiers générés |

## 🎮 Support GPU

L'image supporte automatiquement :
- ✅ **NVIDIA GPU** : CUDA 13.0 (RTX 5090 compatible)
- ✅ **CPU Fallback** : Détection automatique

### Prérequis GPU

1. **NVIDIA Docker Runtime:**
```bash
# Installer nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **Tester:**
```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

## 📊 Commandes utiles

```bash
# Voir les conteneurs
docker ps

# Logs en temps réel
docker logs -f kibali-ia

# Entrer dans le conteneur
docker exec -it kibali-ia bash

# Statistiques ressources
docker stats kibali-ia

# Reconstruire sans cache
docker-compose build --no-cache

# Nettoyer
docker system prune -a
```

## 🔧 Dépannage

### Problème: "CUDA not available"
**Solution:** Vérifier NVIDIA Docker:
```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### Problème: "Permission denied .env"
**Solution:** Vérifier les permissions:
```bash
chmod 644 .env
```

### Problème: Port 8501 déjà utilisé
**Solution:** Changer le port dans docker-compose.yml:
```yaml
ports:
  - "8502:8501"  # Utiliser 8502 à la place
```

### Problème: Espace disque insuffisant
**Solution:** Nettoyer Docker:
```bash
docker system prune -a --volumes
```

## 🏗️ Architecture de l'image

```
Base: nvidia/cuda:13.0.0-cudnn9-runtime-ubuntu22.04
├── Python 3.13
├── PyTorch 2.10 (nightly) + CUDA 13.0
├── Kibali IA v2.0
│   ├── Streaming token activé
│   ├── Fallback CPU automatique
│   └── Licences commerciales OK
└── Dépendances:
    ├── Streamlit, LangChain, Transformers
    ├── pdfplumber (MIT), OpenCV, CLIP
    └── FAISS, Sentence-Transformers
```

## 📈 Performance

**Taille de l'image:** ~8-10 GB (avec PyTorch + CUDA)

**Mémoire requise:**
- CPU: 4 GB minimum, 8 GB recommandé
- GPU: 8 GB VRAM minimum, 16+ GB recommandé

**Build time:** ~15-30 minutes (première fois)

## 🌐 Production

### Docker Hub (pousser l'image)

```bash
# Tag
docker tag kibali-ia:2.0-streaming yourusername/kibali-ia:2.0

# Push
docker push yourusername/kibali-ia:2.0
```

### Kubernetes (exemple)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kibali-ia
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kibali-ia
  template:
    metadata:
      labels:
        app: kibali-ia
    spec:
      containers:
      - name: kibali-ia
        image: kibali-ia:2.0-streaming
        ports:
        - containerPort: 8501
        resources:
          limits:
            nvidia.com/gpu: 1
```

## ✅ Checklist avant build

- [ ] Fichier `.env` configuré avec les API keys
- [ ] NVIDIA Docker installé (pour GPU)
- [ ] Espace disque suffisant (20+ GB)
- [ ] Port 8501 disponible
- [ ] Connexion internet (téléchargement PyTorch)

## 📝 Notes

- L'image utilise **PyTorch 2.10 nightly** pour supporter RTX 5090
- Le **fallback CPU** est automatique si pas de GPU
- Les **modèles locaux** ne sont pas inclus (trop volumineux)
- Utiliser des **volumes** pour la persistance des données

---

**Version:** 2.0 Streaming Edition  
**Date:** 7 décembre 2025  
**Support:** GPU NVIDIA + CPU fallback
