# Dockerfile pour Kibali IA v2.0
# Compatible PyTorch 2.10 + CUDA 13.0 avec fallback CPU automatique
# Base: NVIDIA CUDA 13.0 + Python 3.13

FROM nvidia/cuda:13.0.0-cudnn9-runtime-ubuntu22.04

# Métadonnées
LABEL maintainer="Belikan"
LABEL version="2.0-streaming"
LABEL description="Kibali IA - Assistant Multi-Fonctions avec Streaming Token"

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHON_VERSION=3.13 \
    TORCH_VERSION=2.10.0 \
    CUDA_VERSION=13.0

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    # Dépendances osmium
    libexpat1 \
    libexpat1-dev \
    libboost-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    libosmium2-dev \
    cmake \
    # Dépendances OpenCV et vision
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Créer liens symboliques Python
RUN ln -sf /usr/bin/python3.13 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.13 /usr/bin/python

# Mettre à jour pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copier requirements
COPY requirements.txt .

# Installer PyTorch 2.10 nightly avec CUDA 13.0 (compatible RTX 5090)
RUN pip install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Installer dépendances supplémentaires pour pdfplumber
RUN pip install --no-cache-dir pdfminer.six pypdfium2

# Copier le code de l'application
COPY . .

# Créer la structure de dossiers
RUN mkdir -p kibali_data/models \
             kibali_data/pdfs \
             kibali_data/graphs \
             kibali_data/vectordb

# Exposer le port Streamlit
EXPOSE 8501

# Variables d'environnement Streamlit
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Healthcheck pour vérifier que l'app tourne
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Lancer Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]