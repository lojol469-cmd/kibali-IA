#!/bin/bash
# Script d'installation des dépendances DUSt3R sans casser l'environnement existant

echo "🔧 Installation des dépendances DUSt3R..."
echo "================================================"

# Vérifier Python et PyTorch existants
echo "✓ Python: $(python3 --version)"
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || echo "⚠️  PyTorch pas détecté"

echo ""
echo "📦 Installation des dépendances DUSt3R depuis requirements.txt..."

# Installer les dépendances DUSt3R
cd /home/belikan/dust3r
pip install -r requirements.txt --quiet --no-deps 2>&1 | grep -E "Successfully|ERROR" || true

echo ""
echo "📦 Installation des dépendances supplémentaires pour l'app..."

# Installer les dépendances de l'application (qui ne sont pas déjà installées)
pip install --quiet \
    streamlit \
    plotly \
    open3d \
    scikit-learn \
    transformers \
    pandas \
    psutil \
    nvidia-ml-py3 \
    faiss-cpu 2>&1 | grep -E "Successfully|Requirement already|ERROR" || echo "✓ Installation terminée"

echo ""
echo "================================================"
echo "✅ Installation terminée !"
echo ""
echo "Pour tester DUSt3R:"
echo "  cd /home/belikan/kibali-IA"
echo "  streamlit run app_dust3r.py"
echo ""
