"""
Configuration centralisée des chemins de modèles IA pour Kibali
Tous les modèles sont stockés dans kibali_data/models/
"""
import os
from pathlib import Path

# Chemin de base du projet
BASE_DIR = Path(__file__).parent.parent.parent  # /home/belikan/kibali-IA
MODELS_DIR = Path(__file__).parent  # /home/belikan/kibali-IA/kibali_data/models

# Cache HuggingFace centralisé
HUGGINGFACE_CACHE = MODELS_DIR / "huggingface_cache"

# ============================================================
# CONFIGURATION DES MODÈLES IA
# ============================================================

# 1. LLM Local (Qwen) - Pointe vers le cache HuggingFace
QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_CACHE_DIR = HUGGINGFACE_CACHE

# 2. Sentence Transformers (Embeddings pour RAG)
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_CACHE = HUGGINGFACE_CACHE

# 3. Vision AI (CLIP) - Pointe vers le cache HuggingFace
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_CACHE_DIR = HUGGINGFACE_CACHE

# 4. OCR (EasyOCR) - Garde son propre dossier
EASYOCR_MODEL_DIR = MODELS_DIR / "easyocr"
EASYOCR_LANGUAGES = ['fr', 'en']

# 5. NLP Pipeline Models - Tous dans le cache HuggingFace
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
SUMMARIZER_CACHE = HUGGINGFACE_CACHE

TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-fr-en"
TRANSLATOR_CACHE = HUGGINGFACE_CACHE

NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"
NER_CACHE = HUGGINGFACE_CACHE

# 6. Captioner (pour analyse d'images)
CAPTIONER_MODEL = "Salesforce/blip-image-captioning-base"
CAPTIONER_CACHE = HUGGINGFACE_CACHE

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def ensure_model_dirs():
    """Crée tous les dossiers de modèles s'ils n'existent pas"""
    dirs = [
        QWEN_CACHE_DIR,
        SENTENCE_TRANSFORMER_CACHE,
        CLIP_CACHE_DIR,
        EASYOCR_MODEL_DIR,
        SUMMARIZER_CACHE,
        TRANSLATOR_CACHE,
        NER_CACHE,
        CAPTIONER_CACHE
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Tous les dossiers de modèles sont prêts")

def get_model_info():
    """Retourne les informations sur tous les modèles configurés"""
    return {
        "LLM Local": {
            "name": QWEN_MODEL_NAME,
            "cache": str(QWEN_CACHE_DIR),
            "size": "~3 GB"
        },
        "Embeddings RAG": {
            "name": SENTENCE_TRANSFORMER_MODEL,
            "cache": str(SENTENCE_TRANSFORMER_CACHE),
            "size": "~90 MB"
        },
        "Vision AI": {
            "name": CLIP_MODEL_NAME,
            "cache": str(CLIP_CACHE_DIR),
            "size": "~600 MB"
        },
        "OCR": {
            "name": "EasyOCR",
            "cache": str(EASYOCR_MODEL_DIR),
            "languages": EASYOCR_LANGUAGES,
            "size": "~500 MB"
        },
        "Summarizer": {
            "name": SUMMARIZER_MODEL,
            "cache": str(SUMMARIZER_CACHE),
            "size": "~1.6 GB"
        },
        "Translator": {
            "name": TRANSLATOR_MODEL,
            "cache": str(TRANSLATOR_CACHE),
            "size": "~300 MB"
        },
        "NER": {
            "name": NER_MODEL,
            "cache": str(NER_CACHE),
            "size": "~1.3 GB"
        },
        "Captioner": {
            "name": CAPTIONER_MODEL,
            "cache": str(CAPTIONER_CACHE),
            "size": "~1 GB"
        }
    }

def print_model_summary():
    """Affiche un résumé des modèles configurés"""
    print("\n" + "="*60)
    print("🤖 CONFIGURATION DES MODÈLES IA - KIBALI")
    print("="*60)
    
    info = get_model_info()
    total_size = 0
    
    for model_type, details in info.items():
        print(f"\n📦 {model_type}:")
        print(f"   Nom: {details['name']}")
        print(f"   Cache: {details['cache']}")
        if 'size' in details:
            size_str = details['size']
            # Extraire la taille en GB pour le total
            if 'GB' in size_str:
                total_size += float(size_str.split('~')[1].split(' GB')[0])
            elif 'MB' in size_str:
                total_size += float(size_str.split('~')[1].split(' MB')[0]) / 1000
        if 'languages' in details:
            print(f"   Langues: {', '.join(details['languages'])}")
    
    print(f"\n{'='*60}")
    print(f"💾 Espace disque total estimé: ~{total_size:.1f} GB")
    print(f"📁 Dossier racine: {MODELS_DIR}")
    print("="*60 + "\n")

# ============================================================
# MIGRATION ET MAINTENANCE
# ============================================================

def migrate_models_from_cache():
    """
    Migration des modèles depuis ~/.cache/huggingface vers kibali_data/models
    Note: Cette fonction est informative, la migration réelle se fait par symlinks
    """
    import subprocess
    
    home_cache = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not home_cache.exists():
        print("⚠️ Aucun cache HuggingFace trouvé dans ~/.cache")
        return
    
    print("📋 Modèles détectés dans ~/.cache/huggingface/hub:")
    
    # Liste des modèles
    models_found = []
    for model_dir in home_cache.glob("models--*"):
        model_name = model_dir.name.replace("models--", "").replace("--", "/")
        size = subprocess.check_output(['du', '-sh', str(model_dir)]).split()[0].decode()
        models_found.append((model_name, size, model_dir))
        print(f"  - {model_name} ({size})")
    
    print(f"\n💡 Pour économiser l'espace, créez des symlinks:")
    print(f"   ln -s {home_cache} {MODELS_DIR}/huggingface_cache")
    
    return models_found

def check_disk_space():
    """Vérifie l'espace disque disponible"""
    import shutil
    
    total, used, free = shutil.disk_usage(MODELS_DIR)
    
    print(f"\n💾 Espace disque sur {MODELS_DIR.parent}:")
    print(f"   Total: {total // (2**30)} GB")
    print(f"   Utilisé: {used // (2**30)} GB")
    print(f"   Libre: {free // (2**30)} GB")
    
    if free < 10 * (2**30):  # < 10 GB
        print("⚠️ ATTENTION: Moins de 10 GB disponibles!")
    else:
        print("✅ Espace disque suffisant")

# ============================================================
# INITIALISATION
# ============================================================

if __name__ == "__main__":
    print("🚀 Initialisation de la configuration des modèles IA")
    ensure_model_dirs()
    print_model_summary()
    check_disk_space()
    
    # Migration (info seulement)
    print("\n" + "="*60)
    print("📦 VÉRIFICATION DES MODÈLES EXISTANTS")
    print("="*60)
    migrate_models_from_cache()
