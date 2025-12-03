# 🤖 Configuration des Modèles IA - Kibali

## 📁 Structure des Modèles

Tous les modèles IA utilisés par Kibali sont centralisés dans `kibali_data/models/`:

```
kibali_data/models/
├── MODEL_PATHS.py          # Configuration centralisée (ce fichier)
├── huggingface_cache/      # Symlink vers ~/.cache/huggingface/hub
├── qwen2.5-1.5b/          # LLM local (Qwen 1.5B)
├── clip/                   # Vision AI (CLIP)
├── sentence-transformers/  # Embeddings pour RAG
├── easyocr/               # Modèles OCR
├── summarizer/            # Résumé de texte (BART)
├── translator/            # Traduction FR-EN
├── ner/                   # Named Entity Recognition
└── captioner/             # Génération de légendes d'images
```

## 🎯 Modèles Configurés

### 1. LLM Local (Qwen 2.5-1.5B)
- **Nom:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Licence:** Apache 2.0 ✅ Commercial
- **Taille:** ~3 GB
- **Usage:** Chat local sans API, génération de texte
- **Cache:** `kibali_data/models/qwen2.5-1.5b/`

### 2. Vision AI (CLIP)
- **Nom:** `openai/clip-vit-base-patch32`
- **Licence:** MIT ✅ Commercial
- **Taille:** ~600 MB
- **Usage:** Analyse sémantique d'images, classification
- **Cache:** `kibali_data/models/clip/`

### 3. Embeddings RAG (Sentence Transformers)
- **Nom:** `sentence-transformers/all-MiniLM-L6-v2`
- **Licence:** Apache 2.0 ✅ Commercial
- **Taille:** ~90 MB
- **Usage:** Vectorisation de documents pour recherche RAG
- **Cache:** `kibali_data/models/sentence-transformers/`

### 4. OCR (EasyOCR)
- **Langues:** Français + Anglais
- **Licence:** Apache 2.0 ✅ Commercial
- **Taille:** ~500 MB
- **Usage:** Extraction de texte depuis images
- **Cache:** `kibali_data/models/easyocr/`

### 5. Summarizer (BART)
- **Nom:** `facebook/bart-large-cnn`
- **Licence:** Apache 2.0 ✅ Commercial
- **Taille:** ~1.6 GB
- **Usage:** Résumé automatique de textes longs
- **Cache:** `kibali_data/models/summarizer/`

### 6. Translator (Helsinki-NLP)
- **Nom:** `Helsinki-NLP/opus-mt-fr-en`
- **Licence:** Apache 2.0 ✅ Commercial
- **Taille:** ~300 MB
- **Usage:** Traduction français → anglais
- **Cache:** `kibali_data/models/translator/`

### 7. NER (BERT)
- **Nom:** `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Licence:** MIT ✅ Commercial
- **Taille:** ~1.3 GB
- **Usage:** Extraction d'entités nommées (personnes, lieux, organisations)
- **Cache:** `kibali_data/models/ner/`

### 8. Captioner (BLIP)
- **Nom:** `Salesforce/blip-image-captioning-base`
- **Licence:** BSD-3-Clause ✅ Commercial
- **Taille:** ~1 GB
- **Usage:** Génération de légendes d'images
- **Cache:** `kibali_data/models/captioner/`

## 💾 Espace Disque

- **Total estimé:** ~8.4 GB
- **Recommandé:** 15+ GB d'espace libre
- **Actuel:** 566 GB disponibles ✅

## 🔧 Utilisation

### Dans app.py

```python
from kibali_data.models.MODEL_PATHS import (
    QWEN_MODEL_NAME, QWEN_CACHE_DIR,
    CLIP_MODEL_NAME, CLIP_CACHE_DIR,
    SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMER_CACHE,
    ...
)

# Exemple: Charger CLIP
clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL_NAME,
    cache_dir=str(CLIP_CACHE_DIR)
)
```

### Test de la configuration

```bash
python3.13 kibali_data/models/MODEL_PATHS.py
```

Affiche:
- 📦 Liste de tous les modèles
- 📁 Chemins de cache
- 💾 Espace disque requis
- ✅ Vérification de santé

## 🔄 Migration depuis ~/.cache

Par défaut, HuggingFace stocke les modèles dans `~/.cache/huggingface/hub`.

### Option 1: Symlink (économie d'espace) ✅ Recommandé

```bash
cd kibali_data/models
ln -s ~/.cache/huggingface/hub huggingface_cache
```

**Avantage:** Pas de duplication, utilise les modèles existants

### Option 2: Copie (indépendant)

```bash
cp -r ~/.cache/huggingface/hub/* kibali_data/models/huggingface_cache/
```

**Avantage:** Kibali autonome, portable

## 📝 Maintenance

### Vérifier les modèles installés

```python
from MODEL_PATHS import get_model_info, print_model_summary

print_model_summary()
```

### Nettoyer le cache

```bash
# Supprimer les anciens modèles non utilisés
cd kibali_data/models
du -sh */  # Vérifier la taille
rm -rf <ancien_modele>/
```

### Mettre à jour un modèle

```python
from transformers import AutoModel

# Le modèle sera automatiquement téléchargé dans le bon dossier
model = AutoModel.from_pretrained(
    "nouveau/modele",
    cache_dir=str(MODELS_DIR / "nouveau_dossier")
)
```

## 🚀 Avantages de la Centralisation

✅ **Clarté:** Tous les chemins dans un seul fichier  
✅ **Portabilité:** Facile de déplacer Kibali  
✅ **Maintenance:** Mise à jour centralisée  
✅ **Debugging:** Chemins explicites et traçables  
✅ **Backup:** Un seul dossier à sauvegarder  
✅ **Licence:** Tous les modèles sont commercialement OK  

## 🔐 Sécurité

- Les tokens HuggingFace restent dans `.env` (gitignored)
- Aucun modèle n'est commité sur Git
- Cache local = pas de fuite de données

## 📊 Performances

| Modèle | Chargement | RAM | VRAM (GPU) |
|--------|-----------|-----|------------|
| Qwen 1.5B | ~10s | 3 GB | 2 GB |
| CLIP | ~3s | 600 MB | 400 MB |
| Sentence-T | ~1s | 90 MB | 70 MB |
| EasyOCR | ~5s | 500 MB | 300 MB |

**Total:** ~4.2 GB RAM / ~3 GB VRAM

## 🐛 Dépannage

### Erreur: "Module MODEL_PATHS not found"

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "kibali_data" / "models"))
```

### Erreur: "No space left on device"

```bash
# Vérifier l'espace
df -h

# Nettoyer le cache pip
pip cache purge

# Supprimer les anciens modèles
rm -rf kibali_data/models/ancien_modele/
```

### Modèle ne se charge pas

```python
# Forcer le re-téléchargement
import shutil
shutil.rmtree(QWEN_CACHE_DIR)  # Puis relancer l'app
```

## 📚 Références

- HuggingFace Hub: https://huggingface.co/models
- Transformers Doc: https://huggingface.co/docs/transformers
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Sentence Transformers: https://www.sbert.net

---

**Auteur:** Kibali IA Team  
**Date:** Décembre 2025  
**Version:** 1.0  
**Licence:** Apache 2.0
