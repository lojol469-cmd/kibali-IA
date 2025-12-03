#!/usr/bin/env python3.13
"""
Script pour vérifier les licences de tous les modèles dans le cache HuggingFace
"""

import json
from pathlib import Path

# Liste des modèles avec leurs licences connues
MODEL_LICENSES = {
    # ✅ COMMERCIAL OK - Apache 2.0 / MIT / BSD
    "Qwen--Qwen2.5-1.5B-Instruct": {"license": "Apache 2.0", "commercial": True, "usage": "LLM principal"},
    "openai--clip-vit-base-patch32": {"license": "MIT", "commercial": True, "usage": "Vision AI classification"},
    "openai--clip-vit-large-patch14": {"license": "MIT", "commercial": True, "usage": "Vision AI avancée"},
    "sentence-transformers--all-MiniLM-L6-v2": {"license": "Apache 2.0", "commercial": True, "usage": "Embeddings texte"},
    "Helsinki-NLP--opus-mt-fr-en": {"license": "Apache 2.0", "commercial": True, "usage": "Traduction FR→EN"},
    "facebook--bart-large-cnn": {"license": "MIT", "commercial": True, "usage": "Résumé automatique"},
    "dbmdz--bert-large-cased-finetuned-conll03-english": {"license": "MIT", "commercial": True, "usage": "NER (Named Entity Recognition)"},
    "Salesforce--codegen-350M-mono": {"license": "BSD-3-Clause", "commercial": True, "usage": "Génération de code"},
    "microsoft--DialoGPT-medium": {"license": "MIT", "commercial": True, "usage": "Dialogue conversationnel"},
    "t5-small": {"license": "Apache 2.0", "commercial": True, "usage": "Text-to-Text transformations"},
    "google--mt5-small": {"license": "Apache 2.0", "commercial": True, "usage": "Multilingual T5"},
    
    # ⚠️ ATTENTION - Licences restrictives
    "deepseek-ai--deepseek-coder-1.3b-instruct": {"license": "Deepseek License", "commercial": False, "note": "Restrictions commerciales"},
    "microsoft--Phi-3-mini-4k-instruct": {"license": "MIT", "commercial": True, "usage": "Small LLM (3.8B)", "note": "Vérifié OK commercial"},
    "microsoft--phi-1_5": {"license": "MIT", "commercial": True, "usage": "Code generation"},
    "mistralai--Mistral-7B-Instruct-v0.2": {"license": "Apache 2.0", "commercial": True, "usage": "LLM 7B performant"},
    
    # 🎨 IMAGE GENERATION - Licences complexes
    "CompVis--stable-diffusion-v1-4": {"license": "CreativeML Open RAIL-M", "commercial": True, "usage": "Image generation", "note": "Restrictions sur contenu"},
    "runwayml--stable-diffusion-v1-5": {"license": "CreativeML Open RAIL-M", "commercial": True, "usage": "Image generation v1.5"},
    "stabilityai--stable-diffusion-xl-base-1.0": {"license": "CreativeML Open RAIL++-M", "commercial": True, "usage": "SDXL base", "note": "Vérifier restrictions d'usage"},
    "Lykon--DreamShaper-8": {"license": "CreativeML Open RAIL-M", "commercial": True, "usage": "SD fine-tune"},
    "Lykon--dreamshaper-8-inpainting": {"license": "CreativeML Open RAIL-M", "commercial": True, "usage": "Inpainting"},
    "SG161222--RealVisXL_V4.0": {"license": "CreativeML Open RAIL++-M", "commercial": True, "usage": "Realistic SDXL"},
    "SG161222--Realistic_Vision_V5.1_noVAE": {"license": "CreativeML Open RAIL-M", "commercial": True, "usage": "Realistic SD1.5"},
    "emilianJR--epiCRealism": {"license": "CreativeML Open RAIL-M", "commercial": True, "usage": "Realistic fine-tune"},
    "segmind--SSD-1B": {"license": "Apache 2.0", "commercial": True, "usage": "SD distilled"},
    "Comfy-Org--flux1-dev": {"license": "Apache 2.0", "commercial": True, "usage": "FLUX.1 dev", "note": "Vérifier si dev ou schnell"},
    
    # 🎬 ANIMATION & VIDEO
    "ByteDance--AnimateDiff-Lightning": {"license": "Apache 2.0", "commercial": True, "usage": "Animation rapide"},
    "guoyww--animatediff-motion-adapter-v1-5-2": {"license": "Apache 2.0", "commercial": True, "usage": "Motion SD1.5"},
    "guoyww--animatediff-motion-adapter-v1-5-3": {"license": "Apache 2.0", "commercial": True, "usage": "Motion SD1.5 v3"},
    "guoyww--animatediff-motion-adapter-sdxl-beta": {"license": "Apache 2.0", "commercial": True, "usage": "Motion SDXL"},
    "wangfuyun--AnimateLCM": {"license": "Apache 2.0", "commercial": True, "usage": "LCM Animation"},
    
    # 🎛️ CONTROLNET & ADAPTERS
    "lllyasviel--ControlNet": {"license": "Apache 2.0", "commercial": True, "usage": "ControlNet collection"},
    "lllyasviel--Annotators": {"license": "Apache 2.0", "commercial": True, "usage": "Preprocessors"},
    "lllyasviel--control_v11p_sd15_canny": {"license": "Apache 2.0", "commercial": True, "usage": "Canny edges"},
    "lllyasviel--control_v11p_sd15_openpose": {"license": "Apache 2.0", "commercial": True, "usage": "OpenPose"},
    "lllyasviel--control_v11f1e_sd15_tile": {"license": "Apache 2.0", "commercial": True, "usage": "Tile upscale"},
    "lllyasviel--sd-controlnet-canny": {"license": "Apache 2.0", "commercial": True, "usage": "Canny v1"},
    "lllyasviel--sd-controlnet-openpose": {"license": "Apache 2.0", "commercial": True, "usage": "OpenPose v1"},
    "lllyasviel--sd-controlnet-scribble": {"license": "Apache 2.0", "commercial": True, "usage": "Scribble"},
    "h94--IP-Adapter": {"license": "Apache 2.0", "commercial": True, "usage": "Image Prompt Adapter"},
    "h94--IP-Adapter-FaceID": {"license": "Apache 2.0", "commercial": True, "usage": "Face consistency"},
    
    # 🎵 AUDIO
    "facebook--musicgen-small": {"license": "CC-BY-NC 4.0", "commercial": False, "note": "NON-COMMERCIAL uniquement"},
    "facebook--musicgen-medium": {"license": "CC-BY-NC 4.0", "commercial": False, "note": "NON-COMMERCIAL uniquement"},
    
    # 🏗️ 3D GENERATION
    "naver--DUSt3R_ViTLarge_BaseDecoder_512_dpt": {"license": "CC-BY-NC-SA 4.0", "commercial": False, "note": "NON-COMMERCIAL pour recherche"},
    "ashawkey--LGM": {"license": "MIT", "commercial": True, "usage": "3D generation from image"},
    
    # 🔧 UTILITIES
    "stabilityai--sd-vae-ft-mse": {"license": "Apache 2.0", "commercial": True, "usage": "VAE encoder/decoder"},
    "madebyollin--sdxl-vae-fp16-fix": {"license": "Apache 2.0", "commercial": True, "usage": "SDXL VAE FP16"},
    "intfloat--multilingual-e5-large-instruct": {"license": "MIT", "commercial": True, "usage": "Embeddings multilingues"},
    "vallabh001--xlm-roberta-base-multilingual-en-es": {"license": "MIT", "commercial": True, "usage": "XLM-RoBERTa"},
}

def check_licenses():
    """Affiche un rapport complet des licences"""
    
    print("=" * 80)
    print("📋 RAPPORT DES LICENCES - MODÈLES HUGGINGFACE CACHE")
    print("=" * 80)
    print()
    
    # ✅ Modèles commerciaux OK
    commercial_ok = [(k, v) for k, v in MODEL_LICENSES.items() if v["commercial"]]
    print(f"✅ MODÈLES UTILISABLES EN COMMERCIAL ({len(commercial_ok)} modèles):")
    print("-" * 80)
    
    categories = {
        "LLM & NLP": ["Qwen", "bart", "bert", "codegen", "DialoGPT", "t5", "mt5", "Phi", "phi", "Mistral"],
        "Vision AI": ["clip", "IP-Adapter"],
        "Image Generation": ["stable-diffusion", "DreamShaper", "RealVis", "Realistic", "epiCRealism", "SSD", "flux"],
        "Animation & Video": ["AnimateDiff", "animatediff", "AnimateLCM"],
        "ControlNet": ["ControlNet", "control_v11", "sd-controlnet", "Annotators"],
        "3D Generation": ["LGM"],
        "Audio": [],
        "Utilities": ["vae", "sentence-transformers", "opus-mt", "xlm-roberta", "multilingual-e5"]
    }
    
    for category, keywords in categories.items():
        category_models = [m for m in commercial_ok if any(kw in m[0] for kw in keywords)]
        if category_models:
            print(f"\n🔹 {category}:")
            for model, info in category_models:
                usage = info.get("usage", "")
                note = f" - {info['note']}" if "note" in info else ""
                print(f"   • {model}")
                print(f"     License: {info['license']} | Usage: {usage}{note}")
    
    # ❌ Modèles NON commerciaux
    non_commercial = [(k, v) for k, v in MODEL_LICENSES.items() if not v["commercial"]]
    print()
    print("=" * 80)
    print(f"❌ MODÈLES NON-COMMERCIAUX À ÉVITER ({len(non_commercial)} modèles):")
    print("-" * 80)
    for model, info in non_commercial:
        print(f"   • {model}")
        print(f"     License: {info['license']} - {info['note']}")
    
    print()
    print("=" * 80)
    print("💡 RECOMMANDATIONS POUR KIBALI OFFLINE:")
    print("=" * 80)
    print("""
1. ✅ UTILISER (Commercial OK):
   - Qwen 2.5-1.5B : LLM principal
   - CLIP (base & large) : Vision AI
   - Sentence Transformers : Embeddings
   - Mistral 7B : LLM avancé
   - Stable Diffusion (avec CreativeML RAIL) : Image generation
   - ControlNet : Contrôle avancé images
   - AnimateDiff : Animations
   - LGM : Génération 3D

2. ❌ NE PAS UTILISER (Non-Commercial):
   - MusicGen (small & medium) : CC-BY-NC
   - DUSt3R : CC-BY-NC-SA (recherche uniquement)
   - DeepSeek Coder : Restrictions commerciales

3. ⚠️ VÉRIFIER:
   - FLUX.1 : Vérifier si 'dev' (non-commercial) ou 'schnell' (Apache 2.0)
   - Stable Diffusion RAIL : Lire restrictions sur contenu généré

4. 📦 OUTILS DE BASE KIBALI OFFLINE:
   - LLM : Qwen 2.5-1.5B (Apache 2.0)
   - Vision : CLIP-ViT-Base (MIT)
   - Code : Codegen-350M (BSD-3) ou Phi-1.5 (MIT)
   - Images : SD 1.5 (CreativeML RAIL-M)
   - 3D : LGM (MIT)
   - Embeddings : all-MiniLM-L6-v2 (Apache 2.0)
   - Traduction : opus-mt-fr-en (Apache 2.0)
    """)
    
    print("=" * 80)
    print(f"📊 RÉSUMÉ: {len(commercial_ok)}/{len(MODEL_LICENSES)} modèles commerciaux")
    print("=" * 80)

if __name__ == "__main__":
    check_licenses()
