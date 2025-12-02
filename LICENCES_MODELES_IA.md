# 📜 LICENCES DES MODÈLES IA - USAGE COMMERCIAL

## ✅ Modèles 100% Commerciaux Autorisés

Date de vérification: 2 Décembre 2025

---

## 🎯 Modèles Textuels

### 1. **Qwen 2.5 Series** ✅
- **Licence:** Apache 2.0
- **Usage commercial:** ✅ Autorisé sans restriction
- **Provider:** Alibaba Cloud
- **Modèles:**
  - `Qwen/Qwen2.5-7B-Instruct`
  - `Qwen/Qwen2.5-14B-Instruct`
  - `Qwen/Qwen2.5-32B-Instruct`
  - `Qwen/Qwen2.5-72B-Instruct`
- **Notes:** Excellent pour le français, chinois, anglais. Pas de restrictions commerciales.

---

### 2. **Llama 3 / 3.1 / 3.2** ✅
- **Licence:** Llama 3 Community License (équivalent Apache 2.0 pour usage commercial)
- **Usage commercial:** ✅ Autorisé si < 700M utilisateurs actifs mensuels
- **Provider:** Meta AI
- **Modèles:**
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (multimodal)
  - `meta-llama/Llama-3.2-3B-Instruct`
- **Notes:** Très performant, licence permissive. Multimodal avec Llama 3.2 Vision.

---

### 3. **Mistral AI Series** ✅
- **Licence:** Apache 2.0
- **Usage commercial:** ✅ Totalement libre
- **Provider:** Mistral AI (France)
- **Modèles:**
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `mistralai/Mistral-Nemo-Instruct-2407` (12B)
  - `mistralai/Mixtral-8x7B-Instruct-v0.1`
- **Notes:** Excellent pour le français, licence très permissive.

---

### 4. **Microsoft Phi-3** ✅
- **Licence:** MIT License
- **Usage commercial:** ✅ Totalement libre
- **Provider:** Microsoft
- **Modèles:**
  - `microsoft/Phi-3-mini-4k-instruct` (3.8B)
  - `microsoft/Phi-3-medium-4k-instruct` (14B)
  - `microsoft/Phi-3-small-8k-instruct` (7B)
- **Notes:** Très efficace pour la taille, optimisé pour mobile/edge.

---

### 5. **Google Gemma 2** ✅
- **Licence:** Gemma Terms of Use (usage commercial autorisé)
- **Usage commercial:** ✅ Autorisé
- **Provider:** Google
- **Modèles:**
  - `google/gemma-2-2b-it`
  - `google/gemma-2-9b-it`
  - `google/gemma-2-27b-it`
- **Notes:** Open source de Google, performant et rapide.

---

## 🖼️ Modèles Vision (Multimodal)

### 1. **Llama 3.2 Vision** ✅
- **Licence:** Llama 3 Community License
- **Usage commercial:** ✅ Autorisé (< 700M MAU)
- **Modèle:** `meta-llama/Llama-3.2-11B-Vision-Instruct`
- **Capacités:** Vision + texte, analyse d'images avancée
- **Notes:** Le meilleur modèle vision open source actuellement.

---

### 2. **Qwen2-VL** ✅
- **Licence:** Apache 2.0
- **Usage commercial:** ✅ Totalement libre
- **Modèles:**
  - `Qwen/Qwen2-VL-7B-Instruct`
  - `Qwen/Qwen2-VL-72B-Instruct`
- **Capacités:** Vision + texte multilingue
- **Notes:** Alternative puissante à Llama Vision, multilingue.

---

### 3. **BLIP / BLIP-2** ✅
- **Licence:** BSD-3-Clause
- **Usage commercial:** ✅ Autorisé
- **Provider:** Salesforce
- **Modèles:**
  - `Salesforce/blip-image-captioning-large`
  - `Salesforce/blip2-opt-2.7b`
  - `Salesforce/blip2-flan-t5-xl`
- **Capacités:** Image captioning, VQA
- **Notes:** Très bon pour les captions simples, rapide.

---

### 4. **LLaVA (Open Source)** ✅
- **Licence:** Apache 2.0
- **Usage commercial:** ✅ Autorisé
- **Modèles:**
  - `llava-hf/llava-1.5-7b-hf`
  - `llava-hf/llava-1.5-13b-hf`
  - `llava-hf/llava-v1.6-mistral-7b-hf`
- **Capacités:** Vision + conversation
- **Notes:** Excellent pour le dialogue multimodal.

---

## ❌ Modèles NON Commerciaux (À ÉVITER)

### 1. **DeepSeek V3** ❌
- **Licence:** DeepSeek License (restrictions commerciales)
- **Usage commercial:** ❌ Nécessite autorisation spéciale
- **Raison:** License propriétaire avec clause non-commerciale
- **Alternative:** Qwen2.5 32B, Llama 3.1 70B

---

### 2. **Yi Series (01.AI)** ❌
- **Licence:** Yi License Agreement
- **Usage commercial:** ⚠️ Restrictions selon usage
- **Alternative:** Qwen2.5, Llama 3.1

---

### 3. **Falcon 180B** ⚠️
- **Licence:** Apache 2.0 MAIS restrictions sur données
- **Usage commercial:** ⚠️ Limité
- **Alternative:** Llama 3.1, Mixtral

---

## 📊 Recommandations par Cas d'Usage

### Pour une Application Commerciale Générale
```python
RECOMMANDÉ:
1. Qwen2.5-14B-Instruct (Apache 2.0)
2. Mistral-Nemo-12B (Apache 2.0)
3. Llama-3.1-8B-Instruct (Llama License)
```

### Pour Analyse d'Images (Géologie, Mining)
```python
RECOMMANDÉ:
1. Llama-3.2-11B-Vision (Vision + texte)
2. Qwen2-VL-7B (Multilingue)
3. LLaVA-1.5-13B (Dialogue)

FALLBACK:
- BLIP-2 (Caption rapide)
```

### Pour le Français
```python
RECOMMANDÉ:
1. Mistral-Nemo-12B (Natif français)
2. Qwen2.5-14B (Excellent multilingue)
3. Llama-3.1-8B (Bon français)
```

### Pour Mobile/Edge Computing
```python
RECOMMANDÉ:
1. Phi-3-small-7B (MIT)
2. Qwen2.5-7B (Apache 2.0)
3. Gemma-2-2b (Très léger)
```

---

## 🔍 Vérification des Licences

### Comment Vérifier
1. Aller sur HuggingFace: `https://huggingface.co/[model]`
2. Vérifier l'onglet "Files and versions"
3. Chercher `LICENSE` ou `README.md`
4. Lire les conditions d'usage commercial

### Points Clés à Vérifier
- ✅ "Apache 2.0" = Commercial OK
- ✅ "MIT" = Commercial OK
- ✅ "BSD" = Commercial OK
- ⚠️ "Non-commercial" = À ÉVITER
- ⚠️ "Research only" = À ÉVITER
- ⚠️ "CC BY-NC" = NON Commercial

---

## 📝 Configuration Recommandée pour Kibali IA

```python
WORKING_MODELS = {
    # Usage général - Apache 2.0
    "Qwen2.5 14B (Commercial)": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5 7B (Rapide)": "Qwen/Qwen2.5-7B-Instruct",
    
    # Vision multimodale - Llama License (commercial OK)
    "Llama 3.2 11B Vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    
    # Français - Apache 2.0
    "Mistral Nemo 12B": "mistralai/Mistral-Nemo-Instruct-2407",
    
    # Léger - MIT
    "Phi-3 Medium 14B": "microsoft/Phi-3-medium-4k-instruct",
}

# Vision alternative
VISION_MODELS = {
    "primary": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "fallback": "Salesforce/blip-image-captioning-large",
    "alternative": "Qwen/Qwen2-VL-7B-Instruct"
}
```

---

## ⚖️ Résumé Juridique

### ✅ 100% Sûr pour Usage Commercial
1. **Qwen 2.5** (Apache 2.0)
2. **Mistral AI** (Apache 2.0)
3. **Microsoft Phi-3** (MIT)
4. **BLIP** (BSD-3)
5. **Llama 3.x** (< 700M MAU)
6. **Gemma 2** (Gemma License)

### ⚠️ À Vérifier Selon Usage
1. **Yi Series** (Restrictions possibles)
2. **Falcon** (Limitations sur données)

### ❌ À Éviter pour Commercial
1. **DeepSeek V3** (License propriétaire)
2. **Modèles "Research Only"**
3. **CC BY-NC models**

---

## 🔗 Ressources

### Liens Officiels
- Qwen License: https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE
- Llama License: https://llama.meta.com/llama3/license/
- Mistral License: https://github.com/mistralai/mistral-src/blob/main/LICENSE
- Phi-3 License: https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/blob/main/LICENSE

### Outils de Vérification
- HuggingFace Model Cards
- GitHub Repositories
- OpenRAIL Licenses Database

---

**Dernière mise à jour:** 2 Décembre 2025  
**Validité:** À vérifier périodiquement (licences peuvent changer)  
**Disclaimer:** Ce document est informatif. Consultez un juriste pour confirmation.
