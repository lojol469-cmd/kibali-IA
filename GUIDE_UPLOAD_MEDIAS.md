# 📎 Guide d'Upload et Analyse de Médias - Kibali IA

## 🎯 Nouveau Système d'Upload (Style ChatGPT)

Kibali IA dispose maintenant d'un système complet d'upload et d'analyse de médias, similaire à ChatGPT, permettant d'analyser automatiquement :
- 🖼️ **Images** (analyse visuelle, couleurs, objets)
- 🎵 **Audio** (transcription et analyse)
- 🎥 **Vidéos** (extraction de frames, métadonnées)

---

## 🚀 Comment Utiliser

### 1. **Onglet Chat RAG + Web**

Dans l'onglet principal du chat, vous trouverez maintenant une section :

```
┌─────────────────────────────────────────────┐
│ 📎 Ajouter des médias                        │
├─────────────────────────────────────────────┤
│ 🖼️ Images  |  🎵 Audio  |  🎥 Vidéo        │
└─────────────────────────────────────────────┘
```

### 2. **Upload d'Images** 🖼️

**Formats supportés:**
- JPG, JPEG, PNG, GIF, BMP, WEBP

**Analyse automatique:**
- ✅ Affichage de l'image
- ✅ Dimensions et format
- ✅ Couleurs dominantes (palette visuelle)
- ✅ Détection d'objets
- ✅ Description automatique
- ✅ Analyse de composition

**Exemple d'utilisation:**
```
1. Uploadez une image de sol minier
2. L'analyse s'affiche automatiquement
3. Posez une question: "Que vois-tu dans cette image ?"
4. Kibali répond en intégrant l'analyse visuelle
```

### 3. **Upload d'Audio** 🎵

**Formats supportés:**
- MP3, WAV, OGG, M4A, FLAC

**Analyse automatique:**
- ✅ Lecteur audio intégré
- ✅ Durée et fréquence d'échantillonnage
- ✅ Transcription (simulée - en production utilise Whisper)
- ✅ Analyse du signal audio

**Exemple d'utilisation:**
```
1. Uploadez un fichier audio
2. Écoutez-le directement dans l'interface
3. La transcription s'affiche
4. Posez des questions sur le contenu
```

### 4. **Upload de Vidéos** 🎥

**Formats supportés:**
- MP4, AVI, MOV, MKV, WEBM

**Analyse automatique:**
- ✅ Lecteur vidéo intégré
- ✅ Résolution et durée
- ✅ FPS et nombre de frames
- ✅ Extraction de frames clés
- ✅ Métadonnées complètes

**Exemple d'utilisation:**
```
1. Uploadez une vidéo d'exploration minière
2. Regardez-la directement dans Kibali
3. L'analyse technique s'affiche
4. Demandez une analyse du contenu
```

---

## 🎨 Interface Visuelle

### Affichage d'Analyse d'Image

```
╔════════════════════════════════════════════╗
║ 🖼️ Analyse des Images                      ║
╠════════════════════════════════════════════╣
║ 📷 photo_geologie.jpg                      ║
║                                            ║
║ [Image]  │  📏 Dimensions: 1920x1080       ║
║          │  🎨 Format: JPEG                ║
║          │  🎨 Couleurs dominantes:        ║
║          │  [■][■][■][■][■]                ║
║          │  🔍 Objets détectés: 5          ║
║          │  📝 Description: Image de...    ║
╚════════════════════════════════════════════╝
```

### Popup des Couleurs Dominantes

Les couleurs sont affichées sous forme de carrés colorés cliquables :
```
🎨 Couleurs dominantes:
[#8B4513] [#D2691E] [#A0522D] [#654321] [#3E2723]
```

---

## 💬 Intégration avec le Chat

### Contexte Automatique

Lorsque vous uploadez des médias, le contexte est automatiquement ajouté à vos questions :

**Avant:**
```
Vous: Analyse cette image
```

**Après (enrichi automatiquement):**
```
Vous: Analyse cette image

📎 Médias analysés:
🖼️ Image: photo_geologie.jpg
  - Dimensions: 1920x1080
  - Description: Image montrant un terrain rocheux avec...
```

### Réponses Enrichies

Kibali répond en intégrant l'analyse des médias :

```
🤖 Kibali: D'après l'analyse de l'image photo_geologie.jpg :

📊 Informations techniques:
- Résolution: 1920x1080 pixels
- 5 objets détectés dans la scène

🎨 Analyse visuelle:
Les couleurs dominantes sont des tons terreux (marron, ocre)
indiquant probablement un sol argileux...

🔍 Observations:
[Réponse détaillée basée sur l'analyse]
```

---

## 🔧 Outils Activés Automatiquement

Lorsque vous uploadez des médias, Kibali détecte et active automatiquement les outils appropriés :

### Upload d'Image → Outil `image_analysis`
```
╔════════════════════════════════════════╗
║ 🔧 Outils IA Détectés                  ║
║ • image_analysis                       ║
╚════════════════════════════════════════╝

▼ ⚙️ Exécution de image_analysis...
  {
    "dimensions": "1920x1080",
    "color_analysis": {...},
    "object_detection": {...}
  }
```

---

## 📊 Exemples d'Utilisation Combinée

### Exemple 1: Image + Question Technique
```
1. Upload: photo_sol_minier.jpg
2. Question: "Quel type de sol vois-tu et quelles sont ses propriétés ?"
3. Réponse: Kibali combine l'analyse visuelle + recherche dans les PDFs + web
```

### Exemple 2: Multi-Médias
```
1. Upload: 
   - 3 images de carottes de forage
   - 1 vidéo du site
2. Question: "Compare ces échantillons et analyse le site"
3. Réponse: Analyse comparative détaillée avec toutes les sources
```

### Exemple 3: Audio + PDFs
```
1. Upload: interview_geologue.mp3
2. Question: "Que dit l'expert et qu'en disent mes documents ?"
3. Réponse: Transcription + recherche RAG dans les PDFs
```

---

## ⚙️ Configuration Technique

### Dépendances Requises

Les bibliothèques suivantes sont utilisées :

```python
# Images
opencv-python-headless  # Analyse d'images
Pillow                  # Manipulation d'images
scikit-image           # Traitement avancé

# Audio
librosa                # Analyse audio
soundfile             # Lecture de fichiers audio

# Vidéo
opencv-python         # Extraction de frames
```

### Stockage Temporaire

Les médias uploadés sont temporairement stockés dans :
```
/tmp/temp_image_*.jpg
/tmp/temp_audio_*.mp3
/tmp/temp_video_*.mp4
```

Les fichiers sont automatiquement supprimés après analyse.

---

## 🎯 Cas d'Usage Géophysique

### 1. Analyse de Carottes de Forage
```
Upload: carotte_1.jpg, carotte_2.jpg, carotte_3.jpg
Question: "Compare ces carottes et identifie les couches géologiques"
```

### 2. Analyse Sismique
```
Upload: sismogramme.jpg
Question: "Interprète ce sismogramme et que disent mes documents sur cette zone ?"
```

### 3. Vidéo de Site
```
Upload: site_exploration.mp4
Question: "Analyse cette vidéo du site et identifie les zones d'intérêt"
```

### 4. Présentation Audio
```
Upload: presentation_projet.mp3
Question: "Fais un résumé de la présentation avec les points clés"
```

---

## 🚀 Fonctionnalités Avancées

### Multi-Upload
✅ Uploadez plusieurs fichiers du même type simultanément
✅ Analyse en parallèle de tous les médias
✅ Contexte global pour le chat

### Analyse en Temps Réel
✅ Affichage progressif des résultats
✅ Expanders cliquables pour détails
✅ Prévisualisation immédiate

### Intégration Complète
✅ Combinaison avec recherche RAG
✅ Combinaison avec recherche web
✅ Combinaison avec autres outils IA

---

## 📝 Conseils d'Utilisation

### Pour de Meilleurs Résultats

1. **Images:**
   - Utilisez des images de haute qualité (>1000px)
   - Évitez les images trop sombres ou floues
   - Formats PNG ou JPG recommandés

2. **Audio:**
   - Privilégiez les formats WAV ou FLAC pour la qualité
   - Évitez les enregistrements trop longs (>5 min)
   - Audio clair sans trop de bruit de fond

3. **Vidéos:**
   - Résolution minimum 720p recommandée
   - Formats MP4 ou MOV optimaux
   - Limitez la durée pour un traitement rapide

### Questions Efficaces

✅ **Bon:** "Analyse cette image et identifie le type de roche"
✅ **Bon:** "Compare ces 3 images et trouve les différences"
✅ **Bon:** "Que dit cette vidéo sur le processus d'extraction ?"

❌ **Moins bon:** "Regarde ça" (trop vague)
❌ **Moins bon:** Question sans rapport avec le média uploadé

---

## 🔮 Prochaines Améliorations

### En Développement
- 🎙️ Transcription audio avec Whisper API
- 🎬 Analyse frame-by-frame des vidéos
- 📊 Graphiques et visualisations interactives
- 🔍 OCR (reconnaissance de texte) dans les images
- 🎨 Génération d'images avec DALL-E
- 🌍 Géolocalisation automatique des photos

---

## ✅ Résumé

**Kibali IA offre maintenant un système complet d'analyse multimédia :**

✅ Upload fluide de médias (images, audio, vidéo)
✅ Analyse automatique et intelligente
✅ Intégration transparente avec le chat
✅ Combinaison avec RAG et recherche web
✅ Interface visuelle élégante et intuitive
✅ Support de multiples formats
✅ Traitement en temps réel

**Prêt à analyser vos médias ! 🚀**
