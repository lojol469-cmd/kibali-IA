# 📝 Extraction de Texte des Images (OCR)

## 🎯 Vue d'ensemble

Kibali IA intègre maintenant un système OCR (Optical Character Recognition) avancé pour **extraire automatiquement tout le texte** présent dans les images uploadées.

## ✨ Fonctionnalités

### 🔍 Détection Automatique
- **Extraction multi-langues**: Français + Anglais
- **Détection de confiance**: Chaque texte a un score de fiabilité
- **Positionnement**: Coordonnées des zones de texte détectées
- **Organisation intelligente**: Texte structuré et ordonné

### 📋 Types de Texte Détectés

L'OCR peut extraire:
- ✅ **Légendes** et annotations
- ✅ **Descriptions** et explications
- ✅ **Titres** et en-têtes
- ✅ **Tableaux** et données chiffrées
- ✅ **Graphiques** avec labels
- ✅ **Documents** scannés
- ✅ **Panneaux** et signalétique
- ✅ **Texte manuscrit** (selon qualité)

## 🛠️ Technologies Utilisées

### EasyOCR (Principal)
- **Modèle**: Deep Learning CNN
- **Langues**: Français + Anglais simultané
- **GPU**: Support CUDA automatique
- **Licence**: Apache 2.0 (usage commercial OK ✅)

### Tesseract (Fallback)
- **Fallback** si EasyOCR indisponible
- **Multi-langue**: `fra+eng`
- **Détails**: Extraction avec positions et confiance

## 📊 Format de Sortie

### Structure Organisée

```
📝 TEXTE EXTRAIT DE L'IMAGE:

🔍 Méthode: EasyOCR
   Nombre d'éléments détectés: 12

   1. ✅ [95%] Titre Principal
   2. ✅ [92%] Description complète
   3. ⚠️  [68%] Légende figure 1
   4. ✅ [87%] Données: 25.4 kg/m³
   ...

📄 TEXTE COMPLET RECONSTITUÉ:
Titre Principal Description complète Légende figure 1 Données: 25.4 kg/m³ ...
```

### Indicateurs de Confiance

- ✅ **> 80%**: Haute confiance (vert)
- ⚠️ **50-80%**: Confiance moyenne (orange)
- ❓ **< 50%**: Faible confiance (rouge)

## 🚀 Utilisation

### 1. Upload d'Image
```
1. Cliquez sur 📎 (Upload images)
2. Sélectionnez votre image
3. L'analyse démarre automatiquement
```

### 2. Processus Automatique
```
📦 Chargement du modèle OCR...
✅ Modèle OCR chargé

📝 Extraction du texte de l'image...
✅ 15 éléments de texte détectés!

🔍 Analyse avec CLIP local...
✅ Analyse CLIP réussie!

🤖 Génération de l'analyse approfondie...
```

### 3. Résultat dans le Chat

L'analyse complète inclut:
- 📝 **Section OCR**: Texte extrait organisé
- 🖼️ **Analyse visuelle**: Description CLIP
- 🤖 **Analyse IA**: Interprétation globale

## 💡 Cas d'Usage

### 🗺️ Géologie & Géophysique
- Extraction de **légendes de cartes**
- Lecture de **données de forages**
- Transcription de **rapports de terrain**
- Analyse de **coupes géologiques annotées**

### 📊 Documents Techniques
- Extraction de **tableaux de données**
- Lecture de **graphiques avec labels**
- Transcription de **schémas techniques**
- Analyse de **plans annotés**

### 📷 Images Générales
- Transcription de **panneaux**
- Extraction de **sous-titres**
- Lecture de **documents scannés**
- Analyse de **présentations**

## ⚙️ Configuration

### Modèles Téléchargés Automatiquement

Au premier usage:
```bash
Downloading detection model...
Downloading recognition model...
✅ Modèles téléchargés dans:
   ~/.EasyOCR/model/
```

### GPU vs CPU
- **Avec GPU**: ~2-3 secondes par image
- **Sans GPU**: ~5-10 secondes par image

### Cache Automatique
- Modèles en cache après 1er chargement
- Pas de re-téléchargement nécessaire

## 🎨 Intégration avec l'Analyse IA

### Prompt Enrichi

L'OCR enrichit l'analyse LLM:

```python
📝 TEXTE EXTRAIT DE L'IMAGE (OCR):
[Texte organisé avec confiance]

🎯 Ta mission: Fournis une analyse COMPLÈTE:
1. Description générale
2. **Analyse du texte détecté** ← NOUVEAU
3. Corrélation texte/visuel ← NOUVEAU
...
```

### Analyse Contextuelle

Le LLM peut maintenant:
- ✅ Interpréter le texte dans son contexte visuel
- ✅ Relier légendes aux éléments de l'image
- ✅ Extraire des données chiffrées
- ✅ Identifier le type de document
- ✅ Synthétiser informations textuelles

## 📈 Performances

### Précision
- **Documents clairs**: 90-95%
- **Photos terrain**: 70-85%
- **Manuscrit**: 50-70% (variable)
- **Texte déformé**: 40-60%

### Vitesse
- **Première image**: ~8-10s (chargement modèle)
- **Images suivantes**: ~2-3s (GPU) / ~5-8s (CPU)

### Langues
- 🇫🇷 Français: Natif
- 🇬🇧 Anglais: Natif
- Autres: Disponibles (à activer)

## 🔧 Dépannage

### Erreur "EasyOCR non installé"
```bash
pip install easyocr
```

### Modèle ne charge pas
```bash
# Supprimer cache et retélécharger
rm -rf ~/.EasyOCR/model/
# Relancer l'application
```

### Texte non détecté

Vérifiez:
- ✅ Résolution image > 300px
- ✅ Contraste texte/fond suffisant
- ✅ Texte pas trop incliné (< 45°)
- ✅ Qualité image acceptable

### Mauvaise précision

Améliorations possibles:
- 📸 Augmenter résolution
- 🔆 Améliorer éclairage
- 📐 Redresser l'image
- 🎨 Augmenter contraste

## 📝 Exemples d'Utilisation

### Exemple 1: Carte Géologique
```
Upload: carte_geologique_gabon.jpg

📝 TEXTE EXTRAIT:
1. ✅ [92%] Formation de Franceville
2. ✅ [88%] Échelle: 1:50000
3. ✅ [85%] Coordonnées: 1°30'S 13°15'E
4. ⚠️  [72%] Légende: Grès permien

🤖 ANALYSE IA:
Cette carte géologique montre la Formation de Franceville...
Les coordonnées 1°30'S 13°15'E situent la zone...
```

### Exemple 2: Graphique avec Données
```
Upload: courbe_resistivite.png

📝 TEXTE EXTRAIT:
1. ✅ [95%] Résistivité (Ω·m)
2. ✅ [91%] Profondeur (m)
3. ✅ [89%] Valeur max: 250 Ω·m
4. ✅ [87%] Valeur min: 10 Ω·m

🤖 ANALYSE IA:
Le graphique présente une courbe de résistivité...
Les valeurs varient de 10 à 250 Ω·m...
```

## 🎯 Roadmap

### Version Actuelle (v1.0)
- ✅ EasyOCR FR+EN
- ✅ Support GPU
- ✅ Organisation automatique
- ✅ Intégration analyse IA

### Prochaines Versions
- 📅 Support multi-langues (AR, ES, DE)
- 📅 Détection de tableaux structurés
- 📅 Export texte extrait (TXT, JSON)
- 📅 Correction orthographique OCR

## 📚 Ressources

### Documentation
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [Modèles disponibles](https://www.jaided.ai/easyocr/modellists/)

### Support
- Issues: GitHub Issues
- Questions: Discussions

---

**🎉 Profitez de l'extraction de texte automatique dans vos analyses d'images!**
