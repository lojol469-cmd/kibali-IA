# Guide de Test des Outils Kibali-IA

## 🎯 Outils Disponibles et Comment les Tester

### 1. 📄 **PDF Search Tool** (pdf_document_search)
**Capacités:** Recherche dans les documents PDF, analyse de géophysique, géologie, minéralogie

**Requêtes de test:**
- "Fais moi une recherche sur 10 pdfs concernant la géophysique"
- "Trouve les documents sur la minéralogie"
- "Que disent les rapports sur l'exploration minière ?"
- "Recherche dans les documents techniques"

---

### 2. 🌐 **Web Search Tool** (web_search_advanced)
**Capacités:** Recherche web, actualités, documentation technique

**Requêtes de test:**
- "Quelles sont les dernières actualités sur l'IA ?"
- "Comment faire du web scraping en Python ?"
- "Recherche des informations sur la géophysique moderne"
- "Prix des GPU pour le deep learning"

---

### 3. 📸 **Image Analysis Tool** (image_analysis)
**Capacités:** Analyse d'images, détection d'objets, analyse de couleurs

**Requêtes de test:**
- "Analyse cette image de sol minier"
- "Décris cette photo géologique"
- "Quels objets vois-tu dans l'image ?"
- "Analyse les couleurs de cette photo"

---

### 4. 💻 **Code Generation Tool** (code_generator)
**Capacités:** Génération de code Python, JavaScript, Java, etc.

**Requêtes de test:**
- "Crée une fonction Python pour calculer la factorielle"
- "Génère une classe JavaScript pour gérer les utilisateurs"
- "Écris un algorithme de tri en Python"
- "Crée des tests unitaires pour ma fonction calcul"
- "Debug ce code: for i in range 10 print(i)"

---

### 5. 🔢 **Math Calculator Tool** (math_calculator)
**Capacités:** Calculs mathématiques, résolution d'équations, dérivées, intégrales

**Requêtes de test:**
- "Calcule 25 * 48 + 123"
- "Résous l'équation x + 5 = 10"
- "Quelle est la dérivée de x^2 + 3x"
- "Calcule l'aire d'un cercle de rayon 5"
- "Résous 5! (factorielle)"

---

### 6. 🌍 **Translation Tool** (translation_tool)
**Capacités:** Traduction multilingue avec détection automatique

**Requêtes de test:**
- "Traduis 'Bonjour comment allez-vous' en anglais"
- "Translate 'Hello how are you' en français"
- "Traduis ce texte en espagnol: Je vais bien merci"

---

### 7. 📁 **File Management Tool** (file_manager)
**Capacités:** Gestion de fichiers et dossiers

**Requêtes de test:**
- "Crée un fichier test.txt"
- "Liste le contenu du dossier /home/belikan"
- "Analyse le fichier app.py"
- "Cherche tous les fichiers .py dans le dossier actuel"
- "Crée un dossier nommé 'test_folder'"

---

## 🎨 Affichage des Outils dans le Chat

Lorsqu'un outil est détecté, vous verrez :

1. **Popup violet avec gradient** affichant les outils sélectionnés
2. **Expanders** pour chaque outil montrant:
   - L'exécution en temps réel
   - Les résultats JSON détaillés
   - Le statut (succès/erreur)

---

## ✅ Tests de Validation

### Test Multi-Outils
"Recherche 5 pdfs sur la géophysique et traduis le résumé en anglais"
→ Devrait activer: PDF Search + Translation

### Test Contexte Précis
"Calcule la dérivée de x^3 et génère le code Python correspondant"
→ Devrait activer: Math Calculator + Code Generator

### Test RAG + Web
"Que disent mes documents sur la géophysique et quelles sont les dernières actualités ?"
→ Devrait activer: PDF Search + Web Search

---

## 🐛 Corrections Appliquées

1. ✅ Ajout des imports `time` et `os` manquants dans tous les outils
2. ✅ Amélioration de la détection avec contexte (vectordb, pdfs disponibles)
3. ✅ Affichage popup élégant avec gradient violet/doré
4. ✅ Expanders pour voir les résultats détaillés de chaque outil
5. ✅ Intégration complète avec la recherche RAG pour l'outil PDF
6. ✅ Gestion des erreurs avec affichage clair

---

## 🚀 Comment Utiliser

1. **Onglet Chat RAG + Web**: Posez vos questions normalement
2. **Onglet Outils Dynamiques > Test manuel**: Testez un outil spécifique
3. L'IA détecte automatiquement les outils pertinents selon votre question
4. Les résultats s'affichent dans des expanders cliquables
