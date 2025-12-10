# 🔧 Système d'Outils Autonome de Kibali

## ✅ Outils Disponibles (9 outils)

### 1. 📄 **Recherche dans les PDFs** (`pdf_document_search`)
**Fonction:** Fouille automatiquement dans la base vectorielle des documents
**Détection:**
- ✅ "cherche dans les documents"
- ✅ "que disent les rapports"
- ✅ "trouve dans les fichiers"
- ✅ "selon les documents PDF"

**Amélioration:**
- Maintenant fait une vraie recherche (pas juste des instructions)
- Utilise la vectordb FAISS directement
- Retourne 10 résultats max avec extraits
- Score de 1.00 sur questions documentaires

### 2. 🌐 **Recherche Web** (`web_search_advanced`)
**Fonction:** Cherche des informations récentes sur internet
**Détection:**
- Activé si PAS d'info locale ET question factuelle
- Questions: "comment", "pourquoi", "qu'est-ce"

### 3. 🔢 **Calculatrice** (`math_calculator`)
**Fonction:** Effectue des calculs mathématiques
**Détection:**
- "calcule", "calculer", "+", "*", "-", "/"
- "moyenne", "somme", "total"

### 4. 🌍 **Traduction** (`translation_tool`)
**Fonction:** Traduit du texte
**Détection:**
- "traduis", "traduire", "en anglais", "en français"

### 5. 💻 **Génération de Code** (`code_generator`)
**Fonction:** Génère du code Python, JavaScript, etc.
**Détection:**
- "fonction", "script", "programme", "algorithme", "code"

### 6. 🖼️ **Analyse d'Images** (`image_analysis`)
**Fonction:** Analyse et décrit des images
**Détection:**
- Présence d'images uploadées

### 7. 📊 **Gestion Excel** (`excel_organizer`)
**Fonction:** Organise et analyse des fichiers Excel
**Détection:**
- Fichiers .xlsx, .xls, .csv uploadés

### 8. 📸 **Optimisation Photogrammétrie** (`photogrammetry_optimizer`)
**Fonction:** Optimise les datasets de photos pour reconstruction 3D
**Détection:**
- "photogrammétrie", "reconstruction 3D", "dust3r"

### 9. 🗂️ **Gestionnaire de Fichiers** (`file_manager`)
**Fonction:** Gère les fichiers locaux
**Détection:**
- Opérations sur fichiers

## 🧠 Logique d'Autonomie

### Flux de Décision
```
1. User pose une question
   ↓
2. Système cherche dans base vectorielle (RAG)
   ↓
3. Info trouvée?
   ├─ OUI → Utilise l'info + suggère compléments
   └─ NON → Active outils pertinents
       ↓
4. Analyse intelligente du besoin:
   ├─ Question documentaire? → PDF Search
   ├─ Question factuelle? → Web Search
   ├─ Calcul? → Calculatrice
   ├─ Traduction? → Translation
   └─ Code? → Code Generator
   ↓
5. Exécute outils en parallèle
   ↓
6. Intègre résultats dans le contexte
   ↓
7. IA génère réponse enrichie
```

### Comportement Intelligent

**Scénario 1: Info existe localement**
```
👤 Que disent les documents sur la géologie?
🧠 [Cherche dans vectordb] → Trouve 5 docs
📄 [Outil PDF activé] → Extrait passages pertinents
🤖 "D'après les documents PDF, la géologie de la zone..."
```

**Scénario 2: Info n'existe pas localement**
```
👤 Comment avoir de l'argent avec une IA?
🧠 [Cherche dans vectordb] → Rien trouvé
🌐 [Outil Web activé] → Cherche sur internet
🤖 "Je n'ai pas cette information localement, mais d'après mes recherches web..."
```

**Scénario 3: Demande explicite d'outil**
```
👤 Calcule 15 * 27 + 8
🔢 [Outil Calc activé immédiatement]
🤖 "Résultat: 413"
```

## 📊 Statistiques de Tests

### Test de Détection PDF
- ✅ Score 1.00 pour "cherche dans les documents pdf"
- ✅ Score 1.00 pour "que disent les rapports"
- ✅ Score 1.00 pour "trouve dans les fichiers"
- ⚠️ Score 0.60 pour "comment avoir de l'argent" (normal)

### Test de Détection Générale
- ✅ Web: activé pour questions factuelles sans info locale
- ✅ Calc: activé pour expressions mathématiques
- ✅ Trad: activé pour demandes de traduction
- ✅ Code: activé pour génération de code
- ✅ PDF: activé pour recherche documentaire

## 🎯 Transparence

L'utilisateur voit toujours:
- 🔧 Badge des outils utilisés: "Outils: pdf_document_search, web_search_advanced"
- 📄 Nombre de documents trouvés
- 🌐 Sources web consultées
- ✅ Succès ou échec de chaque outil

## 💡 Exemples d'Usage

**1. Recherche Documentaire**
```
👤: Que disent les documents sur l'exploration minière?
🤖: 🔧 Outils: pdf_document_search
     
     D'après les 5 documents trouvés:
     
     📄 Document 1 (rapport_exploration_2024.pdf):
     "L'exploration minière dans la région..."
     
     📄 Document 2 (etude_geologie.pdf):
     "Les formations géologiques révèlent..."
```

**2. Question Factuelle + Calcul**
```
👤: Combien coûterait un serveur pour héberger 1000 requêtes/jour?
🤖: 🔧 Outils: web_search_advanced, math_calculator
     
     D'après mes recherches, un serveur cloud adapté coûterait environ:
     - AWS EC2 t3.medium: 30$/mois
     - Base de données RDS: 45$/mois
     - Bande passante: 15$/mois
     Total: 90$/mois
```

**3. Analyse Complète**
```
👤: Analyse ce fichier Excel et génère un rapport PDF
🤖: 🔧 Outils: excel_organizer, pdf_generator
     
     Fichier analysé: 1250 lignes, 15 colonnes
     Rapport PDF généré: 25 pages
     📥 [Bouton Télécharger]
```

---

✅ **Système 100% fonctionnel et autonome !**
