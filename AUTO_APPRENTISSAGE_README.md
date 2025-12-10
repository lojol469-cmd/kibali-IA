# 🤖 Système d'Auto-Apprentissage Kibali-IA

## 📋 Vue d'ensemble

Le système d'auto-apprentissage de Kibali-IA est composé de deux modules complémentaires organisés dans des dossiers dédiés:

### 1. 🧠 **auto_learning/** - Apprentissage Automatique
Apprend automatiquement des interactions pour améliorer les réponses futures.

### 2. 🗂️ **knowledge_manager/** - Gestion des Connaissances
Stocke et organise les patterns appris avec un système de poids adaptatifs.

---

## 📁 Structure des Modules

```
kibali-IA/
├── auto_learning/
│   ├── __init__.py              # Point d'entrée du module
│   └── auto_learning.py         # Classe AutoLearning
├── knowledge_manager/
│   ├── __init__.py              # Point d'entrée du module
│   └── knowledge_manager.py     # Classe KnowledgeManager
├── auto_learning_data/          # Données d'apprentissage (créé automatiquement)
│   ├── conversations.json
│   ├── patterns.json
│   ├── concepts.json
│   ├── file_knowledge.json
│   └── embeddings.pkl
└── knowledge_data/              # Base de connaissances (créé automatiquement)
    ├── patterns_db.json
    ├── weights.json
    ├── recommendations.json
    └── history.json
```

---

## 🧠 Module auto_learning

### Fonctionnalités

#### 1. **Apprentissage des Conversations**
```python
from auto_learning import get_auto_learning

al = get_auto_learning()

# Apprendre d'une conversation
result = al.learn_from_conversation(
    user_message="Comment calculer 15 + 27 ?",
    ai_response="Le résultat est 42",
    context={"tools_used": ["math_calculator"]}
)
```

**Ce qui est appris:**
- Questions fréquentes (avec embeddings)
- Réponses efficaces
- Usage des outils
- Tendances des sujets

#### 2. **Apprentissage des PDFs**
```python
# Apprendre du contenu d'un PDF
result = al.learn_from_pdf(
    pdf_content="Python est un langage...",
    metadata={"filename": "guide.pdf", "pages": 10}
)
```

**Ce qui est extrait:**
- Concepts clés (mots importants)
- Patterns techniques (API, JSON, etc.)
- Contextes autour des concepts

#### 3. **Apprentissage des Fichiers Binaires**
```python
# Apprendre des métadonnées
result = al.learn_from_binary({
    "filename": "image.jpg",
    "size": 1024000,
    "use": "photo"
})
```

**Ce qui est appris:**
- Types de fichiers (.jpg, .png, etc.)
- Tailles moyennes
- Usages typiques

#### 4. **Suggestions Intelligentes**
```python
# Obtenir des suggestions pour une requête
suggestions = al.get_suggestions("comment calculer", top_k=5)

# Résultat:
# [
#   {
#     "type": "similar_question",
#     "question": "Comment calculer 15 + 27 ?",
#     "frequency": 3,
#     "score": 0.85
#   },
#   {
#     "type": "tool",
#     "tool": "math_calculator",
#     "frequency": 5,
#     "score": 0.75
#   }
# ]
```

#### 5. **Statistiques**
```python
stats = al.get_stats()

# Retourne:
# {
#   "total_conversations": 100,
#   "total_concepts": 250,
#   "total_files_analyzed": 50,
#   "top_questions": [...],
#   "top_concepts": [...],
#   "top_tools": [...],
#   "unique_topics": 15,
#   "file_types_known": 8
# }
```

---

## 🗂️ Module knowledge_manager

### Fonctionnalités

#### 1. **Stockage de Patterns**
```python
from knowledge_manager import get_knowledge_manager

km = get_knowledge_manager()

# Stocker un pattern d'outil
result = km.store_pattern("tool_patterns", {
    "tool_name": "math_calculator",
    "query_type": "calculation",
    "example": "calcule 15 + 27"
})
```

**Types de patterns:**
- `query_patterns`: Patterns de requêtes
- `response_patterns`: Patterns de réponses
- `tool_patterns`: Patterns d'usage d'outils
- `document_patterns`: Patterns de documents

#### 2. **Récupération de Patterns**
```python
# Chercher des patterns pertinents
patterns = km.retrieve_patterns(
    query="calcule des nombres",
    pattern_type="tool_patterns",  # Optionnel
    top_k=10
)

# Résultat:
# [
#   {
#     "pattern_id": "abc123...",
#     "pattern_type": "tool_patterns",
#     "data": {...},
#     "score": 0.85,
#     "occurrences": 5,
#     "weight": 0.7
#   }
# ]
```

#### 3. **Système de Poids Adaptatifs**
```python
# Mettre à jour le poids d'un pattern
new_weight = km.update_weights(
    pattern_id="abc123",
    feedback=0.2  # -1.0 à 1.0
)

# Le poids est ajusté avec learning_rate
# Poids > 0.7 = Pattern très utile
# Poids < 0.3 = Pattern peu utile
```

#### 4. **Recommandations Contextuelles**
```python
# Obtenir des recommandations
recommendations = km.get_recommendations(
    context={"query": "comment faire un calcul"},
    top_k=5
)

# Résultat:
# [
#   {
#     "type": "tool",
#     "tool": "math_calculator",
#     "reason": "Utilisé 10 fois pour des requêtes similaires",
#     "confidence": 0.8
#   }
# ]
```

#### 5. **Maintenance**

##### Decay des Poids Anciens
```python
# Réduire le poids des patterns non utilisés
result = km.decay_old_weights(days_threshold=30)

# Résultat:
# {
#   "decayed_count": 15,
#   "remaining_weights": 85
# }
```

##### Export/Import
```python
# Exporter toutes les connaissances
export_path = km.export_knowledge()
# Fichier: knowledge_export_20251210_173000.json

# Importer des connaissances
result = km.import_knowledge(export_path)
```

---

## 🎯 Intégration dans Kibali-IA

### Initialisation
Dans `app.py`, les modules sont initialisés automatiquement:

```python
# Import automatique au démarrage
from auto_learning import get_auto_learning
from knowledge_manager import get_knowledge_manager

# Dans session_state
st.session_state.auto_learning = get_auto_learning()
st.session_state.knowledge_manager = get_knowledge_manager()
```

### Apprentissage Automatique

#### 1. Lors d'une Conversation
```python
# Dans le chat normal
learn_result = st.session_state.auto_learning.learn_from_conversation(
    user_message=prompt,
    ai_response=response,
    context={"tools_used": tools_used}
)

# Stocker le pattern dans knowledge_manager
km.store_pattern("query_patterns", {"query": prompt})
km.store_pattern("tool_patterns", {"tool_name": tool_used})
```

#### 2. Lors de l'Upload d'un PDF
```python
# Après extraction du texte
learn_result = al.learn_from_pdf(pdf_text, metadata)

# Stocker dans knowledge_manager
km.store_pattern("document_patterns", {
    "source": filename,
    "type": "pdf",
    "concepts_count": learn_result['concepts_extracted']
})
```

#### 3. Lors de l'Analyse d'un Fichier Binaire
```python
# Après analyse du fichier
learn_result = al.learn_from_binary(file_info)

# Stocker dans knowledge_manager
km.store_pattern("document_patterns", {
    "source": filename,
    "type": "binary",
    "file_type": file_ext
})
```

---

## 📊 Onglet Nano-IA

L'onglet **🤖 Nano-IA & Apprentissage** dans l'interface affiche:

### Vue d'ensemble
- 📊 **Statistiques AutoLearning**: Conversations, concepts, fichiers
- 🗂️ **Statistiques KnowledgeManager**: Patterns, poids, historique

### Détails
- 🏆 **Top Questions Fréquentes**: Les 5 questions les plus posées
- 💡 **Concepts Clés**: Les 10 concepts les plus trouvés
- 🔧 **Outils les Plus Utilisés**: Les 5 outils les plus sollicités

### Test
- 🔮 **Test de Suggestions**: Poser une question et obtenir des suggestions intelligentes

### Maintenance
- 📤 **Export**: Sauvegarder toutes les connaissances
- 🧹 **Decay**: Appliquer le decay aux patterns anciens
- 📊 **Stats Détaillées**: Voir toutes les statistiques en JSON

---

## 🔧 Configuration

### Paramètres de AutoLearning
```python
al = AutoLearning(
    storage_dir="/home/belikan/kibali-IA/auto_learning_data"
)

# Modifier les limites
al.conversations = al.conversations[-1000:]  # Max 1000 conversations
```

### Paramètres de KnowledgeManager
```python
km = KnowledgeManager(
    storage_dir="/home/belikan/kibali-IA/knowledge_data"
)

# Modifier les paramètres d'apprentissage
km.learning_rate = 0.1      # Vitesse d'apprentissage
km.decay_factor = 0.95      # Facteur de decay
km.min_confidence = 0.3     # Confiance minimale
```

---

## 📈 Workflow Typique

### Étape 1: Utilisation Normale
L'utilisateur utilise Kibali-IA normalement (chat, PDFs, fichiers).

### Étape 2: Apprentissage Automatique
- Chaque conversation est analysée
- Les PDFs sont indexés pour les concepts
- Les fichiers binaires sont catégorisés

### Étape 3: Stockage des Patterns
- Les patterns sont stockés avec un poids initial de 0.5
- Les patterns similaires voient leur occurrence augmenter

### Étape 4: Recommandations
- Lors des prochaines requêtes similaires
- Le système suggère proactivement les outils/sources pertinents
- Les patterns avec poids élevé sont prioritaires

### Étape 5: Maintenance
- Les patterns anciens subissent un decay
- Les patterns inutiles (poids < 0.3) sont supprimés
- L'export périodique permet les sauvegardes

---

## 🎓 Exemples d'Usage Avancés

### Exemple 1: Analyse de Tendances
```python
al = get_auto_learning()
stats = al.get_stats()

# Voir les sujets tendance
for topic, timestamps in stats['topic_trends'].items():
    print(f"{topic}: {len(timestamps)} mentions")
```

### Exemple 2: Optimisation des Outils
```python
km = get_knowledge_manager()

# Voir quels outils sont sous-utilisés
for tool, info in km.patterns_db['tool_patterns'].items():
    weight = km.weights.get(tool, 0.5)
    if weight < 0.4:
        print(f"Outil peu efficace: {info['tool_name']}")
```

### Exemple 3: Analyse de Concepts
```python
al = get_auto_learning()

# Trouver les concepts co-occurrents
for concept, info in al.concepts.items():
    if info['count'] > 5:
        print(f"{concept}: {info['count']} occurrences")
        print(f"Contextes: {info['contexts'][:2]}")
```

---

## ✅ Avantages du Système

### 1. **Modularité**
- Chaque module dans son propre dossier
- Facile à maintenir et à étendre
- Import simple: `from auto_learning import ...`

### 2. **Apprentissage Continu**
- Pas besoin d'intervention manuelle
- S'améliore automatiquement avec l'usage
- Adapte ses recommandations en temps réel

### 3. **Performances**
- Embeddings cachés pour éviter les recalculs
- Patterns limités en taille (1000 conversations max)
- Decay automatique des données anciennes

### 4. **Insights**
- Statistiques détaillées
- Visualisation des tendances
- Export pour analyse externe

---

## 🔮 Évolutions Futures

### 1. **Apprentissage Fédéré**
- Partage de connaissances entre instances
- Sans partager les données brutes

### 2. **Nano-IAs Spécialisées**
- Une IA par domaine (code, docs, images)
- Routage intelligent des requêtes

### 3. **Reinforcement Learning**
- Feedback utilisateur sur les suggestions
- Ajustement automatique des poids

### 4. **Clustering de Patterns**
- Regrouper les patterns similaires
- Réduire la redondance

---

## 📚 Ressources

### Fichiers Principaux
- `/home/belikan/kibali-IA/auto_learning/auto_learning.py`: 700 lignes
- `/home/belikan/kibali-IA/knowledge_manager/knowledge_manager.py`: 600 lignes

### Dépendances
- `sentence-transformers`: Pour les embeddings
- `numpy`: Pour les calculs vectoriels
- `json`: Pour la sérialisation
- `pickle`: Pour le cache d'embeddings

### Tests
```bash
# Test complet
python3.13 -c "
from auto_learning import get_auto_learning
from knowledge_manager import get_knowledge_manager
print('✅ Modules chargés avec succès')
"
```

---

## 🎉 Conclusion

Le système d'auto-apprentissage de Kibali-IA est maintenant **pleinement fonctionnel** et organisé de manière professionnelle. Il apprend automatiquement de chaque interaction pour offrir une expérience toujours plus personnalisée et efficace.

**Prêt à apprendre ! 🚀**
