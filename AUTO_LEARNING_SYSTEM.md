# 🤖 Système d'Auto-Apprentissage et Nano-IA

## Vue d'ensemble

Le système d'auto-apprentissage de Kibali permet à l'IA d'apprendre automatiquement de chaque interaction, fichier uploadé et conversation. Il crée des **nano-IA spécialisées** par domaine qui deviennent des experts dans leur domaine respectif.

## Architecture

```
kibali_data/
└── models/
    ├── knowledge_manager.py      # Gestionnaire de connaissances
    ├── auto_learning.py          # Système d'apprentissage automatique
    └── knowledge_base/           # Base de connaissances
        ├── domains_index.json    # Index des domaines
        ├── domain_vectordb/      # Bases vectorielles par domaine
        │   ├── geophysics_ert/
        │   ├── binary_analysis/
        │   ├── pdf_documents/
        │   └── ...
        └── nano_ai/              # Nano-IA spécialisées
            ├── geophysics_ert_nano_ai.json
            ├── binary_analysis_nano_ai.json
            └── ...
```

## Fonctionnalités

### 1. Détection Automatique de Domaine

Le système détecte automatiquement le domaine d'un contenu basé sur :
- Mots-clés dans le contenu
- Extension et nom de fichier
- Analyse sémantique

**Domaines supportés :**
- `geophysics_ert` - Géophysique et résistivité électrique
- `pdf_documents` - Documents et rapports
- `binary_analysis` - Analyse de fichiers binaires
- `image_analysis` - Analyse d'images
- `audio_analysis` - Traitement audio
- `video_analysis` - Traitement vidéo
- `3d_modeling` - Modélisation 3D
- `web_research` - Recherche web
- `osm_mapping` - Cartographie OSM
- `programming` - Code et programmation
- `data_science` - Science des données
- `general_knowledge` - Connaissances générales

### 2. Apprentissage Automatique

Le système apprend automatiquement de :

#### a) Fichiers PDF
```python
# Automatique lors de l'upload
pdf_info = {
    'name': 'document.pdf',
    'text': 'contenu...',
    'pages': 10,
    'word_count': 5000
}
auto_learning.learn_from_pdf(pdf_info)
```

#### b) Fichiers Binaires
```python
# Automatique lors de l'analyse
binary_info = {
    'name': 'file.bin',
    'info': {...},
    'text_strings': [...],
    'analysis_report': '...'
}
auto_learning.learn_from_binary(binary_info)
```

#### c) Conversations
```python
# Automatique après chaque réponse
auto_learning.learn_from_conversation(
    query="Question de l'utilisateur",
    response="Réponse de l'IA"
)
```

#### d) Images
```python
# Automatique lors de l'analyse
image_info = {
    'name': 'photo.jpg',
    'ai_analysis': '...',
    'caption': '...'
}
auto_learning.learn_from_image_analysis(image_info)
```

### 3. Création de Nano-IA

Chaque domaine peut avoir sa propre nano-IA spécialisée :

```python
knowledge_manager.create_nano_ai('geophysics_ert')
```

**Profil d'une Nano-IA :**
```json
{
  "name": "Kibali-GEOPHYSICS_ERT",
  "domain": "geophysics_ert",
  "specialization": "geophysics_ert",
  "knowledge_base": {
    "documents_count": 15,
    "keywords": ["resistivité", "ert", "wenner"],
    "last_training": "2025-12-07T..."
  },
  "capabilities": [
    "Analyse spécialisée en geophysics_ert",
    "Réponses basées sur 15 documents",
    "Recherche vectorielle dans connaissances"
  ],
  "performance": {
    "queries_handled": 42,
    "success_rate": 0.95
  },
  "system_prompt": "Tu es Kibali-GEOPHYSICS_ERT..."
}
```

### 4. Enrichissement Contextuel

Les réponses sont automatiquement enrichies avec :

```python
# Contexte du domaine pertinent
domain_context = auto_learning.get_domain_context(query)

# Combine avec le prompt
enriched_prompt = f"""
{conversation_context}

📚 CONNAISSANCES DU DOMAINE:
{domain_context}

QUESTION: {query}
"""
```

## Utilisation dans l'Interface

### Onglet "🤖 Nano-IA & Apprentissage"

1. **Toggle d'activation**
   - Active/désactive l'apprentissage automatique
   - Par défaut : Activé

2. **Statistiques globales**
   - Nombre de domaines
   - Total de documents appris
   - Requêtes traitées
   - Taux de succès moyen

3. **Liste des Nano-IA**
   - Voir toutes les nano-IA créées
   - Performance par domaine
   - Mots-clés maîtrisés
   - Exemples de conversations

4. **Gestion des domaines**
   - Sélectionner un domaine
   - Voir les détails et statistiques
   - Créer/mettre à jour la nano-IA
   - Tester les connaissances

5. **Historique d'apprentissage**
   - Voir les 20 dernières actions d'apprentissage
   - Type, source, domaine

## API Programmatique

### KnowledgeManager

```python
from knowledge_manager import get_knowledge_manager

km = get_knowledge_manager()

# Ajouter des connaissances
km.add_to_domain_knowledge(
    domain='custom_domain',
    content='contenu à apprendre',
    metadata={'source': 'manual'},
    query='question',
    response='réponse'
)

# Interroger un domaine
results = km.query_domain('question', 'domain_name', k=3)

# Interroger tous les domaines
all_results = km.query_all_domains('question', k=3)

# Statistiques
stats = km.get_statistics()

# Créer nano-IA
km.create_nano_ai('domain_name', force=True)
```

### AutoLearningSystem

```python
from auto_learning import get_auto_learning

als = get_auto_learning()

# Apprentissage manuel
als.learn_from_pdf(pdf_info)
als.learn_from_binary(binary_info)
als.learn_from_conversation(query, response)

# Contexte enrichi
context = als.get_domain_context(query, domain='optional')

# Suggestion de domaine
suggested_domain = als.get_suggested_domain(query)
```

## Intégration RAG

Le système s'intègre au RAG vectoriel existant :

1. **Base vectorielle globale** : Documents uploadés (FAISS)
2. **Bases vectorielles par domaine** : Connaissances spécialisées (FAISS)
3. **Mémoire de conversation** : Historique des échanges (FAISS)

**Recherche en cascade :**
```
Query → Nano-IA Domain Detection
     → Domain Vectorstore (spécialisé)
     → Global Vectorstore (général)
     → Chat Memory (historique)
     → Web Search (si activé)
```

## Performance

### Optimisations

1. **Cache du contexte binaire** : Évite réanalyse
2. **Embeddings réutilisés** : Modèle partagé
3. **Index par domaine** : Recherche plus rapide
4. **Limite de keywords** : Max 50 par domaine
5. **Limite d'exemples** : Max 20 par domaine

### Métriques

- Temps d'apprentissage PDF : ~2-5s
- Temps d'apprentissage binaire : ~1-3s
- Temps de requête domaine : ~0.1-0.5s
- Temps création nano-IA : ~0.5-1s

## Exemples d'utilisation

### Scénario 1 : Analyse de résistivité ERT

1. Upload d'un PDF sur la géophysique ERT
2. → Apprentissage automatique dans `geophysics_ert`
3. → Création nano-IA "Kibali-GEOPHYSICS_ERT"
4. Conversation sur la résistivité
5. → Contexte enrichi avec connaissances du domaine
6. → Réponse experte avec citations

### Scénario 2 : Analyse de fichier binaire

1. Upload d'un fichier `.bin`
2. → Analyse : magic bytes, entropie, chaînes
3. → Apprentissage dans `binary_analysis`
4. Question : "C'est quoi ce fichier ?"
5. → Nano-IA binary_analysis activée
6. → Réponse basée sur analyses passées similaires

### Scénario 3 : Multi-domaines

1. Upload PDF géophysique + fichier binaire + image
2. → 3 domaines créés/enrichis
3. → 3 nano-IA spécialisées
4. Question complexe touchant plusieurs domaines
5. → Recherche dans tous les domaines pertinents
6. → Réponse synthétique multi-sources

## Configuration

```python
# Dans app.py
AUTO_LEARNING_AVAILABLE = True  # Activer le système

# Dans session_state
st.session_state.learning_enabled = True  # Toggle utilisateur
st.session_state.auto_learning = get_auto_learning()  # Instance
```

## Logs et Debugging

```python
# Activer les logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)

# Vérifier l'état
km = get_knowledge_manager()
print(km.get_statistics())

# Lister les domaines
print(km.list_domains())

# Lister les nano-IA
print(km.list_nano_ais())
```

## Limitations

1. **Mémoire** : Chaque domaine charge son vectorstore
2. **Tokens** : Contexte enrichi peut être volumineux
3. **Latence** : Recherche multi-domaines peut prendre du temps
4. **Qualité** : Dépend de la qualité des documents sources

## Roadmap

- [ ] Fusion automatique de domaines similaires
- [ ] Apprentissage par renforcement (feedback utilisateur)
- [ ] Export/import de nano-IA
- [ ] Métriques de confiance par réponse
- [ ] Oubli sélectif (pruning de connaissances)
- [ ] Nano-IA collaborative (communication inter-domaines)

## Support

Pour toute question sur le système d'auto-apprentissage, consultez les logs ou contactez l'équipe de développement.
