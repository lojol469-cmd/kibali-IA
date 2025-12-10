# 📁 Organisation Modulaire de Kibali-IA

## 🎯 Principe d'Organisation

**Règle d'or**: Chaque nouveau module (hors outils) doit être dans son propre dossier avec une structure claire.

---

## 📂 Structure Actuelle

```
kibali-IA/
│
├── 🔧 OUTILS (Système de plugins dynamiques)
│   └── outils/
│       ├── __init__.py                      # BaseTool + ToolManager
│       ├── excel_organizer_tool.py         # Organisation Excel
│       ├── photogrammetry_optimizer_tool.py # Optimisation 3D
│       ├── code_generator_tool.py          # Génération de code
│       ├── pdf_document_search_tool.py     # Recherche dans PDFs
│       ├── image_analysis_tool.py          # Analyse d'images
│       ├── web_search_advanced_tool.py     # Recherche web
│       ├── file_manager_tool.py            # Gestion de fichiers
│       ├── translation_tool.py             # Traduction
│       ├── math_calculator_tool.py         # Calculs mathématiques
│       └── pdf_generator_tool.py           # Génération de PDFs
│
├── 🧠 MODULES D'APPRENTISSAGE
│   ├── auto_learning/                       # Module d'apprentissage automatique
│   │   ├── __init__.py                     # Point d'entrée
│   │   └── auto_learning.py                # Classe AutoLearning
│   │
│   ├── knowledge_manager/                   # Module de gestion des connaissances
│   │   ├── __init__.py                     # Point d'entrée
│   │   └── knowledge_manager.py            # Classe KnowledgeManager
│   │
│   └── chat_memory.py                       # Mémoire conversationnelle (à migrer)
│
├── 💾 DONNÉES PERSISTANTES
│   ├── auto_learning_data/                  # Données d'apprentissage
│   │   ├── conversations.json
│   │   ├── patterns.json
│   │   ├── concepts.json
│   │   ├── file_knowledge.json
│   │   └── embeddings.pkl
│   │
│   ├── knowledge_data/                      # Base de connaissances
│   │   ├── patterns_db.json
│   │   ├── weights.json
│   │   ├── recommendations.json
│   │   └── history.json
│   │
│   ├── chat_memory/                         # Mémoire des conversations
│   │   └── chat_memory.pkl
│   │
│   └── pdfs_uploaded/                       # PDFs uploadés
│
├── 🎨 APPLICATION PRINCIPALE
│   ├── app.py                              # Interface Streamlit principale
│   └── lifemodo_kernel.py                  # Ancien système
│
└── 📚 DOCUMENTATION
    ├── AUTO_APPRENTISSAGE_README.md        # Documentation auto-apprentissage
    ├── ORGANISATION_MODULAIRE.md           # Ce fichier
    ├── RESUME_OUTILS.md                    # Documentation des outils
    └── README.md                           # Documentation générale
```

---

## 🏗️ Template pour Nouveau Module

### Structure Minimale
```
nouveau_module/
├── __init__.py              # Point d'entrée obligatoire
├── nouveau_module.py        # Implémentation principale
├── helpers.py               # (Optionnel) Fonctions utilitaires
└── README.md                # (Recommandé) Documentation
```

### Exemple de `__init__.py`
```python
"""
Module [Nom du Module] pour Kibali-IA
[Description brève]
"""

from .nouveau_module import MaClasse, ma_fonction

__all__ = ['MaClasse', 'ma_fonction']
```

### Exemple de `nouveau_module.py`
```python
#!/usr/bin/env python3
"""
[Nom du Module] - [Description détaillée]
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class MaClasse:
    """
    [Description de la classe]
    """
    
    def __init__(self, storage_dir: str = "/home/belikan/kibali-IA/mon_module_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def ma_methode(self, param: str) -> Dict[str, Any]:
        """
        [Description de la méthode]
        
        Args:
            param: Description du paramètre
        
        Returns:
            Résultat de l'opération
        """
        return {"status": "success"}

# Instance globale
_instance = None

def get_ma_classe() -> MaClasse:
    """Retourne l'instance globale singleton"""
    global _instance
    if _instance is None:
        _instance = MaClasse()
    return _instance

if __name__ == "__main__":
    # Tests unitaires
    mc = get_ma_classe()
    print("✅ Module testé avec succès")
```

---

## 🎯 Bonnes Pratiques

### 1. **Séparation des Responsabilités**
- **Outils** (`/outils/`): Plugins pour actions spécifiques (génération, recherche, calcul)
- **Modules** (`/module_name/`): Systèmes complexes avec état (apprentissage, mémoire)
- **Données** (`/module_data/`): Fichiers de persistance (JSON, pickle, etc.)

### 2. **Nommage Cohérent**
```
Module: auto_learning/
├── Code: auto_learning.py
├── Données: auto_learning_data/
└── Doc: AUTO_APPRENTISSAGE_README.md
```

### 3. **Import Simplifié**
```python
# ✅ BON - Import depuis le module
from auto_learning import get_auto_learning

# ❌ MAUVAIS - Import direct du fichier
from auto_learning.auto_learning import AutoLearning
```

### 4. **Pattern Singleton**
```python
# Utiliser une instance globale pour les modules avec état
_instance = None

def get_module() -> Module:
    global _instance
    if _instance is None:
        _instance = Module()
    return _instance
```

### 5. **Gestion des Données**
```python
# Créer automatiquement les dossiers de données
storage_dir = Path("/home/belikan/kibali-IA/mon_module_data")
storage_dir.mkdir(parents=True, exist_ok=True)

# Utiliser des formats standards
# - JSON pour données lisibles
# - Pickle pour objets Python complexes
# - CSV pour tableaux de données
```

---

## 🔄 Migration d'un Fichier Existant

### Avant
```
kibali-IA/
├── app.py
└── mon_module.py  ← Fichier isolé
```

### Après
```
kibali-IA/
├── app.py
└── mon_module/    ← Module organisé
    ├── __init__.py
    └── mon_module.py
```

### Étapes
```bash
# 1. Créer le dossier module
mkdir mon_module

# 2. Déplacer le fichier
mv mon_module.py mon_module/

# 3. Créer __init__.py
cat > mon_module/__init__.py << 'EOF'
from .mon_module import *
EOF

# 4. Mettre à jour les imports dans app.py
# Avant: import mon_module
# Après: from mon_module import ...
```

---

## 📦 Modules Actuels

### 1. **auto_learning** 🧠
- **But**: Apprentissage automatique des conversations, PDFs, fichiers
- **Données**: `/auto_learning_data/`
- **Usage**: `from auto_learning import get_auto_learning`

### 2. **knowledge_manager** 🗂️
- **But**: Gestion centralisée des patterns et connaissances
- **Données**: `/knowledge_data/`
- **Usage**: `from knowledge_manager import get_knowledge_manager`

### 3. **outils** 🔧
- **But**: Système de plugins dynamiques pour actions spécifiques
- **Données**: Aucune (stateless)
- **Usage**: `from outils import get_tool_manager`

---

## 🚀 Modules à Créer (Suggestions)

### 1. **chat_memory** → Module
```
chat_memory/
├── __init__.py
├── chat_memory.py           # Mémoire vectorielle
├── embeddings.py            # Gestion des embeddings
└── README.md
```

### 2. **web_scraper** → Module
```
web_scraper/
├── __init__.py
├── web_scraper.py           # Scraping intelligent
├── cache.py                 # Cache des résultats
└── README.md
```

### 3. **task_orchestrator** → Module
```
task_orchestrator/
├── __init__.py
├── orchestrator.py          # Orchestration des tâches
├── task_queue.py            # File d'attente
└── README.md
```

### 4. **file_analyzer** → Module
```
file_analyzer/
├── __init__.py
├── analyzer.py              # Analyse multi-format
├── binary_detector.py       # Détection binaire
└── README.md
```

---

## ✅ Checklist pour Nouveau Module

- [ ] Créer dossier `/nouveau_module/`
- [ ] Créer `__init__.py` avec exports
- [ ] Créer `nouveau_module.py` avec implémentation
- [ ] Implémenter pattern singleton si nécessaire
- [ ] Créer dossier données `/nouveau_module_data/` si persistance
- [ ] Ajouter docstrings complètes
- [ ] Créer tests unitaires dans `if __name__ == "__main__"`
- [ ] Créer `README.md` avec documentation
- [ ] Mettre à jour `ORGANISATION_MODULAIRE.md`
- [ ] Tester l'import: `from nouveau_module import ...`
- [ ] Intégrer dans `app.py` si nécessaire

---

## 🎓 Exemples d'Import

### Module Simple
```python
# Dans app.py
from mon_module import MaClasse, ma_fonction

# Utilisation
obj = MaClasse()
result = ma_fonction(param)
```

### Module avec Singleton
```python
# Dans app.py
from mon_module import get_mon_module

# Utilisation (toujours la même instance)
module = get_mon_module()
result = module.faire_quelquechose()
```

### Gestion Conditionnelle
```python
# Dans app.py
try:
    from mon_module import get_mon_module
    MON_MODULE_AVAILABLE = True
    print("✅ Module chargé")
except Exception as e:
    MON_MODULE_AVAILABLE = False
    get_mon_module = None
    print(f"⚠️ Module non disponible: {e}")

# Plus tard dans le code
if MON_MODULE_AVAILABLE:
    module = get_mon_module()
    result = module.action()
```

---

## 🔧 Outils vs Modules

### Différences Clés

| Aspect | **Outils** | **Modules** |
|--------|-----------|-----------|
| **Localisation** | `/outils/` | `/nom_module/` |
| **Héritage** | `BaseTool` | Classe libre |
| **État** | Stateless | Stateful possible |
| **Persistance** | Non | Oui (`/nom_module_data/`) |
| **Chargement** | Dynamique (ToolManager) | Import direct |
| **Usage** | Actions ponctuelles | Systèmes complexes |

### Quand Créer un Outil ?
- Action spécifique et isolée
- Pas besoin de mémoire entre appels
- Utilisé via le système de sélection automatique
- Exemples: calculatrice, traducteur, générateur

### Quand Créer un Module ?
- Système avec état et mémoire
- Besoin de persistance des données
- Utilisation fréquente dans toute l'app
- Exemples: apprentissage, cache, base de données

---

## 📊 Structure Recommandée Finale

```
kibali-IA/
├── 🔧 Outils (Actions)
│   └── outils/
│
├── 🧠 Modules (Systèmes)
│   ├── auto_learning/
│   ├── knowledge_manager/
│   ├── chat_memory/
│   ├── web_scraper/
│   └── task_orchestrator/
│
├── 💾 Données
│   ├── auto_learning_data/
│   ├── knowledge_data/
│   ├── chat_memory/
│   └── cache/
│
├── 🎨 Interface
│   └── app.py
│
└── 📚 Documentation
    ├── AUTO_APPRENTISSAGE_README.md
    ├── ORGANISATION_MODULAIRE.md
    └── README.md
```

---

## 🎉 Avantages de cette Organisation

### 1. **Clarté**
- Chaque module a son dossier
- Structure prévisible
- Facile à naviguer

### 2. **Maintenabilité**
- Modifications isolées par module
- Tests unitaires par module
- Documentation co-localisée

### 3. **Scalabilité**
- Ajout facile de nouveaux modules
- Pas de conflit de noms
- Dépendances claires

### 4. **Collaboration**
- Plusieurs développeurs peuvent travailler sur des modules différents
- Merge conflicts réduits
- Responsabilités définies

### 5. **Réutilisabilité**
- Modules peuvent être extraits facilement
- Import dans d'autres projets
- Tests indépendants

---

## 📝 Conclusion

L'organisation modulaire de Kibali-IA suit maintenant les meilleures pratiques:
- ✅ Modules dans des dossiers dédiés
- ✅ `__init__.py` pour chaque module
- ✅ Pattern singleton pour instances globales
- ✅ Données séparées du code
- ✅ Documentation claire

**Prêt pour une croissance propre et organisée ! 🚀**
