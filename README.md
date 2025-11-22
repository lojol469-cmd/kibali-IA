# 🚀 Kibali IA - Assistant Avancé

[![GitHub](https://img.shields.io/badge/GitHub-lojol469--cmd/kibali--IA-blue)](https://github.com/lojol469-cmd/kibali-IA)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)

> **Kibali** - Un assistant IA avancé avec interface moderne et système d'outils dynamiques

![Kibali Logo](https://img.shields.io/badge/K-00ff88?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJDMTMuMSAyIDE0IDIuOSAxNCA0VjIwQzE0IDIxLjEgMTMuMSAyMiAxMiAyMkgxMEM4LjkgMjIgOCAycTEuMSAwIDIgMnoiIGZpbGw9IiMwMGZmODgiLz4KPHN2Zz4=)

## ✨ Fonctionnalités Principales

### 🤖 Intelligence Artificielle
- **Modèles avancés** : DeepSeek V3, Gemma 2B, Llama 3.1 8B, Qwen 2.5 7B, SmolLM 3B
- **Chat RAG** : Recherche dans vos documents avec base vectorielle FAISS
- **Génération de contenu** : Images, vidéos, audio, modèles 3D avec Stable Diffusion

### 🌐 Recherche Web Intelligente
- **Recherche hybride** : Combine données locales + web en temps réel
- **Cache intelligent** : Optimise les performances et réduit les coûts API
- **Sources multiples** : Tavily API pour recherche web avancée
- **Extraction de contenu** : Analyse automatique des pages web

### 🗺️ Géolocalisation & Cartes
- **Calcul de trajets** : Intinéraires routiers avec OpenStreetMap
- **Cartes interactives** : Visualisation des parcours avec NetworkX
- **Points d'intérêt** : Recherche et affichage des POIs

### 📸 Analyse d'Images Avancée
- **Classification de sols** : Analyse géologique et agricole
- **Détection d'objets** : Routes, bâtiments, clôtures avec OpenCV
- **Simulation IR** : Imagerie infrarouge simulée
- **Analyse d'anomalies** : Détection de défauts structurels

### 🔧 Système d'Outils Dynamiques
- **Architecture modulaire** : Outils dans le dossier `/outils/`
- **Chargement dynamique** : Les outils s'adaptent au contexte
- **IA et logiciels** : Intégration d'outils spécialisés

## 🎨 Interface Moderne

### Design Kibali
- **Logo K animé** : Effets 3D avec couleurs vert/jaune/bleu
- **Thème sombre amélioré** : Moins noir, plus moderne et lisible
- **Animations fluides** : Transitions CSS avancées
- **Responsive design** : Adapté mobile et desktop

### UX/UI
- **Cartes flex** : Layout responsive avec zoom au survol
- **Gradients subtils** : Effets visuels modernes
- **Animations interactives** : Feedback utilisateur engageant

## 🚀 Installation & Utilisation

### Prérequis
```bash
Python 3.8+
Git
```

### Installation
```bash
# Clonez le repository
git clone https://github.com/lojol469-cmd/kibali-IA.git
cd kibali-IA

# Installez les dépendances
pip install -r requirements.txt
```

### Configuration
```bash
# Copiez le fichier d'exemple
cp .env.example .env

# Éditez .env avec vos tokens
nano .env
```

**Tokens requis :**
- **HF_TOKEN** : [HuggingFace](https://huggingface.co/settings/tokens)
- **TAVILY_API_KEY** : [Tavily](https://app.tavily.com/)

### Lancement
```bash
# Lancez l'application
streamlit run app.py

# Accédez à http://localhost:8503
```

## 📁 Structure du Projet

```
kibali-IA/
├── app.py                 # Application principale Streamlit
├── requirements.txt       # Dépendances Python
├── .env.example          # Configuration d'exemple
├── .gitignore           # Fichiers exclus
├── kibali_data/         # Données locales (auto-généré)
│   ├── vectordb/        # Base vectorielle FAISS
│   ├── pdfs/           # Documents PDF
│   ├── graphs/         # Graphes OSM
│   ├── maps/           # Cartes générées
│   └── generated/      # Contenu généré
└── outils/              # Système d'outils dynamiques
    ├── __init__.py
    ├── web_search_tool.py
    ├── image_analysis_tool.py
    ├── code_generation_tool.py
    ├── file_management_tool.py
    ├── math_tool.py
    └── translation_tool.py
```

## 🛠️ Technologies Utilisées

### Core
- **Streamlit** : Interface web interactive
- **HuggingFace** : Modèles d'IA et API
- **LangChain** : Orchestration d'agents IA
- **FAISS** : Base de données vectorielle

### Analyse & Traitement
- **OpenCV** : Vision par ordinateur
- **PyMuPDF** : Traitement PDF
- **NetworkX** : Graphes et réseaux
- **Open3D** : Nuages de points 3D

### Génération
- **Diffusers** : Génération d'images/vidéos
- **Transformers** : Modèles de langage
- **Torch** : Deep learning

### Web & API
- **Requests** : HTTP client
- **BeautifulSoup** : Parsing HTML
- **Tavily API** : Recherche web

## 🔧 Architecture

### Système d'Outils Dynamiques
```python
# Exemple d'utilisation
from outils import ToolManager

tool_manager = ToolManager()
result = tool_manager.execute("web_search", query="IA 2024")
```

### Agent LangChain
```python
# Configuration d'agent avec outils
agent = create_enhanced_agent(
    model_name="deepseek-ai/DeepSeek-V3-0324",
    vectordb=vectordb,
    graph=osm_graph,
    pois=points_of_interest
)
```

## 📊 Fonctionnalités Détaillées

### Chat RAG + Web
- Recherche hybride locale + web
- Cache intelligent avec expiration
- Expansion automatique de requêtes
- Sources multiples avec citations

### Analyse d'Images
- **Classification** : Sols argileux, sableux, limoneux
- **Détection** : Objets, clôtures, anomalies
- **Simulation** : Infrarouge, photogrammétrie
- **Géologie** : Clustering de textures

### Calcul de Trajets
- **OSM Integration** : Données OpenStreetMap
- **Algorithmes** : Plus court chemin avec NetworkX
- **Visualisation** : Cartes 2D interactives
- **POIs** : Recherche de points d'intérêt

## 🔐 Sécurité

- **Tokens chiffrés** : Stockage sécurisé des API keys
- **.gitignore configuré** : Exclusion des données sensibles
- **Validation d'entrée** : Protection contre les injections
- **Rate limiting** : Gestion des quotas API

## 🤝 Contribution

1. Fork le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez (`git commit -m 'Add some AmazingFeature'`)
4. Pushez (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- **HuggingFace** pour les modèles d'IA
- **Streamlit** pour le framework web
- **OpenStreetMap** pour les données cartographiques
- **Tavily** pour l'API de recherche

## 📞 Support

- **Issues** : [GitHub Issues](https://github.com/lojol469-cmd/kibali-IA/issues)
- **Discussions** : [GitHub Discussions](https://github.com/lojol469-cmd/kibali-IA/discussions)

---

**Développé avec ❤️ par [lojol469-cmd](https://github.com/lojol469-cmd)**

![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red)</content>
<parameter name="filePath">/home/belikan/KIBAlONEia/README.md