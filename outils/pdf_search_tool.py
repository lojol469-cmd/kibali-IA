"""
Outil de Recherche dans les PDFs
Recherche et analyse des documents PDF dans la base vectorielle
"""

import re
from typing import Dict, List, Any, Optional
from .__init__ import BaseTool

class PDFSearchTool(BaseTool):
    """Outil spécialisé dans la recherche de documents PDF"""

    @property
    def name(self) -> str:
        return "pdf_document_search"

    @property
    def description(self) -> str:
        return "Recherche intelligente dans les documents PDF avec analyse sémantique"

    @property
    def capabilities(self) -> List[str]:
        return [
            "recherche pdf",
            "recherche documents",
            "analyse documents",
            "géophysique",
            "géologie",
            "minéralogie",
            "documents techniques",
            "rapports",
            "études",
            "recherche scientifique",
            "base de données",
            "base vectorielle",
            "rag",
            "extraction pdf"
        ]

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        """Détermine si la requête nécessite une recherche dans les PDFs"""
        pdf_keywords = [
            "pdf", "document", "documents", "fichier", "fichiers",
            "rapport", "rapports", "étude", "études",
            "recherche", "chercher dans", "trouver dans",
            "géophysique", "géologie", "géologique", "géologiques",
            "minéral", "minéraux", "minéralogie", "mining",
            "exploration", "forage", "sondage",
            "base", "base de données", "vectordb", "vectorielle",
            "rag", "extraction", "texte",
            "analyse", "analyser", "consulter",
            "qu'est-ce qui", "que dit", "selon les documents"
        ]

        query_lower = query.lower()

        # Score basé sur les mots-clés
        score = 0.0
        for keyword in pdf_keywords:
            if keyword in query_lower:
                score += 0.35

        # Questions indiquant une recherche documentaire
        doc_patterns = [
            r"(?:recherche|cherche|trouve).+(?:pdf|document|fichier)",
            r"(?:combien|liste|quels?).+(?:pdf|document|fichier)",
            r"(?:géophysique|géologie|minéral|exploration)",
            r"(?:dans|selon|d'après).+(?:document|rapport|étude)",
            r"(?:base|vectordb|vectorielle)",
            r"\d+\s+(?:pdf|document|fichier)"  # Ex: "10 pdfs"
        ]
        
        for pattern in doc_patterns:
            if re.search(pattern, query_lower):
                score += 0.4

        # Contexte avec PDFs
        if context:
            if context.get('has_pdfs', False):
                score += 0.3
            if context.get('vectordb_available', False):
                score += 0.3

        return min(score, 1.0)

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Effectue une recherche dans les PDFs"""
        try:
            # Message indiquant que l'outil nécessite l'intégration avec RAG
            return {
                'query': query,
                'tool': 'pdf_document_search',
                'action': 'search_pdfs',
                'message': '🔍 Recherche dans les documents PDF...',
                'instructions': [
                    "1. Charger la base vectorielle si pas déjà chargée",
                    "2. Effectuer une recherche sémantique sur la requête",
                    "3. Extraire les passages pertinents des PDFs",
                    "4. Synthétiser les informations trouvées"
                ],
                'search_params': {
                    'k': 10,  # Nombre de résultats à chercher
                    'search_type': 'semantic',
                    'filter_topic': self._extract_topic(query)
                },
                'success': True
            }

        except Exception as e:
            return {
                'error': f"Erreur lors de la recherche PDF: {str(e)}",
                'query': query,
                'success': False
            }

    def _extract_topic(self, query: str) -> str:
        """Extrait le sujet principal de la requête"""
        query_lower = query.lower()
        
        # Thèmes scientifiques
        topics = {
            'géophysique': ['géophysique', 'géophysiques', 'sismique', 'magnétique', 'gravimétrique'],
            'géologie': ['géologie', 'géologique', 'géologiques', 'formation', 'stratigraphie'],
            'minéralogie': ['minéral', 'minéraux', 'minéralogie', 'minerai', 'gisement'],
            'exploration': ['exploration', 'prospection', 'forage', 'sondage'],
            'hydrogéologie': ['eau', 'hydro', 'aquifère', 'nappe'],
            'environnement': ['environnement', 'impact', 'écologique']
        }
        
        for topic, keywords in topics.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return topic
        
        return 'général'
