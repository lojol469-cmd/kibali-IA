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
        """Effectue une recherche dans les PDFs via la base vectorielle"""
        try:
            # Récupérer la vectordb du contexte
            vectordb = context.get('vectordb') if context else None
            rag_docs = context.get('rag_docs', []) if context else []
            
            if not vectordb and not rag_docs:
                return {
                    'success': False,
                    'error': 'Aucune base de documents disponible',
                    'output': '⚠️ Aucun document PDF n\'est actuellement chargé dans le système.'
                }
            
            # Si des docs ont déjà été trouvés, les utiliser
            if rag_docs:
                results_text = f"📚 **{len(rag_docs)} documents pertinents trouvés:**\n\n"
                for i, doc in enumerate(rag_docs[:5], 1):
                    content = doc.page_content[:400].replace('\n', ' ').strip()
                    source = doc.metadata.get('source', 'Source inconnue') if hasattr(doc, 'metadata') else 'Source inconnue'
                    results_text += f"**Document {i}** ({source}):\n{content}...\n\n"
                
                return {
                    'success': True,
                    'output': results_text,
                    'content': results_text,
                    'num_results': len(rag_docs),
                    'documents': [{'content': doc.page_content, 'metadata': doc.metadata} for doc in rag_docs]
                }
            
            # Sinon, faire une recherche dans la vectordb
            elif vectordb:
                search_results = vectordb.similarity_search(query, k=10)
                
                if not search_results:
                    return {
                        'success': False,
                        'output': '🔍 Aucun résultat trouvé pour cette recherche dans les documents.',
                        'num_results': 0
                    }
                
                results_text = f"📚 **{len(search_results)} résultats trouvés:**\n\n"
                for i, doc in enumerate(search_results[:5], 1):
                    content = doc.page_content[:400].replace('\n', ' ').strip()
                    source = doc.metadata.get('source', 'Source inconnue') if hasattr(doc, 'metadata') else 'Source inconnue'
                    results_text += f"**{i}.** ({source})\n{content}...\n\n"
                
                return {
                    'success': True,
                    'output': results_text,
                    'content': results_text,
                    'num_results': len(search_results),
                    'documents': [{'content': doc.page_content, 'metadata': doc.metadata} for doc in search_results]
                }
            
            return {
                'success': False,
                'error': 'Configuration invalide',
                'output': '⚠️ Erreur de configuration de la recherche documentaire.'
            }

        except Exception as e:
            return {
                'error': f"Erreur lors de la recherche PDF: {str(e)}",
                'output': f"❌ Erreur: {str(e)}",
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
