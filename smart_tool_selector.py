"""
Sélecteur intelligent d'outils pour Kibali AI
Analyse le contexte et choisit dynamiquement les outils appropriés
"""

import re
from typing import List, Dict, Set, Any
from dataclasses import dataclass

@dataclass
class ToolIntent:
    """Représente une intention d'utilisation d'outil"""
    name: str
    confidence: float  # 0.0 à 1.0
    reason: str
    priority: int  # Plus petit = plus prioritaire

class SmartToolSelector:
    """Sélectionneur intelligent d'outils basé sur l'analyse contextuelle"""
    
    def __init__(self):
        # Patterns de détection pour chaque type d'outil
        self.tool_patterns = {
            'calculation': {
                'keywords': ['calcul', 'moyenne', 'somme', 'total', 'combien', 'statistique', 
                            'minimum', 'maximum', 'médiane', 'écart-type', 'nombre de', 'compte'],
                'patterns': [r'\d+\s*[+\-*/]\s*\d+', r'combien de', r'quelle est la', r'calcule'],
                'priority': 1
            },
            'web_search': {
                'keywords': ['recherche', 'trouve', 'internet', 'web', 'google', 'information sur',
                            'qu\'est-ce que', 'c\'est quoi', 'définition', 'explique', 'qui est'],
                'patterns': [r'(qu\'?est[- ]ce que|c\'?est quoi)', r'recherche.*sur', r'trouve.*info'],
                'priority': 2
            },
            'document_search': {
                'keywords': ['document', 'pdf', 'texte', 'cherche dans', 'trouve dans les documents',
                            'selon les documents', 'd\'après les fichiers'],
                'patterns': [r'dans (le|les) (document|pdf|fichier)s?', r'selon (le|les) (document|pdf)s?'],
                'priority': 1
            },
            'data_extraction': {
                'keywords': ['extrait', 'données', 'valeurs', 'liste', 'tableau', 'affiche',
                            'montre-moi', 'donne-moi les', 'quelles sont les'],
                'patterns': [r'(extrait|liste|affiche).*données?', r'montre(-| )moi', r'donne(-| )moi'],
                'priority': 1
            },
            'visualization': {
                'keywords': ['graphique', 'courbe', 'diagramme', 'visualise', 'plot', 'chart',
                            'histogramme', 'camembert', 'barres', 'ligne'],
                'patterns': [r'(fait|crée|génère).*graphique', r'visualise', r'dessine'],
                'priority': 2
            },
            'file_analysis': {
                'keywords': ['analyse', 'fichier', 'structure', 'format', 'contenu', 'type de fichier',
                            'qu\'est-ce que ce fichier', 'c\'est quel fichier'],
                'patterns': [r'analys[eé].*fichier', r'quel (type|format)', r'structure du fichier'],
                'priority': 1
            },
            'translation': {
                'keywords': ['traduis', 'translate', 'traduction', 'en français', 'en anglais',
                            'en espagnol', 'en allemand'],
                'patterns': [r'tradui[st]', r'en (français|anglais|espagnol|allemand|chinois)'],
                'priority': 2
            },
            'code_generation': {
                'keywords': ['code', 'programme', 'script', 'fonction', 'classe', 'python', 'javascript',
                            'génère du code', 'écris un programme'],
                'patterns': [r'(écris|génère|crée).*code', r'(fonction|classe|script).*pour'],
                'priority': 3
            },
            'summarization': {
                'keywords': ['résume', 'résumé', 'synthèse', 'en bref', 'principales idées',
                            'essentiel', 'condensé'],
                'patterns': [r'résum[eé]', r'en bref', r'synthèse', r'principa(les|ux)'],
                'priority': 2
            },
            'comparison': {
                'keywords': ['compare', 'comparaison', 'différence', 'versus', 'vs', 'contre',
                            'similaire', 'diffère'],
                'patterns': [r'compar[eé]', r'différence entre', r'versus|vs|contre'],
                'priority': 2
            }
        }
        
    def analyze_intent(self, query: str, context: Dict[str, Any] = None) -> List[ToolIntent]:
        """
        Analyse une requête et retourne les outils recommandés avec leur confiance
        
        Args:
            query: La question/requête de l'utilisateur
            context: Contexte additionnel (fichiers chargés, historique, etc.)
            
        Returns:
            Liste de ToolIntent triés par priorité et confiance
        """
        query_lower = query.lower()
        intents = []
        
        # Analyser chaque outil potentiel
        for tool_name, config in self.tool_patterns.items():
            confidence = 0.0
            reasons = []
            
            # 1. Vérification des mots-clés
            keyword_matches = sum(1 for kw in config['keywords'] if kw in query_lower)
            if keyword_matches > 0:
                confidence += min(keyword_matches * 0.2, 0.6)  # Max 0.6 pour les keywords
                reasons.append(f"{keyword_matches} mot(s)-clé détecté(s)")
            
            # 2. Vérification des patterns regex
            pattern_matches = sum(1 for pattern in config['patterns'] if re.search(pattern, query_lower))
            if pattern_matches > 0:
                confidence += min(pattern_matches * 0.3, 0.4)  # Max 0.4 pour les patterns
                reasons.append(f"{pattern_matches} pattern(s) reconnu(s)")
            
            # 3. Analyse contextuelle
            if context:
                # Fichiers binaires présents → boost file_analysis et data_extraction
                if context.get('has_binary_files') and tool_name in ['file_analysis', 'data_extraction']:
                    confidence += 0.3
                    reasons.append("Fichier binaire présent")
                
                # PDFs chargés → boost document_search
                if context.get('has_pdfs') and tool_name == 'document_search':
                    confidence += 0.3
                    reasons.append("Documents PDF disponibles")
                
                # Données numériques détectées → boost calculation et visualization
                if context.get('has_numeric_data') and tool_name in ['calculation', 'visualization']:
                    confidence += 0.2
                    reasons.append("Données numériques détectées")
                
                # Recherche web activée → boost web_search
                if context.get('web_enabled') and tool_name == 'web_search':
                    confidence += 0.1
                    reasons.append("Web activé")
            
            # 4. Heuristiques spéciales
            # Questions ouvertes → boost web_search
            if tool_name == 'web_search' and any(q in query_lower for q in ['qu\'est-ce', 'c\'est quoi', 'pourquoi', 'comment']):
                confidence += 0.15
                reasons.append("Question ouverte détectée")
            
            # Question avec chiffres → boost calculation
            if tool_name == 'calculation' and re.search(r'\d+', query):
                confidence += 0.1
                reasons.append("Chiffres dans la question")
            
            # Si confiance significative, ajouter l'intent
            if confidence >= 0.3:  # Seuil minimum de 30%
                intents.append(ToolIntent(
                    name=tool_name,
                    confidence=min(confidence, 1.0),  # Cap à 1.0
                    reason=" | ".join(reasons),
                    priority=config['priority']
                ))
        
        # Trier par priorité puis par confiance
        intents.sort(key=lambda x: (x.priority, -x.confidence))
        
        return intents
    
    def select_tools(self, query: str, context: Dict[str, Any] = None, max_tools: int = 3) -> List[str]:
        """
        Sélectionne les outils les plus pertinents
        
        Args:
            query: La requête utilisateur
            context: Contexte additionnel
            max_tools: Nombre maximum d'outils à retourner
            
        Returns:
            Liste des noms d'outils sélectionnés
        """
        intents = self.analyze_intent(query, context)
        
        # Filtrer les outils avec confiance > 40%
        high_confidence_tools = [intent.name for intent in intents if intent.confidence >= 0.4]
        
        # Retourner max_tools outils
        return high_confidence_tools[:max_tools]
    
    def get_tool_explanation(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Génère une explication des outils sélectionnés
        
        Returns:
            Texte explicatif HTML
        """
        intents = self.analyze_intent(query, context)
        
        if not intents:
            return "Aucun outil spécifique détecté - conversation générale"
        
        explanation = "**🔧 Outils sélectionnés automatiquement :**\n\n"
        
        for i, intent in enumerate(intents[:3], 1):
            confidence_bar = "🟢" * int(intent.confidence * 5) + "⚪" * (5 - int(intent.confidence * 5))
            explanation += f"{i}. **{intent.name}** {confidence_bar} ({intent.confidence*100:.0f}%)\n"
            explanation += f"   └─ {intent.reason}\n\n"
        
        return explanation
    
    def should_combine_tools(self, intents: List[ToolIntent]) -> bool:
        """Détermine si plusieurs outils doivent être combinés"""
        # Combiner si plusieurs outils ont une confiance > 50%
        high_conf_count = sum(1 for intent in intents if intent.confidence >= 0.5)
        return high_conf_count >= 2
    
    def get_tool_execution_order(self, tool_names: List[str]) -> List[str]:
        """
        Détermine l'ordre optimal d'exécution des outils
        Certains outils doivent s'exécuter avant d'autres
        """
        # Ordre de dépendances
        order_priority = {
            'file_analysis': 1,      # Toujours en premier
            'data_extraction': 2,    # Avant calculs et viz
            'document_search': 2,    # Avant autres recherches
            'calculation': 3,        # Après extraction
            'visualization': 4,      # Après calculs
            'web_search': 3,         # Peut être parallèle
            'summarization': 5,      # Après tout
            'comparison': 4,         # Après recherches
            'translation': 5,        # En dernier
            'code_generation': 4     # Après analyse
        }
        
        # Trier selon la priorité
        sorted_tools = sorted(tool_names, key=lambda x: order_priority.get(x, 99))
        return sorted_tools


# Instance globale
_selector = None

def get_smart_selector() -> SmartToolSelector:
    """Retourne l'instance globale du sélecteur"""
    global _selector
    if _selector is None:
        _selector = SmartToolSelector()
    return _selector
