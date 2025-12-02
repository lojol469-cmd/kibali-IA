"""
Outil de Traduction Multilingue
Traduit du texte entre différentes langues
"""

import re
import time
from typing import Dict, List, Any, Optional
from .__init__ import BaseTool

class TranslationTool(BaseTool):
    """Outil spécialisé dans la traduction de texte"""

    @property
    def name(self) -> str:
        return "translation_tool"

    @property
    def description(self) -> str:
        return "Traducteur multilingue avec détection automatique de langue"

    @property
    def capabilities(self) -> List[str]:
        return [
            "traduction français-anglais",
            "traduction anglais-français",
            "traduction espagnol",
            "traduction allemand",
            "traduction italien",
            "détection de langue",
            "traduction technique",
            "traduction littéraire"
        ]

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        """Détermine si la requête concerne une traduction"""
        translation_keywords = [
            "traduire", "traduis", "translate", "translation",
            "français", "anglais", "english", "espagnol", "spanish",
            "allemand", "german", "italien", "italian", "chinois",
            "japonais", "russe", "arabe", "en français", "in english",
            "vers", "to", "from"
        ]

        query_lower = query.lower()

        # Mots-clés de traduction
        score = 0.0
        for keyword in translation_keywords:
            if keyword in query_lower:
                score += 0.4

        # Patterns de langues
        language_patterns = [
            r'\b(français|english|espagnol|allemand|italien)\b',
            r'\b(en|to|vers|dans)\s+(français|english|espagnol|allemand|italien)\b',
            r'\b(de|du|from)\s+(français|english|espagnol|allemand|italien)\b'
        ]

        for pattern in language_patterns:
            if re.search(pattern, query_lower):
                score += 0.5

        # Contexte avec texte à traduire
        if context and 'text_to_translate' in context:
            score += 0.8

        return min(score, 1.0)

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Effectue la traduction demandée"""
        try:
            # Extraire les langues source et cible
            source_lang, target_lang = self._extract_languages(query)

            # Extraire le texte à traduire
            text_to_translate = self._extract_text_to_translate(query, context)

            if not text_to_translate:
                return {
                    'error': 'Aucun texte à traduire trouvé',
                    'query': query
                }

            # Détecter la langue source si non spécifiée
            if not source_lang:
                source_lang = self._detect_language(text_to_translate)

            # Traduire le texte
            translation = self._translate_text(text_to_translate, source_lang, target_lang)

            return {
                'query': query,
                'original_text': text_to_translate,
                'source_language': source_lang,
                'target_language': target_lang,
                'translation': translation,
                'timestamp': time.time()
            }

        except Exception as e:
            return {
                'error': f"Erreur lors de la traduction: {str(e)}",
                'query': query
            }

    def _extract_languages(self, query: str) -> tuple:
        """Extrait les langues source et cible de la requête"""
        query_lower = query.lower()

        # Mapping des langues
        language_map = {
            'français': 'fr', 'french': 'fr', 'francais': 'fr',
            'anglais': 'en', 'english': 'en',
            'espagnol': 'es', 'spanish': 'es',
            'allemand': 'de', 'german': 'de',
            'italien': 'it', 'italian': 'it',
            'chinois': 'zh', 'chinese': 'zh',
            'japonais': 'ja', 'japanese': 'ja',
            'russe': 'ru', 'russian': 'ru',
            'arabe': 'ar', 'arabic': 'ar'
        }

        source_lang = None
        target_lang = None

        # Chercher les patterns de traduction
        patterns = [
            r'traduir?\s+(?:du\s+)?(\w+)\s+(?:en|vers|to|in)\s+(\w+)',
            r'translate\s+(?:from\s+)?(\w+)\s+(?:to|in)\s+(\w+)',
            r'(\w+)\s+to\s+(\w+)',
            r'en\s+(\w+)',
            r'vers\s+(?:le\s+)?(\w+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                lang1, lang2 = match.groups()
                if lang1 in language_map and lang2 in language_map:
                    source_lang = language_map[lang1]
                    target_lang = language_map[lang2]
                    break
                elif lang2 in language_map:
                    target_lang = language_map[lang2]
                    # La source sera détectée automatiquement

        # Si seulement la langue cible est trouvée
        if target_lang and not source_lang:
            source_lang = 'auto'  # Détection automatique

        return source_lang, target_lang

    def _extract_text_to_translate(self, query: str, context: Dict[str, Any] = None) -> str:
        """Extrait le texte à traduire de la requête"""

        # Vérifier le contexte d'abord
        if context and 'text_to_translate' in context:
            return context['text_to_translate']

        # Chercher du texte entre guillemets
        quoted_text = re.findall(r'"([^"]*)"', query)
        if quoted_text:
            return quoted_text[0]

        # Chercher du texte après les mots-clés de traduction
        translation_keywords = [
            r'traduire\s+(.+)',
            r'traduis\s+(.+)',
            r'translate\s+(.+)',
            r'"(.+)"'
        ]

        for pattern in translation_keywords:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Nettoyer les mots de langue restants
                text = re.sub(r'\s+(en|vers|to|in|du|from)\s+\w+', '', text, flags=re.IGNORECASE)
                return text.strip()

        return ""

    def _detect_language(self, text: str) -> str:
        """Détecte la langue du texte (simplifié)"""
        # Comptage de mots caractéristiques
        french_words = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'à', 'un', 'une']
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of']

        text_lower = text.lower()

        french_count = sum(1 for word in french_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)

        if french_count > english_count:
            return 'fr'
        else:
            return 'en'

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Traduit le texte (simulation - en production utiliser une vraie API)"""

        # Dictionnaire de traductions simples pour démonstration
        translations = {
            ('fr', 'en'): {
                'bonjour': 'hello',
                'au revoir': 'goodbye',
                'merci': 'thank you',
                'comment allez-vous': 'how are you',
                'je m\'appelle': 'my name is'
            },
            ('en', 'fr'): {
                'hello': 'bonjour',
                'goodbye': 'au revoir',
                'thank you': 'merci',
                'how are you': 'comment allez-vous',
                'my name is': 'je m\'appelle'
            }
        }

        # Normaliser les langues
        if source_lang == 'auto':
            source_lang = self._detect_language(text)

        key = (source_lang, target_lang)
        if key in translations:
            text_lower = text.lower()
            for source_word, target_word in translations[key].items():
                text_lower = text_lower.replace(source_word, target_word)
            return text_lower.capitalize()
        else:
            # Traduction simulée pour les langues non supportées
            return f"[Traduction simulée de {source_lang} vers {target_lang}] {text}"

    def _get_supported_languages(self) -> Dict[str, str]:
        """Retourne les langues supportées"""
        return {
            'fr': 'Français',
            'en': 'English',
            'es': 'Español',
            'de': 'Deutsch',
            'it': 'Italiano',
            'zh': '中文',
            'ja': '日本語',
            'ru': 'Русский',
            'ar': 'العربية'
        }