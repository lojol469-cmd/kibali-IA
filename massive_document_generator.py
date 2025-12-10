"""
🚀 SYSTÈME DE GÉNÉRATION MASSIVE DE DOCUMENTS PAR RELAIS D'IA
===============================================================

Génère des documents de 100 à 500 pages en combinant plusieurs IA
via un système de relais intelligent avec gestion de cache et mémoire.

CAPACITÉS:
- Génération jusqu'à 1 milliard de tokens (théorique)
- Relais automatique entre modèles IA
- Cache intelligent et streaming optimisé
- Organisation automatique du contenu
- Export PDF professionnel

TECHNIQUE:
1. Découpage en sections (25-50 pages chacune)
2. Chaque IA génère une section complète
3. Passage de relais avec contexte résumé
4. Cache des sections générées
5. Assemblage final en PDF
"""

import streamlit as st
from typing import List, Dict, Any, Generator
import time
from datetime import datetime
import json
from pathlib import Path
import hashlib

class MassiveDocumentGenerator:
    """Générateur de documents massifs par relais d'IA"""
    
    def __init__(self):
        self.models_available = {
            'qwen2.5_14b': {'max_tokens': 8192, 'speed': 'medium', 'quality': 'high'},
            'qwen2.5_7b': {'max_tokens': 8192, 'speed': 'fast', 'quality': 'medium'},
            'qwen2.5_32b': {'max_tokens': 8192, 'speed': 'slow', 'quality': 'very_high'},
            'llama3.2_11b': {'max_tokens': 4096, 'speed': 'medium', 'quality': 'high'},
            'llama3.1_8b': {'max_tokens': 4096, 'speed': 'fast', 'quality': 'medium'},
            'mistral_nemo_12b': {'max_tokens': 8192, 'speed': 'medium', 'quality': 'high'},
            'mistral_7b': {'max_tokens': 4096, 'speed': 'fast', 'quality': 'medium'}
        }
        
        self.cache_dir = Path.home() / ".kibali_cache" / "massive_docs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_capacity(self, target_pages: int = 100) -> Dict[str, Any]:
        """
        Calcule la capacité théorique de génération
        
        Args:
            target_pages: Nombre de pages cibles
            
        Returns:
            Analyse de faisabilité
        """
        # Estimations moyennes
        words_per_page = 500
        tokens_per_word = 1.3  # Moyenne pour le français
        total_tokens_needed = target_pages * words_per_page * tokens_per_word
        
        # Avec relais : chaque modèle génère ~6000 tokens max par section
        tokens_per_section = 6000
        sections_needed = int(total_tokens_needed / tokens_per_section) + 1
        
        # Nombre de relais nécessaires (on utilise 7 modèles en rotation)
        relays_per_model = sections_needed // len(self.models_available)
        
        # Temps estimé (40 tokens/sec moyenne)
        tokens_per_second = 40
        total_time_seconds = total_tokens_needed / tokens_per_second
        
        return {
            'target_pages': target_pages,
            'total_tokens_needed': int(total_tokens_needed),
            'sections_needed': sections_needed,
            'relays_per_model': relays_per_model,
            'estimated_time_minutes': int(total_time_seconds / 60),
            'estimated_time_hours': round(total_time_seconds / 3600, 2),
            'feasible': total_tokens_needed <= 1_000_000_000,  # 1 milliard de tokens
            'strategy': 'relay' if sections_needed > 10 else 'single_model',
            'cache_size_mb': int(total_tokens_needed * 4 / 1024 / 1024)  # 4 bytes par token
        }
    
    def create_document_plan(self, topic: str, pages: int, structure: str = "auto") -> Dict[str, Any]:
        """
        Crée un plan structuré pour le document
        
        Args:
            topic: Sujet du document
            pages: Nombre de pages cibles
            structure: Type de structure (auto, academic, report, manual)
            
        Returns:
            Plan détaillé avec sections
        """
        sections = []
        pages_per_section = 25  # Chaque section = 25 pages max
        num_sections = (pages // pages_per_section) + 1
        
        if structure == "academic":
            # Structure académique
            base_structure = [
                {"title": "Résumé exécutif", "pages": 2, "model": "qwen2.5_14b"},
                {"title": "Introduction", "pages": 5, "model": "mistral_nemo_12b"},
                {"title": "Revue de littérature", "pages": 20, "model": "qwen2.5_32b"},
                {"title": "Méthodologie", "pages": 15, "model": "llama3.2_11b"},
                {"title": "Résultats", "pages": 30, "model": "qwen2.5_14b"},
                {"title": "Discussion", "pages": 20, "model": "mistral_nemo_12b"},
                {"title": "Conclusion", "pages": 5, "model": "llama3.1_8b"},
                {"title": "Références", "pages": 3, "model": "qwen2.5_7b"}
            ]
        elif structure == "report":
            # Rapport professionnel
            base_structure = [
                {"title": "Sommaire exécutif", "pages": 3, "model": "qwen2.5_14b"},
                {"title": "Contexte et enjeux", "pages": 10, "model": "mistral_nemo_12b"},
                {"title": "Analyse de situation", "pages": 25, "model": "qwen2.5_32b"},
                {"title": "Recommandations", "pages": 20, "model": "llama3.2_11b"},
                {"title": "Plan d'action", "pages": 15, "model": "qwen2.5_14b"},
                {"title": "Annexes", "pages": pages - 73, "model": "mistral_7b"}
            ]
        elif structure == "manual":
            # Manuel technique
            base_structure = [
                {"title": "Préface", "pages": 2, "model": "qwen2.5_14b"},
                {"title": "Table des matières", "pages": 3, "model": "mistral_7b"},
                {"title": "Introduction générale", "pages": 10, "model": "mistral_nemo_12b"}
            ]
            # Ajouter des chapitres
            remaining_pages = pages - 15
            num_chapters = remaining_pages // 20
            for i in range(num_chapters):
                base_structure.append({
                    "title": f"Chapitre {i+1}",
                    "pages": 20,
                    "model": list(self.models_available.keys())[i % len(self.models_available)]
                })
        else:
            # Structure automatique
            pages_intro = max(5, int(pages * 0.05))
            pages_main = pages - pages_intro - 5
            pages_per_chapter = 25
            num_chapters = pages_main // pages_per_chapter
            
            base_structure = [
                {"title": "Introduction", "pages": pages_intro, "model": "qwen2.5_14b"}
            ]
            
            for i in range(num_chapters):
                base_structure.append({
                    "title": f"Partie {i+1}",
                    "pages": pages_per_chapter,
                    "model": list(self.models_available.keys())[i % len(self.models_available)]
                })
            
            base_structure.append({"title": "Conclusion", "pages": 5, "model": "mistral_nemo_12b"})
        
        return {
            "topic": topic,
            "total_pages": pages,
            "structure": structure,
            "sections": base_structure,
            "generation_strategy": "relay",
            "created_at": datetime.now().isoformat()
        }
    
    def generate_section_prompt(self, section: Dict, previous_context: str, full_plan: Dict) -> str:
        """
        Génère un prompt optimisé pour une section
        
        Args:
            section: Info sur la section à générer
            previous_context: Résumé des sections précédentes
            full_plan: Plan complet du document
            
        Returns:
            Prompt structuré
        """
        prompt = f"""📚 GÉNÉRATION DE DOCUMENT MASSIF - SECTION: {section['title']}
{'='*80}

🎯 DOCUMENT GLOBAL:
   • Sujet: {full_plan['topic']}
   • Pages totales: {full_plan['total_pages']}
   • Structure: {full_plan['structure']}

📖 SECTION ACTUELLE:
   • Titre: {section['title']}
   • Pages à générer: {section['pages']} pages
   • Position: Section {section.get('index', 0) + 1}/{len(full_plan['sections'])}

"""
        
        if previous_context:
            prompt += f"""📋 CONTEXTE DES SECTIONS PRÉCÉDENTES:
{previous_context}

⚠️ CONTINUITÉ: Assure une transition fluide depuis les sections précédentes.

"""
        
        prompt += f"""✍️ INSTRUCTIONS DE GÉNÉRATION:

1. **VOLUME**: Génère EXACTEMENT {section['pages'] * 500} mots (~{section['pages']} pages)
2. **STRUCTURE**: 
   - Introduction à la section (1 paragraphe)
   - Contenu principal détaillé (80% du volume)
   - Transition vers la section suivante (1 paragraphe)

3. **QUALITÉ**:
   - Contenu riche et détaillé
   - Arguments sourcés et exemples concrets
   - Style professionnel et académique
   - Pas de répétitions

4. **FORMAT**:
   - Utilise des titres ##, ### pour structurer
   - Inclus des listes à puces
   - Ajoute des tableaux si pertinent
   - Citations et références

🚀 COMMENCE LA GÉNÉRATION MAINTENANT:
"""
        
        return prompt
    
    def generate_section(self, section: Dict, previous_context: str, 
                        full_plan: Dict, client, model: str) -> Generator[str, None, None]:
        """
        Génère une section complète avec streaming
        
        Yields:
            Chunks de texte générés
        """
        prompt = self.generate_section_prompt(section, previous_context, full_plan)
        
        messages = [{"role": "user", "content": prompt}]
        
        # Génération avec streaming
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=section['pages'] * 650,  # ~650 tokens par page
            temperature=0.7,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        
        # Sauvegarder dans le cache
        self.cache_section(section, full_response, full_plan)
    
    def cache_section(self, section: Dict, content: str, plan: Dict):
        """Sauvegarde une section dans le cache"""
        cache_key = hashlib.md5(
            f"{plan['topic']}_{section['title']}_{section.get('index', 0)}".encode()
        ).hexdigest()
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'section': section,
                'content': content,
                'plan_topic': plan['topic'],
                'timestamp': datetime.now().isoformat(),
                'word_count': len(content.split())
            }, f, ensure_ascii=False, indent=2)
    
    def get_cached_section(self, section: Dict, plan: Dict) -> str:
        """Récupère une section du cache si disponible"""
        cache_key = hashlib.md5(
            f"{plan['topic']}_{section['title']}_{section.get('index', 0)}".encode()
        ).hexdigest()
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['content']
        return None
    
    def summarize_for_context(self, text: str, max_words: int = 200) -> str:
        """
        Résume un texte pour le passer en contexte à la section suivante
        
        Args:
            text: Texte à résumer
            max_words: Nombre max de mots dans le résumé
            
        Returns:
            Résumé concis
        """
        # Extraction des points clés (titres, premières phrases, etc.)
        lines = text.split('\n')
        key_points = []
        
        for line in lines:
            if line.startswith('#') or line.startswith('**'):
                key_points.append(line.strip('#').strip('*').strip())
        
        # Si pas assez de points clés, prendre les premières phrases
        if len(key_points) < 3:
            sentences = text.split('.')
            key_points = [s.strip() for s in sentences[:5] if len(s.strip()) > 20]
        
        summary = ' • '.join(key_points[:10])
        
        # Tronquer si trop long
        words = summary.split()
        if len(words) > max_words:
            summary = ' '.join(words[:max_words]) + '...'
        
        return summary


# Instance globale
_generator = None

def get_massive_doc_generator() -> MassiveDocumentGenerator:
    """Retourne l'instance globale du générateur"""
    global _generator
    if _generator is None:
        _generator = MassiveDocumentGenerator()
    return _generator
