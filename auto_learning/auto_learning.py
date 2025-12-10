#!/usr/bin/env python3
"""
Système d'apprentissage automatique pour Kibali-IA
Apprend des conversations, documents PDF et fichiers binaires
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
import hashlib

class AutoLearning:
    """
    Système d'apprentissage automatique qui analyse et mémorise:
    - Les patterns de conversations (questions fréquentes, réponses efficaces)
    - Le contenu des documents PDF (concepts clés, relations)
    - Les métadonnées des fichiers binaires (types, usages)
    """
    
    def __init__(self, storage_dir: str = "/home/belikan/kibali-IA/auto_learning_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemins de stockage
        self.conversations_path = self.storage_dir / "conversations.json"
        self.patterns_path = self.storage_dir / "patterns.json"
        self.concepts_path = self.storage_dir / "concepts.json"
        self.file_knowledge_path = self.storage_dir / "file_knowledge.json"
        self.embeddings_path = self.storage_dir / "embeddings.pkl"
        
        # Modèle d'embeddings (même que chat_memory pour cohérence)
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.embedder = None
        
        # Charger les données existantes
        self.conversations = self._load_json(self.conversations_path, [])
        self.patterns = self._load_json(self.patterns_path, {
            "frequent_questions": [],
            "successful_responses": [],
            "tool_usage": {},
            "topic_trends": {}
        })
        self.concepts = self._load_json(self.concepts_path, {})
        self.file_knowledge = self._load_json(self.file_knowledge_path, {})
        
        # Embeddings cache
        self.embeddings_cache = self._load_pickle(self.embeddings_path, {})
        
        # Compteurs pour statistiques
        self.stats = {
            "total_conversations": len(self.conversations),
            "total_concepts": len(self.concepts),
            "total_files_analyzed": len(self.file_knowledge)
        }
    
    def _load_json(self, path: Path, default: Any) -> Any:
        """Charge un fichier JSON ou retourne la valeur par défaut"""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return default
        return default
    
    def _save_json(self, path: Path, data: Any):
        """Sauvegarde des données en JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_pickle(self, path: Path, default: Any) -> Any:
        """Charge un fichier pickle ou retourne la valeur par défaut"""
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                return default
        return default
    
    def _save_pickle(self, path: Path, data: Any):
        """Sauvegarde des données en pickle"""
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Génère l'embedding d'un texte"""
        if not self.embedder or not text:
            return None
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        try:
            embedding = self.embedder.encode(text, convert_to_numpy=True)
            self.embeddings_cache[text_hash] = embedding
            return embedding
        except:
            return None
    
    def learn_from_conversation(self, user_message: str, ai_response: str, 
                               context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Apprend d'une conversation utilisateur-IA
        
        Args:
            user_message: Message de l'utilisateur
            ai_response: Réponse de l'IA
            context: Contexte additionnel (outils utilisés, fichiers, etc.)
        
        Returns:
            Statistiques d'apprentissage
        """
        if not user_message or not ai_response:
            return {"status": "skipped", "reason": "empty_messages"}
        
        timestamp = datetime.now().isoformat()
        
        # Enregistrer la conversation
        conv_entry = {
            "timestamp": timestamp,
            "user_message": user_message[:500],  # Limiter la taille
            "ai_response": ai_response[:500],
            "context": context or {}
        }
        self.conversations.append(conv_entry)
        
        # Garder seulement les 1000 dernières conversations
        if len(self.conversations) > 1000:
            self.conversations = self.conversations[-1000:]
        
        # Analyser les patterns
        user_lower = user_message.lower()
        
        # Questions fréquentes
        user_embedding = self._get_embedding(user_message)
        if user_embedding is not None:
            # Chercher des questions similaires
            similar_found = False
            for fq in self.patterns["frequent_questions"]:
                if "embedding" in fq:
                    similarity = np.dot(user_embedding, fq["embedding"]) / (
                        np.linalg.norm(user_embedding) * np.linalg.norm(fq["embedding"])
                    )
                    if similarity > 0.85:  # Très similaire
                        fq["count"] += 1
                        fq["last_seen"] = timestamp
                        similar_found = True
                        break
            
            if not similar_found:
                self.patterns["frequent_questions"].append({
                    "question": user_message[:200],
                    "embedding": user_embedding.tolist(),
                    "count": 1,
                    "first_seen": timestamp,
                    "last_seen": timestamp
                })
        
        # Réponses réussies (basé sur la longueur et la présence d'infos)
        if len(ai_response) > 100 and not "erreur" in ai_response.lower():
            self.patterns["successful_responses"].append({
                "user_query_type": self._classify_query(user_message),
                "response_length": len(ai_response),
                "timestamp": timestamp,
                "had_context": bool(context)
            })
            
            # Garder seulement les 500 dernières
            if len(self.patterns["successful_responses"]) > 500:
                self.patterns["successful_responses"] = self.patterns["successful_responses"][-500:]
        
        # Usage des outils
        if context and "tools_used" in context:
            for tool in context["tools_used"]:
                if tool not in self.patterns["tool_usage"]:
                    self.patterns["tool_usage"][tool] = {"count": 0, "contexts": []}
                self.patterns["tool_usage"][tool]["count"] += 1
                self.patterns["tool_usage"][tool]["contexts"].append(
                    user_message[:100]
                )
                # Garder seulement 50 contextes
                if len(self.patterns["tool_usage"][tool]["contexts"]) > 50:
                    self.patterns["tool_usage"][tool]["contexts"] = \
                        self.patterns["tool_usage"][tool]["contexts"][-50:]
        
        # Tendances de sujets
        topic = self._extract_topic(user_message)
        if topic:
            if topic not in self.patterns["topic_trends"]:
                self.patterns["topic_trends"][topic] = []
            self.patterns["topic_trends"][topic].append(timestamp)
            # Garder seulement 100 mentions par sujet
            if len(self.patterns["topic_trends"][topic]) > 100:
                self.patterns["topic_trends"][topic] = \
                    self.patterns["topic_trends"][topic][-100:]
        
        # Sauvegarder
        self._save_json(self.conversations_path, self.conversations)
        self._save_json(self.patterns_path, self.patterns)
        self._save_pickle(self.embeddings_path, self.embeddings_cache)
        
        self.stats["total_conversations"] = len(self.conversations)
        
        return {
            "status": "learned",
            "conversation_stored": True,
            "patterns_updated": True,
            "total_conversations": self.stats["total_conversations"]
        }
    
    def learn_from_pdf(self, pdf_content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Apprend du contenu d'un document PDF
        
        Args:
            pdf_content: Contenu textuel du PDF
            metadata: Métadonnées (nom, taille, date, etc.)
        
        Returns:
            Statistiques d'apprentissage
        """
        if not pdf_content or len(pdf_content) < 50:
            return {"status": "skipped", "reason": "content_too_short"}
        
        # Extraire les concepts clés
        concepts = self._extract_concepts(pdf_content)
        
        # Hash du document pour l'identifier
        doc_hash = hashlib.md5(pdf_content.encode()).hexdigest()
        
        # Stocker les concepts
        for concept in concepts:
            if concept not in self.concepts:
                self.concepts[concept] = {
                    "count": 0,
                    "documents": [],
                    "contexts": []
                }
            
            self.concepts[concept]["count"] += 1
            if doc_hash not in self.concepts[concept]["documents"]:
                self.concepts[concept]["documents"].append(doc_hash)
            
            # Extraire un contexte autour du concept
            context = self._extract_context(pdf_content, concept)
            if context:
                self.concepts[concept]["contexts"].append(context)
                # Garder seulement 10 contextes par concept
                if len(self.concepts[concept]["contexts"]) > 10:
                    self.concepts[concept]["contexts"] = \
                        self.concepts[concept]["contexts"][-10:]
        
        # Sauvegarder
        self._save_json(self.concepts_path, self.concepts)
        self.stats["total_concepts"] = len(self.concepts)
        
        return {
            "status": "learned",
            "concepts_extracted": len(concepts),
            "document_hash": doc_hash,
            "total_concepts": self.stats["total_concepts"]
        }
    
    def learn_from_binary(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apprend des métadonnées d'un fichier binaire
        
        Args:
            file_info: Informations sur le fichier (nom, type, taille, etc.)
        
        Returns:
            Statistiques d'apprentissage
        """
        if not file_info or "filename" not in file_info:
            return {"status": "skipped", "reason": "missing_filename"}
        
        filename = file_info["filename"]
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in self.file_knowledge:
            self.file_knowledge[file_ext] = {
                "count": 0,
                "examples": [],
                "sizes": [],
                "uses": []
            }
        
        self.file_knowledge[file_ext]["count"] += 1
        
        # Ajouter exemple
        if filename not in self.file_knowledge[file_ext]["examples"]:
            self.file_knowledge[file_ext]["examples"].append(filename)
            if len(self.file_knowledge[file_ext]["examples"]) > 20:
                self.file_knowledge[file_ext]["examples"] = \
                    self.file_knowledge[file_ext]["examples"][-20:]
        
        # Ajouter taille
        if "size" in file_info:
            self.file_knowledge[file_ext]["sizes"].append(file_info["size"])
            if len(self.file_knowledge[file_ext]["sizes"]) > 100:
                self.file_knowledge[file_ext]["sizes"] = \
                    self.file_knowledge[file_ext]["sizes"][-100:]
        
        # Ajouter usage si disponible
        if "use" in file_info:
            self.file_knowledge[file_ext]["uses"].append(file_info["use"])
            if len(self.file_knowledge[file_ext]["uses"]) > 50:
                self.file_knowledge[file_ext]["uses"] = \
                    self.file_knowledge[file_ext]["uses"][-50:]
        
        # Sauvegarder
        self._save_json(self.file_knowledge_path, self.file_knowledge)
        self.stats["total_files_analyzed"] = sum(
            fk["count"] for fk in self.file_knowledge.values()
        )
        
        return {
            "status": "learned",
            "file_type": file_ext,
            "total_files": self.stats["total_files_analyzed"]
        }
    
    def get_suggestions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Obtient des suggestions basées sur l'apprentissage
        
        Args:
            query: Question ou requête de l'utilisateur
            top_k: Nombre de suggestions à retourner
        
        Returns:
            Liste de suggestions avec scores
        """
        suggestions = []
        
        if not query:
            return suggestions
        
        query_embedding = self._get_embedding(query)
        
        # Suggestions basées sur les questions fréquentes
        if query_embedding is not None:
            for fq in self.patterns["frequent_questions"]:
                if "embedding" in fq and fq["count"] >= 2:
                    similarity = np.dot(query_embedding, fq["embedding"]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(fq["embedding"])
                    )
                    if similarity > 0.7:
                        suggestions.append({
                            "type": "similar_question",
                            "question": fq["question"],
                            "frequency": fq["count"],
                            "score": float(similarity)
                        })
        
        # Suggestions basées sur les concepts
        query_lower = query.lower()
        for concept, info in self.concepts.items():
            if concept.lower() in query_lower:
                suggestions.append({
                    "type": "concept",
                    "concept": concept,
                    "frequency": info["count"],
                    "contexts": info["contexts"][:3],
                    "score": 0.8
                })
        
        # Suggestions d'outils basées sur l'usage
        query_type = self._classify_query(query)
        for tool, info in self.patterns["tool_usage"].items():
            if info["count"] >= 3:
                # Vérifier si le contexte correspond
                for context in info["contexts"][:10]:
                    if any(word in context.lower() for word in query_lower.split()[:5]):
                        suggestions.append({
                            "type": "tool",
                            "tool": tool,
                            "frequency": info["count"],
                            "score": 0.75
                        })
                        break
        
        # Trier par score et retourner top_k
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'apprentissage"""
        # Top 10 questions fréquentes
        top_questions = sorted(
            self.patterns["frequent_questions"],
            key=lambda x: x["count"],
            reverse=True
        )[:10]
        
        # Top 10 concepts
        top_concepts = sorted(
            self.concepts.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:10]
        
        # Top outils utilisés
        top_tools = sorted(
            self.patterns["tool_usage"].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:10]
        
        return {
            "total_conversations": self.stats["total_conversations"],
            "total_concepts": self.stats["total_concepts"],
            "total_files_analyzed": self.stats["total_files_analyzed"],
            "top_questions": [{"q": q["question"][:100], "count": q["count"]} 
                            for q in top_questions],
            "top_concepts": [{"concept": c[0], "count": c[1]["count"]} 
                           for c in top_concepts],
            "top_tools": [{"tool": t[0], "count": t[1]["count"]} 
                        for t in top_tools],
            "unique_topics": len(self.patterns["topic_trends"]),
            "file_types_known": len(self.file_knowledge)
        }
    
    def _classify_query(self, query: str) -> str:
        """Classifie une requête en type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["calcul", "combien", "+"]):
            return "math"
        elif any(word in query_lower for word in ["pdf", "document", "rapport"]):
            return "document"
        elif any(word in query_lower for word in ["image", "photo", "visuel"]):
            return "image"
        elif any(word in query_lower for word in ["code", "fonction", "programmer"]):
            return "code"
        elif any(word in query_lower for word in ["cherche", "trouve", "recherche"]):
            return "search"
        elif any(word in query_lower for word in ["tradui", "translate"]):
            return "translation"
        elif any(word in query_lower for word in ["fichier", "dossier", "organise"]):
            return "file_management"
        else:
            return "general"
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extrait le sujet principal d'un texte"""
        text_lower = text.lower()
        
        # Topics communs
        topics = {
            "programmation": ["code", "fonction", "python", "javascript"],
            "documentation": ["pdf", "document", "rapport", "fichier"],
            "intelligence_artificielle": ["ia", "ai", "modèle", "apprentissage"],
            "images": ["image", "photo", "visuel", "3d"],
            "finance": ["argent", "crypto", "bitcoin", "gagner"],
            "recherche": ["cherche", "trouve", "recherche", "google"],
            "mathématiques": ["calcul", "équation", "math"],
        }
        
        for topic, keywords in topics.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        
        return None
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extrait les concepts clés d'un texte"""
        # Mots importants (noms, concepts techniques)
        words = text.split()
        concepts = []
        
        # Chercher les mots capitalisés (noms propres/concepts)
        for i, word in enumerate(words):
            if len(word) > 3 and word[0].isupper():
                # Vérifier si c'est pas le début d'une phrase
                if i > 0 and not words[i-1].endswith('.'):
                    concepts.append(word.strip('.,;:!?'))
        
        # Chercher les patterns techniques
        technical_patterns = [
            "API", "SDK", "JSON", "XML", "HTTP", "URL", "GPU", "CPU",
            "Python", "JavaScript", "TypeScript", "React", "Vue",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP"
        ]
        
        text_upper = text.upper()
        for pattern in technical_patterns:
            if pattern.upper() in text_upper:
                concepts.append(pattern)
        
        # Retourner concepts uniques
        return list(set(concepts))[:50]  # Max 50 concepts
    
    def _extract_context(self, text: str, concept: str, window: int = 100) -> str:
        """Extrait le contexte autour d'un concept"""
        text_lower = text.lower()
        concept_lower = concept.lower()
        
        pos = text_lower.find(concept_lower)
        if pos == -1:
            return ""
        
        start = max(0, pos - window)
        end = min(len(text), pos + len(concept) + window)
        
        return text[start:end].strip()


# Instance globale
_auto_learning_instance = None

def get_auto_learning() -> AutoLearning:
    """Retourne l'instance globale d'AutoLearning"""
    global _auto_learning_instance
    if _auto_learning_instance is None:
        _auto_learning_instance = AutoLearning()
    return _auto_learning_instance


if __name__ == "__main__":
    # Test du système
    al = get_auto_learning()
    
    # Test conversation
    result = al.learn_from_conversation(
        "Comment calculer 15 + 27 ?",
        "Le résultat de 15 + 27 est 42.",
        {"tools_used": ["math_calculator"]}
    )
    print("Test conversation:", result)
    
    # Test PDF
    result = al.learn_from_pdf(
        "Python est un langage de programmation. L'API REST permet la communication HTTP.",
        {"filename": "test.pdf"}
    )
    print("Test PDF:", result)
    
    # Test fichier binaire
    result = al.learn_from_binary({
        "filename": "image.jpg",
        "size": 1024000,
        "use": "photo"
    })
    print("Test fichier:", result)
    
    # Suggestions
    suggestions = al.get_suggestions("comment calculer des nombres")
    print("Suggestions:", suggestions)
    
    # Stats
    stats = al.get_stats()
    print("Stats:", json.dumps(stats, indent=2, ensure_ascii=False))
