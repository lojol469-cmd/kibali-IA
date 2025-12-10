#!/usr/bin/env python3
"""
Gestionnaire de connaissances pour Kibali-IA
Centralise et organise toutes les connaissances apprises
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict
import hashlib

class KnowledgeManager:
    """
    Gestionnaire centralisé de connaissances:
    - Stocke et récupère les patterns appris
    - Gère les poids et scores de pertinence
    - Fournit des recommandations intelligentes
    - Suit l'évolution des connaissances dans le temps
    """
    
    def __init__(self, storage_dir: str = "/home/belikan/kibali-IA/knowledge_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemins de stockage
        self.patterns_db_path = self.storage_dir / "patterns_db.json"
        self.weights_path = self.storage_dir / "weights.json"
        self.recommendations_path = self.storage_dir / "recommendations.json"
        self.history_path = self.storage_dir / "history.json"
        
        # Bases de données
        self.patterns_db = self._load_json(self.patterns_db_path, {
            "query_patterns": {},
            "response_patterns": {},
            "tool_patterns": {},
            "document_patterns": {}
        })
        
        self.weights = self._load_json(self.weights_path, {})
        self.recommendations = self._load_json(self.recommendations_path, [])
        self.history = self._load_json(self.history_path, [])
        
        # Paramètres d'apprentissage
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        self.min_confidence = 0.3
    
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
    
    def store_pattern(self, pattern_type: str, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stocke un nouveau pattern appris
        
        Args:
            pattern_type: Type de pattern (query, response, tool, document)
            pattern_data: Données du pattern
        
        Returns:
            Résultat du stockage
        """
        if pattern_type not in self.patterns_db:
            return {"status": "error", "reason": f"unknown_pattern_type: {pattern_type}"}
        
        # Générer un ID unique pour le pattern
        pattern_id = hashlib.md5(
            f"{pattern_type}_{json.dumps(pattern_data, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        
        timestamp = datetime.now().isoformat()
        
        # Vérifier si le pattern existe déjà
        if pattern_id in self.patterns_db[pattern_type]:
            # Mettre à jour le pattern existant
            existing = self.patterns_db[pattern_type][pattern_id]
            existing["occurrences"] += 1
            existing["last_seen"] = timestamp
            
            # Mettre à jour les poids avec apprentissage
            if pattern_id in self.weights:
                self.weights[pattern_id] = min(
                    1.0,
                    self.weights[pattern_id] + self.learning_rate
                )
            
            return {
                "status": "updated",
                "pattern_id": pattern_id,
                "occurrences": existing["occurrences"],
                "weight": self.weights.get(pattern_id, 0.5)
            }
        else:
            # Créer un nouveau pattern
            self.patterns_db[pattern_type][pattern_id] = {
                "data": pattern_data,
                "occurrences": 1,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "feedback_score": 0.0
            }
            
            # Initialiser le poids
            self.weights[pattern_id] = 0.5  # Poids initial neutre
            
            # Sauvegarder
            self._save_json(self.patterns_db_path, self.patterns_db)
            self._save_json(self.weights_path, self.weights)
            
            return {
                "status": "created",
                "pattern_id": pattern_id,
                "weight": 0.5
            }
    
    def retrieve_patterns(self, query: str, pattern_type: Optional[str] = None,
                         top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les patterns pertinents pour une requête
        
        Args:
            query: Requête de recherche
            pattern_type: Type de pattern à chercher (None = tous)
            top_k: Nombre maximum de patterns à retourner
        
        Returns:
            Liste de patterns triés par pertinence
        """
        results = []
        query_lower = query.lower()
        
        # Types de patterns à chercher
        types_to_search = [pattern_type] if pattern_type else list(self.patterns_db.keys())
        
        for ptype in types_to_search:
            if ptype not in self.patterns_db:
                continue
            
            for pattern_id, pattern_info in self.patterns_db[ptype].items():
                # Calculer le score de pertinence
                score = self._calculate_relevance_score(
                    query_lower,
                    pattern_info["data"],
                    pattern_id
                )
                
                if score >= self.min_confidence:
                    results.append({
                        "pattern_id": pattern_id,
                        "pattern_type": ptype,
                        "data": pattern_info["data"],
                        "score": score,
                        "occurrences": pattern_info["occurrences"],
                        "weight": self.weights.get(pattern_id, 0.5)
                    })
        
        # Trier par score de pertinence
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def update_weights(self, pattern_id: str, feedback: float):
        """
        Met à jour les poids d'un pattern basé sur le feedback
        
        Args:
            pattern_id: ID du pattern
            feedback: Score de feedback (-1.0 à 1.0)
        """
        if pattern_id not in self.weights:
            self.weights[pattern_id] = 0.5
        
        # Mise à jour avec learning rate
        current_weight = self.weights[pattern_id]
        delta = self.learning_rate * feedback
        new_weight = np.clip(current_weight + delta, 0.0, 1.0)
        
        self.weights[pattern_id] = new_weight
        
        # Sauvegarder
        self._save_json(self.weights_path, self.weights)
        
        # Mettre à jour le feedback_score dans le pattern
        for ptype in self.patterns_db.values():
            if pattern_id in ptype:
                ptype[pattern_id]["feedback_score"] = new_weight
                break
        
        self._save_json(self.patterns_db_path, self.patterns_db)
        
        return new_weight
    
    def get_recommendations(self, context: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Génère des recommandations basées sur le contexte
        
        Args:
            context: Contexte actuel (query, tools_available, etc.)
            top_k: Nombre de recommandations
        
        Returns:
            Liste de recommandations
        """
        recommendations = []
        
        query = context.get("query", "")
        if not query:
            return recommendations
        
        # Récupérer les patterns pertinents
        relevant_patterns = self.retrieve_patterns(query, top_k=20)
        
        # Générer des recommandations
        for pattern in relevant_patterns:
            ptype = pattern["pattern_type"]
            data = pattern["data"]
            
            if ptype == "tool_patterns":
                # Recommander un outil
                if "tool_name" in data:
                    recommendations.append({
                        "type": "tool",
                        "tool": data["tool_name"],
                        "reason": f"Utilisé {pattern['occurrences']} fois pour des requêtes similaires",
                        "confidence": pattern["score"]
                    })
            
            elif ptype == "response_patterns":
                # Recommander un type de réponse
                if "response_type" in data:
                    recommendations.append({
                        "type": "response_style",
                        "style": data["response_type"],
                        "reason": f"Format efficace basé sur {pattern['occurrences']} interactions",
                        "confidence": pattern["score"]
                    })
            
            elif ptype == "document_patterns":
                # Recommander une source
                if "source" in data:
                    recommendations.append({
                        "type": "source",
                        "source": data["source"],
                        "reason": f"Source pertinente trouvée {pattern['occurrences']} fois",
                        "confidence": pattern["score"]
                    })
        
        # Trier par confidence et retourner top_k
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        return recommendations[:top_k]
    
    def add_history_entry(self, entry: Dict[str, Any]):
        """Ajoute une entrée dans l'historique d'apprentissage"""
        entry["timestamp"] = datetime.now().isoformat()
        self.history.append(entry)
        
        # Garder seulement les 1000 dernières entrées
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        self._save_json(self.history_path, self.history)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire de connaissances"""
        total_patterns = sum(len(patterns) for patterns in self.patterns_db.values())
        
        # Patterns les plus utilisés
        all_patterns = []
        for ptype, patterns in self.patterns_db.items():
            for pid, pinfo in patterns.items():
                all_patterns.append({
                    "type": ptype,
                    "id": pid,
                    "occurrences": pinfo["occurrences"],
                    "weight": self.weights.get(pid, 0.5)
                })
        
        all_patterns.sort(key=lambda x: x["occurrences"], reverse=True)
        top_patterns = all_patterns[:10]
        
        # Distribution des poids
        weights_list = list(self.weights.values())
        avg_weight = np.mean(weights_list) if weights_list else 0.5
        
        return {
            "total_patterns": total_patterns,
            "patterns_by_type": {
                ptype: len(patterns) 
                for ptype, patterns in self.patterns_db.items()
            },
            "total_weights": len(self.weights),
            "average_weight": float(avg_weight),
            "top_patterns": top_patterns,
            "history_size": len(self.history),
            "learning_rate": self.learning_rate,
            "min_confidence": self.min_confidence
        }
    
    def _calculate_relevance_score(self, query: str, pattern_data: Dict[str, Any],
                                   pattern_id: str) -> float:
        """Calcule le score de pertinence d'un pattern pour une requête"""
        score = 0.0
        
        # Score basé sur le poids appris
        weight = self.weights.get(pattern_id, 0.5)
        score += weight * 0.4  # 40% du score
        
        # Score basé sur la correspondance textuelle
        text_fields = []
        for key, value in pattern_data.items():
            if isinstance(value, str):
                text_fields.append(value.lower())
        
        text_content = " ".join(text_fields)
        query_words = set(query.split())
        content_words = set(text_content.split())
        
        if query_words and content_words:
            overlap = len(query_words & content_words) / len(query_words)
            score += overlap * 0.6  # 60% du score
        
        return min(1.0, score)
    
    def decay_old_weights(self, days_threshold: int = 30):
        """
        Applique un decay aux poids des patterns anciens non utilisés
        
        Args:
            days_threshold: Nombre de jours avant de commencer le decay
        """
        from datetime import timedelta
        
        now = datetime.now()
        decayed_count = 0
        
        for ptype, patterns in self.patterns_db.items():
            for pattern_id, pattern_info in patterns.items():
                last_seen = datetime.fromisoformat(pattern_info["last_seen"])
                days_since = (now - last_seen).days
                
                if days_since > days_threshold:
                    if pattern_id in self.weights:
                        # Appliquer le decay
                        old_weight = self.weights[pattern_id]
                        new_weight = old_weight * self.decay_factor
                        
                        if new_weight >= self.min_confidence:
                            self.weights[pattern_id] = new_weight
                            decayed_count += 1
                        else:
                            # Supprimer les patterns avec poids trop faible
                            del self.weights[pattern_id]
        
        # Sauvegarder
        self._save_json(self.weights_path, self.weights)
        
        return {
            "decayed_count": decayed_count,
            "remaining_weights": len(self.weights)
        }
    
    def export_knowledge(self, export_path: Optional[Path] = None) -> Path:
        """Exporte toutes les connaissances dans un fichier unique"""
        if export_path is None:
            export_path = self.storage_dir / f"knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_date": datetime.now().isoformat(),
            "patterns_db": self.patterns_db,
            "weights": self.weights,
            "stats": self.get_stats()
        }
        
        self._save_json(export_path, export_data)
        return export_path
    
    def import_knowledge(self, import_path: Path) -> Dict[str, Any]:
        """Importe des connaissances depuis un fichier"""
        if not import_path.exists():
            return {"status": "error", "reason": "file_not_found"}
        
        try:
            import_data = self._load_json(import_path, {})
            
            if "patterns_db" in import_data:
                # Merger avec les patterns existants
                for ptype, patterns in import_data["patterns_db"].items():
                    if ptype not in self.patterns_db:
                        self.patterns_db[ptype] = {}
                    self.patterns_db[ptype].update(patterns)
            
            if "weights" in import_data:
                self.weights.update(import_data["weights"])
            
            # Sauvegarder
            self._save_json(self.patterns_db_path, self.patterns_db)
            self._save_json(self.weights_path, self.weights)
            
            return {
                "status": "success",
                "patterns_imported": sum(len(p) for p in import_data.get("patterns_db", {}).values()),
                "weights_imported": len(import_data.get("weights", {}))
            }
        except Exception as e:
            return {"status": "error", "reason": str(e)}


# Instance globale
_knowledge_manager_instance = None

def get_knowledge_manager() -> KnowledgeManager:
    """Retourne l'instance globale du KnowledgeManager"""
    global _knowledge_manager_instance
    if _knowledge_manager_instance is None:
        _knowledge_manager_instance = KnowledgeManager()
    return _knowledge_manager_instance


if __name__ == "__main__":
    # Test du système
    km = get_knowledge_manager()
    
    # Test stockage de pattern
    result = km.store_pattern("tool_patterns", {
        "tool_name": "math_calculator",
        "query_type": "calculation",
        "example": "calcule 15 + 27"
    })
    print("Test stockage:", result)
    
    # Test récupération
    patterns = km.retrieve_patterns("calcule des nombres")
    print("Patterns trouvés:", len(patterns))
    
    # Test recommandations
    recommendations = km.get_recommendations({
        "query": "comment calculer une équation"
    })
    print("Recommandations:", recommendations)
    
    # Test mise à jour poids
    if patterns:
        pid = patterns[0]["pattern_id"]
        new_weight = km.update_weights(pid, 0.2)
        print(f"Nouveau poids: {new_weight}")
    
    # Stats
    stats = km.get_stats()
    print("Stats:", json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Test export
    export_path = km.export_knowledge()
    print(f"Connaissances exportées vers: {export_path}")
