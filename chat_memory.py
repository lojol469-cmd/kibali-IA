"""
🧠 Système de Mémoire Vectorielle pour l'Historique des Conversations
Permet à Kibali de se souvenir des conversations passées et d'utiliser ce contexte
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Chemin de la base vectorielle des conversations
CHAT_VECTORDB_PATH = os.path.join(
    os.path.dirname(__file__), 
    "kibali_data", 
    "chat_vectordb"
)

def get_embedding_model():
    """Obtient le modèle d'embeddings pour la mémoire"""
    import torch
    import os
    
    # Solution simple: utiliser directement SentenceTransformer avec un wrapper minimal
    try:
        from sentence_transformers import SentenceTransformer
        
        # Charger directement avec SentenceTransformer
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device='cpu'
        )
        
        # Créer un wrapper simple qui implémente embed_documents et embed_query
        class SimpleEmbeddings:
            def __init__(self, model):
                self.client = model
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            def embed_documents(self, texts):
                """Encode une liste de documents"""
                return self.client.encode(texts, normalize_embeddings=True).tolist()
            
            def embed_query(self, text):
                """Encode une requête unique"""
                return self.client.encode([text], normalize_embeddings=True)[0].tolist()
            
            def __call__(self, text):
                """Permet d'appeler l'objet directement comme une fonction"""
                if isinstance(text, list):
                    return self.embed_documents(text)
                else:
                    return self.embed_query(text)
        
        return SimpleEmbeddings(model)
        
    except Exception as e:
        print(f"⚠️ Erreur chargement embeddings: {e}")
        return None

def load_chat_vectordb() -> Tuple[Optional[object], str]:
    """
    Charge la base vectorielle pour l'historique des conversations
    
    Returns:
        Tuple (vectordb, message)
    """
    if not os.path.exists(CHAT_VECTORDB_PATH):
        return None, "⚠️ Aucune mémoire de conversation trouvée (sera créée automatiquement)"
    
    try:
        embedding_model = get_embedding_model()
        
        if embedding_model is None:
            return None, "⚠️ Impossible de charger le modèle d'embeddings - mémoire désactivée"
        
        chat_vectordb = FAISS.load_local(
            CHAT_VECTORDB_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        return chat_vectordb, "✅ Mémoire de conversation chargée"
    except Exception as e:
        return None, f"⚠️ Mémoire désactivée: {e}"

def add_to_chat_memory(user_msg: str, ai_msg: str, chat_vectordb: Optional[object]) -> object:
    """
    Ajoute un échange utilisateur-IA à la mémoire vectorielle
    
    Args:
        user_msg: Message de l'utilisateur
        ai_msg: Réponse de l'IA
        chat_vectordb: Base vectorielle existante (ou None)
    
    Returns:
        Base vectorielle mise à jour
    """
    # Créer la base si elle n'existe pas
    if chat_vectordb is None:
        embedding_model = get_embedding_model()
        chat_vectordb = FAISS.from_texts(["Init"], embedding_model)
        os.makedirs(CHAT_VECTORDB_PATH, exist_ok=True)
    
    # Créer un document représentant l'échange
    exchange = f"👤 Utilisateur: {user_msg}\n\n🤖 Kibali: {ai_msg}"
    doc = Document(
        page_content=exchange,
        metadata={
            "type": "chat_exchange",
            "timestamp": time.time(),
            "user_query": user_msg[:100],  # Première partie pour recherche
        }
    )
    
    # Ajouter à la base
    chat_vectordb.add_documents([doc])
    
    # Sauvegarder
    try:
        chat_vectordb.save_local(CHAT_VECTORDB_PATH)
    except Exception as e:
        print(f"⚠️ Erreur sauvegarde mémoire: {e}")
    
    return chat_vectordb

def search_chat_memory(query: str, chat_vectordb: Optional[object], k: int = 3) -> List[Document]:
    """
    Recherche dans l'historique des conversations pour trouver du contexte pertinent
    
    Args:
        query: Question/requête de recherche
        chat_vectordb: Base vectorielle des conversations
        k: Nombre de résultats à retourner
    
    Returns:
        Liste de documents pertinents de l'historique
    """
    if not chat_vectordb:
        return []
    
    try:
        results = chat_vectordb.similarity_search(query, k=k)
        return results
    except Exception as e:
        print(f"❌ Erreur recherche mémoire: {e}")
        return []

def get_conversation_context(query: str, chat_vectordb: Optional[object], max_results: int = 2) -> str:
    """
    Récupère le contexte pertinent des conversations passées
    
    Args:
        query: Question actuelle
        chat_vectordb: Base vectorielle
        max_results: Nombre maximum de conversations passées à inclure
    
    Returns:
        Texte du contexte formaté
    """
    if not chat_vectordb:
        return ""
    
    relevant_memories = search_chat_memory(query, chat_vectordb, k=max_results)
    
    if not relevant_memories:
        return ""
    
    context = "📚 **Contexte des conversations précédentes:**\n\n"
    for i, mem in enumerate(relevant_memories, 1):
        context += f"**Souvenir {i}:**\n{mem.page_content[:300]}...\n\n"
    
    return context

def clear_chat_memory() -> str:
    """
    Efface complètement la mémoire des conversations
    
    Returns:
        Message de statut
    """
    import shutil
    
    try:
        if os.path.exists(CHAT_VECTORDB_PATH):
            shutil.rmtree(CHAT_VECTORDB_PATH)
            return "✅ Mémoire des conversations effacée"
        else:
            return "⚠️ Aucune mémoire à effacer"
    except Exception as e:
        return f"❌ Erreur effacement: {e}"

def get_memory_stats(chat_vectordb: Optional[object]) -> dict:
    """
    Obtient des statistiques sur la mémoire des conversations
    
    Returns:
        Dictionnaire avec les stats
    """
    if not chat_vectordb:
        return {
            "count": 0,
            "size": "0 MB",
            "status": "Non initialisée"
        }
    
    try:
        # Compter le nombre de documents
        total_docs = chat_vectordb.index.ntotal if hasattr(chat_vectordb, 'index') else 0
        
        # Taille sur disque
        size_mb = 0
        if os.path.exists(CHAT_VECTORDB_PATH):
            for root, dirs, files in os.walk(CHAT_VECTORDB_PATH):
                size_mb += sum(os.path.getsize(os.path.join(root, f)) for f in files)
            size_mb = size_mb / (1024 * 1024)  # Convertir en MB
        
        return {
            "count": total_docs,
            "size": f"{size_mb:.2f} MB",
            "status": "✅ Active"
        }
    except Exception as e:
        return {
            "count": 0,
            "size": "Erreur",
            "status": f"❌ {str(e)}"
        }
