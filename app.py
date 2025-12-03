# ===============================================
# Installation des dépendances (pour Colab/Jupyter - commenter pour Streamlit)
# ===============================================
# !apt-get install -y osmium-tool > /dev/null
# !pip install -U streamlit huggingface_hub langchain sentence-transformers faiss-cpu pymupdf osmium networkx shapely matplotlib opencv-python-headless open3d ipywidgets pandas scikit-image scikit-learn torch torchvision langchain-community langchain-huggingface requests beautifulsoup4 python-dotenv diffusers accelerate transformers librosa soundfile tavily-python flash-attn --no-build-isolation --quiet
# ===============================================
# Configuration HuggingFace Token depuis .env
# ===============================================
import os
from pathlib import Path
# Charger le token depuis .env
from dotenv import load_dotenv

# Obtenir le dossier du script actuel
script_dir = Path(__file__).parent.absolute()

# Import de la configuration centralisée des modèles
import sys
sys.path.insert(0, str(script_dir / "kibali_data" / "models"))
from MODEL_PATHS import (
    QWEN_MODEL_NAME, QWEN_CACHE_DIR,
    SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMER_CACHE,
    CLIP_MODEL_NAME, CLIP_CACHE_DIR,
    EASYOCR_MODEL_DIR, EASYOCR_LANGUAGES,
    SUMMARIZER_MODEL, SUMMARIZER_CACHE,
    TRANSLATOR_MODEL, TRANSLATOR_CACHE,
    NER_MODEL, NER_CACHE,
    ensure_model_dirs
)

# Initialiser les dossiers de modèles
ensure_model_dirs()

# Chercher le fichier .env dans plusieurs emplacements possibles
env_paths = [
    script_dir / ".env",  # Dans le même dossier que app.py
    Path.cwd() / ".env",  # Dans le dossier courant
    Path.home() / ".env"  # Dans le dossier home
]

token_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Fichier .env trouvé: {env_path}")
        token_loaded = True
        break

if not token_loaded:
    print(f"⚠️ Aucun fichier .env trouvé dans: {[str(p) for p in env_paths]}")
    print(f"Créez un fichier .env dans {script_dir} avec: HF_TOKEN=hf_votre_token")

# Récupérer le token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"❌ HF_TOKEN non trouvé ! Vérifiez votre fichier .env dans {script_dir}")
else:
    print(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")

# Définir la variable d'environnement pour huggingface_hub
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError(f"❌ TAVILY_API_KEY non trouvé ! Vérifiez votre fichier .env dans {script_dir}")
# ===============================================
# Imports
# ===============================================
import math
import fitz # pymupdf
import osmium
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import base64
import json
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from shapely.geometry import Point
import io
from PIL import Image
import cv2
try:
    import open3d as o3d
except ImportError:
    o3d = None
from io import BytesIO
import pandas as pd
from skimage import measure, segmentation
from sklearn.cluster import KMeans
import torch
from torchvision import models, transforms
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import initialize_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import pipeline
except ImportError:
    # Fallback si pipeline n'est pas disponible directement
    pipeline = None
import requests
from bs4 import BeautifulSoup
import time
import shutil
try:
    import easyocr
except ImportError:
    easyocr = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
from diffusers import DiffusionPipeline, AudioLDMPipeline, ShapEPipeline, ShapEImg2ImgPipeline
import imageio
import scipy.io.wavfile as wavfile
from tavily import TavilyClient
# ===============================================
# Import du système de modèle local Qwen
# ===============================================
import torch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from typing import Iterator
# ===============================================
# Import du système d'outils dynamiques
# ===============================================
try:
    from outils import ToolManager
    TOOLS_SYSTEM_AVAILABLE = True
    print("✅ Système d'outils dynamiques chargé")
except ImportError as e:
    print(f"⚠️ Système d'outils non disponible: {e}")
    TOOLS_SYSTEM_AVAILABLE = False
# ===============================================
# Classe ChatModel personnalisée pour LangChain utilisant Qwen2.5-1.5B
# ===============================================
class QwenChatModel(BaseChatModel):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    tools_available: bool = True
   
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.tools_available = True
       
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-1.5b-local-enhanced"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response using tools and analyses."""
        # Extraire le contenu du message utilisateur
        user_message = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                user_message = message.content
                break
       
        # Détecter si l'utilisateur demande une analyse
        needs_analysis = any(keyword in user_message.lower() for keyword in [
            "analyse", "resistivité", "ert", "recherche", "données", "matériaux",
            "couleurs", "graphique", "tableau", "comparaison", "approfondie"
        ])
       
        if needs_analysis and self.tools_available:
            # Utiliser les outils disponibles pour une analyse complète
            try:
                # Recherche web pour informations
                if any(keyword in user_message.lower() for keyword in ["recherche", "informations", "approfondie"]):
                    search_query = user_message.replace("fais maintenant une recherche plus approfondie pour obtenir toutes ces informations précises", "")
                    web_results = web_search_enhanced(search_query + " ERT electrical resistivity geophysics materials")
                   
                # Recherche RAG si disponible
                rag_results = ""
                if st.session_state.get('vectordb'):
                    rag_results = rag_search(user_message, st.session_state.vectordb)
               
                # Génération de données et analyses si demandées
                analysis_results = ""
                if any(keyword in user_message.lower() for keyword in ["tableau", "graphique", "données"]):
                    # Simuler des données ERT pour démonstration
                    import numpy as np
                    sample_data = [0.05, 0.3, 10.0, 50.0, 200.0, 1000.0, 5000.0, 0.0000024, 1000000]
                    analysis_results = resistivity_color_analysis(sample_data)
               
                # Construire la réponse enrichie avec outils
                enhanced_context = f"""
🔍 ANALYSE COMPLÈTE AVEC OUTILS ACTIVÉS:
🌐 RECHERCHE WEB EFFECTUÉE:
{web_results}
📚 RECHERCHE RAG:
{rag_results}
📊 ANALYSE ERT AVANCÉE:
{analysis_results}
CONTEXTE UTILISATEUR: {user_message}
"""
               
                # Générer la réponse avec le contexte enrichi
                enhanced_messages = [
                    {"role": "system", "content": """Tu es un expert en géophysique ERT avec accès à des outils puissants.
                    Tu DOIS utiliser les données fournies pour créer des analyses détaillées, tableaux, graphiques et comparaisons.
                    Réponds toujours avec des données concrètes et des analyses approfondies basées sur les outils utilisés.
                    Ne dis JAMAIS que tu n'as pas accès aux outils - utilise les résultats fournis."""},
                    {"role": "user", "content": enhanced_context}
                ]
            except Exception as e:
                print(f"Erreur outils: {e}")
                enhanced_messages = [
                    {"role": "system", "content": "Tu es un expert en analyse de données ERT."},
                    {"role": "user", "content": user_message}
                ]
        else:
            # Messages standard
            enhanced_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    enhanced_messages.append({"role": "system", "content": message.content})
                elif isinstance(message, HumanMessage):
                    enhanced_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    enhanced_messages.append({"role": "assistant", "content": message.content})
       
        # Génération avec les messages enrichis
        inputs = self.tokenizer.apply_chat_template(
            enhanced_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
       
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
       
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=3000,  # 3000 tokens pour réponses TRÈS détaillées
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.05,  # Éviter répétitions
                pad_token_id=self.tokenizer.eos_token_id
            )
       
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
       
        if stop:
            for stop_token in stop:
                if stop_token in response:
                    response = response.split(stop_token)[0]
                    break
       
        return AIMessage(content=response)
    
    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator:
        """Streaming is not implemented for simplicity."""
        yield self._generate(messages, stop, run_manager, **kwargs)
# ===============================================
# Chargement du modèle LLM local Qwen2.5-1.5B
# ===============================================
@st.cache_resource
def load_local_llm_model():
    """Charge le modèle Qwen2.5-1.5B depuis kibali_data/models"""
    
    print(f"🚀 Chargement de {QWEN_MODEL_NAME}...")
    print(f"📁 Cache: {QWEN_CACHE_DIR}")
    
    # Récupérer le token depuis les variables d'environnement
    hf_token = os.getenv("HF_TOKEN", "")
    
    # Détection GPU optimisée
    device = 'cpu'
    gpu_info = ""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)"
        print(f"🚀 GPU détecté: {gpu_info}")
    else:
        print("🖥️ Utilisation du CPU")
   
    # Charger tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_NAME,
        trust_remote_code=True,
        token=hf_token if hf_token else None,
        use_fast=True,
        resume_download=True,
        cache_dir=str(QWEN_CACHE_DIR)
    )
    
    # Corriger le problème du pad_token = eos_token pour éviter les warnings
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
   
    # Configuration optimisée selon le device
    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=hf_token if hf_token else None,
            low_cpu_mem_usage=True,
            resume_download=True,
            cache_dir=str(QWEN_CACHE_DIR)
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            token=hf_token if hf_token else None,
            low_cpu_mem_usage=True,
            resume_download=True,
            cache_dir=str(QWEN_CACHE_DIR)
        ).to(device)
   
    return tokenizer, model, device, gpu_info
# ===============================================
# Configuration - CHEMINS UNIFIÉS
# ===============================================
# Modèles 100% Open Source avec Licence Commerciale (Apache 2.0 / MIT)
# ===============================================
WORKING_MODELS = {
    # Qwen 2.5 - Apache 2.0 License - Usage commercial autorisé
    "Qwen2.5 14B (Apache 2.0 - Commercial OK)": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5 7B (Apache 2.0 - Rapide)": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5 32B (Apache 2.0 - Très puissant)": "Qwen/Qwen2.5-32B-Instruct",
    
    # Llama 3.2 - Apache 2.0 License (Meta autorisation commerciale)
    "Llama 3.2 11B Vision (Apache 2.0 - Multimodal)": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Llama 3.1 8B (Apache 2.0 - Optimisé)": "meta-llama/Llama-3.1-8B-Instruct",
    
    # Mistral - Apache 2.0 License - Usage commercial libre
    "Mistral Nemo 12B (Apache 2.0 - Français)": "mistralai/Mistral-Nemo-Instruct-2407",
    "Mistral 7B v0.3 (Apache 2.0 - Rapide)": "mistralai/Mistral-7B-Instruct-v0.3",
    
    # Phi-3 Medium - MIT License - Microsoft open source commercial
    "Phi-3 Medium 14B (MIT - Commercial)": "microsoft/Phi-3-medium-4k-instruct",
    
    # Gemma 2 - Gemma License (usage commercial autorisé par Google)
    "Gemma 2 9B (Gemma License - Commercial OK)": "google/gemma-2-9b-it",
}
# TOUS LES FICHIERS DANS LE MÊME DOSSIER CHATBOT
CHATBOT_DIR = os.path.join(os.getcwd(), "kibali_data")
VECTORDB_PATH = os.path.join(CHATBOT_DIR, "vectordb")
PDFS_PATH = os.path.join(CHATBOT_DIR, "pdfs")
GRAPHS_PATH = os.path.join(CHATBOT_DIR, "graphs") # Graphes OSM ici aussi
MAPS_PATH = os.path.join(CHATBOT_DIR, "maps")
METADATA_PATH = os.path.join(CHATBOT_DIR, "metadata.json")
TRAJECTORIES_PATH = os.path.join(CHATBOT_DIR, "trajectories.json")
WEB_CACHE_PATH = os.path.join(CHATBOT_DIR, "web_cache.json")
GENERATED_PATH = os.path.join(CHATBOT_DIR, "generated")
# ===============================================
# Test de connexion HuggingFace
# ===============================================
def test_hf_connection():
    """Teste la connexion à HuggingFace"""
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=HF_TOKEN)
        print(f"✅ Connexion HuggingFace réussie: {user_info.get('name', 'Utilisateur')}")
        return True
    except Exception as e:
        print(f"❌ Erreur connexion HuggingFace: {e}")
        return False
# Tester la connexion au démarrage
if not test_hf_connection():
    print("⚠️ Problème de connexion HuggingFace, vérifiez votre token")
# ===============================================
# Fonctions utilitaires
# ===============================================
def setup_drive():
    """Monte Google Drive et crée les dossiers"""
    print("📁 Configuration des dossiers...")
    os.makedirs(CHATBOT_DIR, exist_ok=True)
    os.makedirs(PDFS_PATH, exist_ok=True)
    os.makedirs(GRAPHS_PATH, exist_ok=True)
    os.makedirs(MAPS_PATH, exist_ok=True)
    os.makedirs(VECTORDB_PATH, exist_ok=True)
    os.makedirs(GENERATED_PATH, exist_ok=True)
    print(f"📁 Dossier principal : {CHATBOT_DIR}")
    return True
def extract_text_from_pdf(pdf_path):
    """Extraire le texte d'un PDF"""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text += f"\n[Page {page_num + 1}]\n{page_text}\n"
        return text
    except Exception as e:
        print(f"❌ Erreur PDF {pdf_path}: {e}")
        return ""
def upload_and_process_pbf(pbf_file):
    """Traitement du fichier PBF uploadé"""
    if pbf_file is None:
        return None, None, "❌ Aucun fichier uploadé"
    pbf_path = pbf_file
    print("⚙️ Lecture du PBF et construction du graphe...")
    handler = RoadPOIHandler()
    handler.apply_file(pbf_path, locations=True)
    G = handler.graph
    pois = handler.pois
    # Sauvegarder dans le dossier chatbot
    graph_name = os.path.basename(pbf_path).replace('.osm.pbf', '_graph.graphml')
    graph_path = os.path.join(GRAPHS_PATH, graph_name)
    nx.write_graphml(G, graph_path)
    # Sauvegarder les POIs
    pois_name = graph_name.replace('_graph.graphml', '_pois.json')
    pois_path = os.path.join(GRAPHS_PATH, pois_name)
    with open(pois_path, 'w', encoding='utf-8') as f:
        json.dump(pois, f, indent=2, ensure_ascii=False)
    print(f"✅ Graphe: {len(G)} nœuds, {G.size()} arêtes")
    print(f"✅ POIs: {len(pois)} points")
    print(f"💾 Sauvegardé: {graph_path}")
    return G, pois, f"✅ Graphe créé: {len(G)} nœuds, {len(pois)} POIs"
def load_existing_graph():
    """Charge un graphe existant"""
    graph_files = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
    if not graph_files:
        return None, None, "❌ Aucun graphe trouvé"
    graph_file = graph_files[0]
    graph_path = os.path.join(GRAPHS_PATH, graph_file)
    pois_path = os.path.join(GRAPHS_PATH, graph_file.replace('_graph.graphml', '_pois.json'))
    try:
        G = nx.read_graphml(graph_path)
        pois = []
        if os.path.exists(pois_path):
            with open(pois_path, 'r', encoding='utf-8') as f:
                pois = json.load(f)
        return G, pois, f"✅ Graphe chargé: {len(G)} nœuds, {len(pois)} POIs"
    except Exception as e:
        return None, None, f"❌ Erreur: {e}"
def process_pdfs():
    """Traiter les PDFs"""
    print("📄 Traitement des PDFs...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    # Charger vectordb existante si elle existe
    if os.path.exists(VECTORDB_PATH):
        vectordb, _ = load_vectordb()
    else:
        vectordb = None
    # Charger métadonnées existantes
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {"processed_files": [], "total_chunks": 0}
    processed_filenames = {p["filename"] for p in metadata["processed_files"]}
    all_documents = []
    pdf_files = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
    if not pdf_files:
        return vectordb, "⚠️ Aucun PDF trouvé"
    new_chunks_count = 0
    new_processed = []
    for pdf_file in pdf_files:
        if pdf_file in processed_filenames:
            print(f" 📖 {pdf_file} déjà traité, sauté.")
            continue
        pdf_path = os.path.join(PDFS_PATH, pdf_file)
        print(f" 📖 Traitement nouveau PDF : {pdf_file}")
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            continue
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_file,
                    "chunk_id": i,
                    "type": "pdf"
                }
            )
            all_documents.append(doc)
        new_processed.append({"filename": pdf_file, "chunks": len(chunks)})
        new_chunks_count += len(chunks)
    # Ajouter les trajets sauvegardés (toujours, car ils peuvent changer)
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
        for traj in trajectories:
            traj_text = f"""Trajet: {traj.get('question', '')}
Départ: {traj.get('start_name', '')}
Arrivée: {traj.get('end_name', '')}
Distance: {traj.get('distance', 0)/1000:.2f} km"""
            doc = Document(
                page_content=traj_text,
                metadata={"source": "trajectories", "type": "trajectory"}
            )
            all_documents.append(doc)
    if all_documents:
        if vectordb is None:
            vectordb = FAISS.from_documents(all_documents, embedding_model)
        else:
            vectordb.add_documents(all_documents)
        vectordb.save_local(VECTORDB_PATH)
    # Mettre à jour métadonnées
    metadata["processed_files"].extend(new_processed)
    metadata["total_chunks"] += new_chunks_count
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return vectordb, f"✅ Base mise à jour : {len(new_processed)} nouveaux PDFs traités, {new_chunks_count} nouveaux chunks (total : {metadata['total_chunks']})"
def load_vectordb():
    """Charger la base vectorielle"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Vérifier si la base existe
        index_file = os.path.join(VECTORDB_PATH, "index.faiss")
        if not os.path.exists(index_file):
            # Créer une base vide si elle n'existe pas
            os.makedirs(VECTORDB_PATH, exist_ok=True)
            
            # Créer un document factice pour initialiser FAISS
            from langchain.schema import Document
            dummy_doc = Document(page_content="Base vectorielle initialisée", metadata={"source": "system"})
            vectordb = FAISS.from_documents([dummy_doc], embedding_model)
            vectordb.save_local(VECTORDB_PATH)
            
            return vectordb, "✅ Base vectorielle créée (vide - ajoutez des PDFs)"
        
        # Charger la base existante
        vectordb = FAISS.load_local(VECTORDB_PATH, embedding_model, allow_dangerous_deserialization=True)
        return vectordb, "✅ Base chargée"
    except Exception as e:
        return None, f"❌ Erreur: {e}"
def save_trajectory(question, response, trajectory_info):
    """Sauvegarde un trajet"""
    trajectories = []
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
    new_trajectory = {
        "question": question,
        "response": response,
        "start_name": trajectory_info.get('start', {}).get('name', ''),
        "end_name": trajectory_info.get('end', {}).get('name', ''),
        "distance": trajectory_info.get('distance', 0)
    }
    trajectories.append(new_trajectory)
    with open(TRAJECTORIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
def upload_pdfs(uploaded_files):
    """Upload des PDFs"""
    if uploaded_files is None:
        return []
    saved_files = []
    for file in uploaded_files:
        filename = os.path.basename(file)
        filepath = os.path.join(PDFS_PATH, filename)
        shutil.copy(file, filepath)
        saved_files.append(filename)
    return saved_files
# ===============================================
# Système de Cache Web Intelligent
# ===============================================
def load_web_cache():
    """Charge le cache web"""
    if os.path.exists(WEB_CACHE_PATH):
        try:
            with open(WEB_CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}
def save_web_cache(cache):
    """Sauvegarde le cache web"""
    try:
        with open(WEB_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erreur sauvegarde cache: {e}")
def get_cache_key(query, source="text"):
    """Génère une clé de cache pour une requête"""
    return f"{source}:{query.lower().strip()}"
def is_cache_expired(cache_entry, max_age_hours=24):
    """Vérifie si l'entrée du cache a expiré"""
    current_time = time.time()
    return (current_time - cache_entry.get('timestamp', 0)) > (max_age_hours * 3600)
def get_cache_stats():
    """Obtient les statistiques du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        total_entries = len(cache)
        expired_count = sum(1 for entry in cache.values() if is_cache_expired(entry))
        valid_count = total_entries - expired_count
        return f"📊 Cache: {total_entries} entrées total, {valid_count} valides, {expired_count} expirées"
    except Exception as e:
        return f"❌ Erreur stats: {e}"

def get_system_status():
    """Retourne le statut complet du système"""
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "directories": {
            "chatbot": os.path.exists(CHATBOT_DIR),
            "pdfs": os.path.exists(PDFS_PATH),
            "graphs": os.path.exists(GRAPHS_PATH),
            "maps": os.path.exists(MAPS_PATH)
        },
        "files": {
            "vectordb": os.path.exists(VECTORDB_PATH),
            "metadata": os.path.exists(METADATA_PATH),
            "trajectories": os.path.exists(TRAJECTORIES_PATH),
            "web_cache": os.path.exists(WEB_CACHE_PATH)
        },
        "counts": {
            "pdfs": len([f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')]) if os.path.exists(PDFS_PATH) else 0,
            "graphs": len([f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')]) if os.path.exists(GRAPHS_PATH) else 0
        },
        "cache_stats": get_cache_stats(),
        "token_configured": bool(HF_TOKEN and len(HF_TOKEN) > 10)
    }
    return status

# ===============================================
# Fonctions RAG et Web Search Améliorées
# ===============================================
def create_client():
    """Créer le client Inference avec gestion d'erreurs améliorée"""
    try:
        client = InferenceClient(token=HF_TOKEN)
        return client
    except Exception as e:
        print(f"❌ Erreur création client: {e}")
        raise e

# ===============================================
# Chargement des modèles de vision locaux
# ===============================================
@st.cache_resource
def load_vision_models():
    """Charge les modèles CLIP depuis le cache local"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Charger CLIP pour analyse sémantique
        print("📦 Chargement du modèle CLIP local...")
        print(f"📁 Cache: {CLIP_CACHE_DIR}")
        clip_model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME,
            cache_dir=str(CLIP_CACHE_DIR)
        ).to(device)
        clip_processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME,
            cache_dir=str(CLIP_CACHE_DIR)
        )
        
        print(f"✅ Modèle CLIP chargé sur {device}")
        
        return {
            'clip_model': clip_model,
            'clip_processor': clip_processor,
            'device': device
        }
    except Exception as e:
        print(f"⚠️ Erreur chargement modèles vision: {e}")
        return None

@st.cache_resource
def load_ocr_reader():
    """Charge le lecteur OCR EasyOCR avec cache centralisé"""
    try:
        if easyocr:
            print("📦 Chargement du modèle OCR EasyOCR...")
            print(f"📁 Cache: {EASYOCR_MODEL_DIR}")
            
            # Créer le dossier s'il n'existe pas
            EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            reader = easyocr.Reader(
                EASYOCR_LANGUAGES, 
                gpu=torch.cuda.is_available(),
                model_storage_directory=str(EASYOCR_MODEL_DIR)
            )
            print("✅ Modèle OCR chargé")
            return reader
        else:
            print("⚠️ EasyOCR non installé")
            return None
    except Exception as e:
        print(f"⚠️ Erreur chargement OCR: {e}")
        return None

def extract_text_from_image(image_path, ocr_reader=None):
    """Extrait le texte d'une image avec OCR"""
    extracted_texts = []
    
    try:
        # Méthode 1: EasyOCR (meilleur pour textes complexes)
        if ocr_reader:
            results = ocr_reader.readtext(image_path)
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Seuil de confiance
                    extracted_texts.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'method': 'EasyOCR'
                    })
        
        # Méthode 2: Tesseract (fallback)
        elif pytesseract:
            try:
                img = Image.open(image_path)
                # Extraction simple
                text = pytesseract.image_to_string(img, lang='fra+eng')
                if text.strip():
                    extracted_texts.append({
                        'text': text.strip(),
                        'confidence': 1.0,
                        'method': 'Tesseract'
                    })
                
                # Extraction avec détails (positions)
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='fra+eng')
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 30:  # Confiance > 30%
                        txt = data['text'][i].strip()
                        if txt:
                            extracted_texts.append({
                                'text': txt,
                                'confidence': int(data['conf'][i]) / 100,
                                'bbox': (data['left'][i], data['top'][i], 
                                        data['width'][i], data['height'][i]),
                                'method': 'Tesseract'
                            })
            except Exception as tess_error:
                print(f"⚠️ Tesseract error: {tess_error}")
        
        return extracted_texts
    
    except Exception as e:
        print(f"❌ Erreur OCR: {e}")
        return []

def organize_extracted_text(extracted_texts):
    """Organise le texte extrait de manière structurée"""
    if not extracted_texts:
        return "Aucun texte détecté dans l'image."
    
    # Grouper par méthode
    by_method = {}
    for item in extracted_texts:
        method = item.get('method', 'Unknown')
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(item)
    
    # Formater la sortie
    output = []
    output.append("📝 **TEXTE EXTRAIT DE L'IMAGE:**\n")
    
    for method, items in by_method.items():
        output.append(f"\n🔍 **Méthode: {method}**")
        output.append(f"   Nombre d'éléments détectés: {len(items)}\n")
        
        # Trier par confiance
        items_sorted = sorted(items, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for idx, item in enumerate(items_sorted[:50], 1):  # Limiter à 50 éléments
            confidence = item.get('confidence', 0)
            text = item.get('text', '').strip()
            
            if text:
                # Formater selon la confiance
                if confidence > 0.8:
                    output.append(f"   {idx}. ✅ [{confidence:.0%}] {text}")
                elif confidence > 0.5:
                    output.append(f"   {idx}. ⚠️  [{confidence:.0%}] {text}")
                else:
                    output.append(f"   {idx}. ❓ [{confidence:.0%}] {text}")
    
    # Texte complet reconstitué
    output.append("\n\n📄 **TEXTE COMPLET RECONSTITUÉ:**\n")
    full_text = " ".join([item['text'] for item in extracted_texts if item.get('text', '').strip()])
    output.append(full_text)
    
    return "\n".join(output)

def analyze_image_with_clip(image_path, vision_models):
    """Analyse une image avec CLIP local"""
    try:
        if not vision_models:
            return None, "Modèles non chargés"
        
        clip_model = vision_models['clip_model']
        clip_processor = vision_models['clip_processor']
        device = vision_models['device']
        
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        
        # Labels pour classification sémantique (géophysique focusé)
        labels = [
            "geological rock formation",
            "mineral sample",
            "geophysical survey equipment",
            "topographic map",
            "seismic data visualization",
            "core sample",
            "field work photography",
            "satellite imagery",
            "landscape terrain",
            "technical diagram",
            "graph or chart",
            "document or report",
            "person or people",
            "building or structure",
            "natural scenery"
        ]
        
        # Préparer inputs
        inputs = clip_processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Top 3 prédictions
        top3_probs, top3_indices = torch.topk(probs[0], 3)
        
        results = []
        for prob, idx in zip(top3_probs, top3_indices):
            results.append({
                'label': labels[idx.item()],
                'confidence': prob.item()
            })
        
        # Générer caption
        main_label = results[0]['label']
        confidence = results[0]['confidence']
        
        caption = f"Image appears to be: {main_label} (confidence: {confidence:.1%})"
        
        return caption, results
        
    except Exception as e:
        return None, str(e)

def rag_search(question, vectordb, k=3):
    """Rechercher dans la base vectorielle"""
    if not vectordb:
        return []
    try:
        return vectordb.similarity_search(question, k=k)
    except Exception as e:
        print(f"❌ Erreur recherche: {e}")
        return []
def enhanced_web_search(query, max_results=5, search_type="text", use_cache=True):
    """
    Recherche web avancée avec cache intelligent et multiple sources
    Args:
        query: Requête de recherche
        max_results: Nombre max de résultats
        search_type: Type de recherche ("text", "news", "both")
        use_cache: Utiliser le cache
    Returns:
        Liste de résultats enrichis
    """
    cache = load_web_cache() if use_cache else {}
    results = []
    try:
        # Recherche texte
        if search_type in ["text", "both"]:
            cache_key = get_cache_key(query, "text")
            if cache_key in cache and not is_cache_expired(cache[cache_key]):
                print(f"📋 Utilisation cache pour: {query}")
                text_results = cache[cache_key]['results']
            else:
                print(f"🔍 Recherche web pour: {query}")
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                text_results = []
                try:
                    raw_results = tavily.search(query, max_results=max_results, search_depth="advanced", topic="general")
                    for r in raw_results.get('results', []):
                        text_results.append({
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'href': r.get('url', ''),
                            'source_type': 'web_search'
                        })
                    # Sauvegarder en cache
                    cache[cache_key] = {
                        'results': text_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    print(f"Erreur recherche texte: {e}")
                    text_results = []
            results.extend(text_results)
        # Recherche actualités
        if search_type in ["news", "both"]:
            cache_key = get_cache_key(query, "news")
            if cache_key in cache and not is_cache_expired(cache[cache_key], max_age_hours=6):
                news_results = cache[cache_key]['results']
            else:
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                news_results = []
                try:
                    raw_news = tavily.search(query, max_results=max_results//2 if search_type == "both" else max_results, search_depth="advanced", topic="news")
                    for r in raw_news.get('results', []):
                        news_results.append({
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'url': r.get('url', ''),
                            'date': r.get('published_date', ''),
                            'source': r.get('source', ''),
                            'source_type': 'news'
                        })
                    # Sauvegarder en cache (6h pour les news)
                    cache[cache_key] = {
                        'results': news_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    print(f"Erreur recherche news: {e}")
                    news_results = []
            results.extend(news_results)
    except Exception as e:
        print(f"❌ Erreur recherche web globale: {e}")
        results = [{'title': 'Erreur de recherche', 'body': f'Erreur: {e}', 'source_type': 'error'}]
    return results
def smart_content_extraction(url, max_length=1000):
    """
    Extraction intelligente du contenu d'une page web
    Args:
        url: URL à scraper
        max_length: Longueur max du contenu
    Returns:
        Contenu extrait et nettoyé
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Supprimer les éléments non pertinents
        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
            element.decompose()
        # Extraire le texte principal
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        # Nettoyer et tronquer
        text = ' '.join(text.split()) # Normaliser les espaces
        return text[:max_length] + ('...' if len(text) > max_length else '')
    except Exception as e:
        print(f"Erreur extraction contenu {url}: {e}")
        return f"Impossible d'extraire le contenu de {url}"
def intelligent_query_expansion(query):
    """
    Expansion intelligente des requêtes pour améliorer les résultats
    Args:
        query: Requête originale
    Returns:
        Liste de requêtes expandues
    """
    expanded_queries = [query] # Toujours inclure la requête originale
    # Détection de mots-clés pour expansion contextuelle
    keywords = {
        'actualité': ['news', 'dernières nouvelles', 'récent'],
        'comment': ['tutorial', 'guide', 'étapes'],
        'pourquoi': ['raison', 'cause', 'explication'],
        'comparaison': ['vs', 'différence', 'comparatif'],
        'prix': ['coût', 'tarif', 'budget'],
        'avis': ['opinion', 'critique', 'review']
    }
    query_lower = query.lower()
    for trigger, expansions in keywords.items():
        if trigger in query_lower:
            for expansion in expansions:
                expanded_queries.append(f"{query} {expansion}")
    return expanded_queries[:3] # Limiter à 3 requêtes max
def hybrid_search_enhanced(query, vectordb, k=3, web_search_enabled=True, search_type="both"):
    """
    Recherche hybride améliorée combinant RAG local et web avec intelligence
    Args:
        query: Requête de recherche
        vectordb: Base vectorielle locale
        k: Nombre de résultats RAG
        web_search_enabled: Activer la recherche web
        search_type: Type de recherche web
    Returns:
        Liste de documents combinés et enrichis
    """
    all_results = []
    # 1. Recherche RAG locale
    local_docs = rag_search(query, vectordb, k)
    for doc in local_docs:
        doc.metadata['search_source'] = 'local_rag'
        doc.metadata['relevance_score'] = 1.0 # Score max pour les docs locaux
    all_results.extend(local_docs)
    # 2. Recherche web intelligente si activée
    if web_search_enabled:
        print(f"🌐 Recherche web activée pour: {query}")
        # Expansion de requête pour de meilleurs résultats
        expanded_queries = intelligent_query_expansion(query)
        web_results = []
        for exp_query in expanded_queries:
            try:
                search_results = enhanced_web_search(
                    exp_query,
                    max_results=3,
                    search_type=search_type
                )
                for result in search_results:
                    # Créer un document à partir du résultat web
                    content = f"Titre: {result.get('title', '')}\n"
                    content += f"Contenu: {result.get('body', '')}\n"
                    if result.get('source_type') == 'news' and result.get('date'):
                        content += f"Date: {result.get('date')}\n"
                        content += f"Source: {result.get('source', '')}\n"
                    # Extraction de contenu supplémentaire si URL disponible
                    url = result.get('href') or result.get('url')
                    if url and len(result.get('body', '')) < 200:
                        print(f"📄 Extraction contenu de: {url}")
                        extra_content = smart_content_extraction(url)
                        if extra_content and "Impossible d'extraire" not in extra_content:
                            content += f"\nContenu détaillé: {extra_content}"
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': url or 'web_search',
                            'type': result.get('source_type', 'web'),
                            'search_source': 'web',
                            'query_used': exp_query,
                            'relevance_score': 0.8 if exp_query == query else 0.6
                        }
                    )
                    web_results.append(doc)
            except Exception as e:
                print(f"Erreur recherche pour '{exp_query}': {e}")
                continue
        # Filtrer les doublons et trier par pertinence
        unique_web_results = []
        seen_urls = set()
        for doc in web_results:
            url = doc.metadata.get('source', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_web_results.append(doc)
        # Trier par score de pertinence
        unique_web_results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        all_results.extend(unique_web_results[:5]) # Max 5 résultats web
    return all_results
def generate_answer_enhanced(question, context_docs, model_name, include_sources=True):
    """
    Génération de réponse améliorée avec gestion des sources multiples
    Args:
        question: Question posée
        context_docs: Documents de contexte
        model_name: Modèle à utiliser
        include_sources: Inclure les sources dans la réponse
    Returns:
        Réponse générée avec sources
    """
    if not context_docs:
        context = "Aucun contexte spécifique trouvé."
    else:
        context_parts = []
        local_sources = []
        web_sources = []
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Document inconnu')
            doc_type = doc.metadata.get('type', 'unknown')
            search_source = doc.metadata.get('search_source', 'unknown')
            content = doc.page_content.strip()
            # Classifier les sources
            if search_source == 'local_rag':
                local_sources.append(f"[{i+1}] {source} ({doc_type})")
            else:
                web_sources.append(f"[{i+1}] {source}")
            context_parts.append(f"[Source {i+1} - {doc_type}]\n{content}")
        context = "\n\n".join(context_parts)
    # Prompt amélioré avec instructions pour les sources
    prompt = f"""Tu es un assistant IA intelligent qui répond aux questions en utilisant à la fois des documents locaux et des informations web récentes.
CONTEXTE DISPONIBLE:
{context}
QUESTION: {question}
INSTRUCTIONS:
- Utilise toutes les sources disponibles pour donner une réponse complète et précise
- Si les informations web contredisent les documents locaux, mentionne les deux perspectives
- Privilégie les informations récentes pour les sujets d'actualité
- Sois précis et cite tes sources si nécessaire
- Si certaines informations manquent, dis-le clairement
RÉPONSE DÉTAILLÉE:"""
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=600,
            temperature=0.3
        )
        answer = response.choices[0].message.content
        # Ajouter les sources si demandé
        if include_sources and context_docs:
            sources_text = "\n\n📚 **Sources consultées:**\n"
            if local_sources:
                sources_text += "**Documents locaux:**\n"
                for source in local_sources[:3]: # Limiter l'affichage
                    sources_text += f"• {source}\n"
            if web_sources:
                sources_text += "**Sources web:**\n"
                for source in web_sources[:3]: # Limiter l'affichage
                    sources_text += f"• {source}\n"
            answer += sources_text
        return answer
    except Exception as e:
        return f"❌ Erreur génération: {str(e)}"
# ===============================================
# Fonctions Web Search et Hybrid (Mises à jour)
# ===============================================
def web_search(query, max_results=5):
    """Version simplifiée pour compatibilité"""
    try:
        results = enhanced_web_search(query, max_results, "text")
        return [f"{r.get('title', '')}: {r.get('href', r.get('url', ''))} - {r.get('body', '')}" for r in results]
    except Exception as e:
        return [f"❌ Erreur recherche web: {e}"]
def hybrid_search(query, vectordb, k=3):
    """Version simplifiée pour compatibilité"""
    return hybrid_search_enhanced(query, vectordb, k, web_search_enabled=True)
def final_search(question, vectordb, graph, pois):
    """Recherche finale combinant toutes les sources"""
    results = hybrid_search_enhanced(question, vectordb, k=3, web_search_enabled=True)
    # OSM si mention lieu
    if any(keyword in question.lower() for keyword in ["aller", "trajet", "itinéraire", "route", "navigation"]):
        try:
            carte, reponse, traj = calculer_trajet(question, graph, pois)
            if traj:
                results.append(Document(
                    page_content=reponse,
                    metadata={"source": "trajet_osm", "type": "navigation"}
                ))
        except:
            pass
    return results
# ===============================================
# Fonctions Modèles Hugging Face Spécialisés
# ===============================================
def initialize_specialized_models():
    """Initialise les modèles spécialisés avec gestion d'erreurs"""
    models = {}
    
    # Vérifier si pipeline est disponible
    if pipeline is None:
        print("⚠️ transformers.pipeline non disponible, modèles spécialisés désactivés")
        return {
            'summarizer': None,
            'translator': None,
            'captioner': None,
            'ner': None
        }
    
    try:
        models['summarizer'] = pipeline("summarization", model=SUMMARIZER_MODEL, cache_dir=str(SUMMARIZER_CACHE))
        print("✅ Modèle de résumé chargé")
    except Exception as e:
        print(f"⚠️ Erreur chargement summarizer: {e}")
        models['summarizer'] = None
    try:
        models['translator'] = pipeline("translation", model=TRANSLATOR_MODEL, cache_dir=str(TRANSLATOR_CACHE))
        print("✅ Modèle de traduction chargé")
    except Exception as e:
        print(f"⚠️ Erreur chargement translator: {e}")
        models['translator'] = None
    try:
        models['captioner'] = None
        print("✅ Captioner configuré pour utiliser LLM (llava)")
    except Exception as e:
        print(f"⚠️ Erreur chargement captioner: {e}")
        models['captioner'] = None
    try:
        models['ner'] = pipeline("ner", model=NER_MODEL, cache_dir=str(NER_CACHE))
        print("✅ Modèle NER chargé")
    except Exception as e:
        print(f"⚠️ Erreur chargement NER: {e}")
        models['ner'] = None
    return models
# Initialiser les modèles
SPECIALIZED_MODELS = initialize_specialized_models()
def summarize_text(text):
    if SPECIALIZED_MODELS.get('summarizer') is None:
        return "❌ Modèle de résumé non disponible"
    try:
        return SPECIALIZED_MODELS['summarizer'](text[:1024], max_length=200, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"❌ Erreur résumé: {e}"
def translate_text(text, src_lang="fr", tgt_lang="en"):
    if SPECIALIZED_MODELS.get('translator') is None:
        return "❌ Modèle de traduction non disponible"
    try:
        return SPECIALIZED_MODELS['translator'](text)[0]['translation_text']
    except Exception as e:
        return f"❌ Erreur traduction: {e}"
def caption_image(image_path):
    client = create_client()
    model = "llava-hf/llava-1.5-7b-hf"
    prompt = "Generate a detailed caption for this image."
    try:
        return client.image_to_text(image_path, prompt=prompt, model=model, max_tokens=500)
    except Exception as e:
        return f"❌ Erreur caption: {e}"
def extract_entities(text):
    if SPECIALIZED_MODELS.get('ner') is None:
        return "❌ Modèle NER non disponible"
    try:
        return SPECIALIZED_MODELS['ner'](text)
    except Exception as e:
        return f"❌ Erreur NER: {e}"
# ===============================================
# Fonctions de génération avec Stable Diffusion et similaires
# ===============================================
def generate_text_to_image(prompt):
    """Génère une image à partir de texte"""
    try:
        pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = pipe(prompt).images[0]
        path = os.path.join(GENERATED_PATH, f"image_{int(time.time())}.png")
        image.save(path)
        return f"Image générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération image: {e}"
def generate_text_to_video(prompt):
    """Génère une vidéo à partir de texte"""
    try:
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        gen = pipe(prompt, num_inference_steps=25)
        frames = gen.frames[0] # Assuming batch size 1
        path = os.path.join(GENERATED_PATH, f"video_{int(time.time())}.gif")
        imageio.mimsave(path, frames, fps=5)
        return f"Vidéo générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération vidéo: {e}"
def generate_text_to_audio(prompt):
    """Génère un son à partir de texte"""
    try:
        pipe = AudioLDMPipeline.from_pretrained("cvssp/audio-ldm", torch_dtype=torch.float16, use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        audio = pipe(prompt, audio_length_in_s=5.0).audios[0]
        path = os.path.join(GENERATED_PATH, f"audio_{int(time.time())}.wav")
        wavfile.write(path, rate=16000, data=audio) # Assuming 16kHz sample rate
        return f"Son généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération son: {e}"
def generate_text_to_3d(prompt):
    """Génère un modèle 3D à partir de texte (rendue image)"""
    try:
        pipe = ShapEPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        output = pipe(prompt, num_inference_steps=64)
        image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_text_{int(time.time())}.png")
        image.save(path)
        return f"Rendu 3D généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (texte): {e}"
def generate_image_to_3d(image_path):
    """Génère un modèle 3D à partir d'une image (rendue image)"""
    try:
        pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = Image.open(image_path)
        output = pipe(image, num_inference_steps=64)
        rendered_image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_image_{int(time.time())}.png")
        rendered_image.save(path)
        return f"Rendu 3D généré à partir de l'image et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (image): {e}"
# ===============================================
# Agent LangChain Amélioré avec Recherche Web
# ===============================================
def create_enhanced_agent(model_name, vectordb, graph, pois):
    """
    Crée un agent LangChain amélioré avec capacités de recherche web
    Args:
        model_name: Nom du modèle HuggingFace
        vectordb: Base vectorielle locale
        graph: Graphe OSM
        pois: Points d'intérêt
    Returns:
        Agent configuré avec tous les outils
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3,
            max_new_tokens=600
        )
        # Configuration des outils de recherche web
        search_wrapper = DuckDuckGoSearchAPIWrapper(
            region="fr-fr",
            time="d",
            max_results=5
        )
        search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
        search_results_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5, include_raw_content=True)
        tools = [
            # Outils de base RAG et recherche
            Tool(
                name="Local_Knowledge_Base",
                func=lambda q: "\n\n".join([d.page_content for d in rag_search(q, vectordb, k=3)]),
                description="Recherche dans la base de connaissances locale (PDFs et documents internes). Utilise ceci en PREMIER pour les questions sur des documents spécifiques."
            ),
            Tool(
                name="Web_Search",
                func=lambda q: search_tool.run(q),
                description="Recherche sur Internet pour des informations récentes, actualités, ou des connaissances générales non disponibles localement."
            ),
            Tool(
                name="Web_Search_Detailed",
                func=lambda q: search_results_tool.run(q),
                description="Recherche web détaillée avec sources et liens. Utilise pour obtenir des résultats web structurés avec URLs."
            ),
            Tool(
                name="Hybrid_Search",
                func=lambda q: "\n\n".join([d.page_content for d in hybrid_search_enhanced(q, vectordb, k=3, web_search_enabled=True)]),
                description="Recherche hybride combinant base locale ET web. Idéal pour des questions nécessitant à la fois des données internes et externes."
            ),
            Tool(
                name="Current_News_Search",
                func=lambda q: "\n\n".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in enhanced_web_search(q, search_type="news")]),
                description="Recherche spécialisée pour les actualités récentes et informations temporelles."
            ),
            # Outils spécialisés
            Tool(
                name="OSM_Route_Calculator",
                func=lambda q: calculer_trajet(q, graph, pois)[1] if graph and pois else "❌ Aucune carte OSM disponible",
                description="Calcule des itinéraires routiers entre deux lieux. Utilise pour les questions de navigation, trajets, ou géolocalisation."
            ),
            Tool(
                name="Smart_Content_Extractor",
                func=lambda url: smart_content_extraction(url) if url.startswith('http') else "❌ URL invalide",
                description="Extrait le contenu détaillé d'une page web spécifique. Fournis une URL complète."
            ),
            Tool(
                name="Text_Summarizer",
                func=summarize_text,
                description="Résume un texte long en version concise. Utile pour synthétiser des informations volumineuses."
            ),
            Tool(
                name="Language_Translator",
                func=translate_text,
                description="Traduit du français vers l'anglais. Utile pour traiter des sources en langue étrangère."
            ),
            Tool(
                name="Image_Analyzer",
                func=caption_image,
                description="Analyse et décrit le contenu d'une image. Fournis le chemin vers un fichier image."
            ),
            Tool(
                name="Entity_Extractor",
                func=lambda t: json.dumps(extract_entities(t)),
                description="Extrait des entités nommées (personnes, lieux, organisations) d'un texte."
            ),
            # Nouveaux outils Stable Diffusion via API
            Tool(
                name="Text_To_Image_Generator",
                func=generate_text_to_image,
                description="Génère une image à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Text_To_Video_Generator",
                func=generate_text_to_video,
                description="Génère une vidéo à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Text_To_Audio_Generator",
                func=generate_text_to_audio,
                description="Génère un son ou audio à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Text_To_3D_Generator",
                func=generate_text_to_3d,
                description="Génère un modèle 3D (rendue image) à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Image_To_3D_Generator",
                func=generate_image_to_3d,
                description="Génère un modèle 3D (rendue image) à partir d'une image. Fournis le chemin vers un fichier image."
            ),
        ]
        # Configuration de l'agent avec prompt personnalisé
        agent_prompt = """Tu es Kibali, un assistant IA avancé avec accès à de multiples sources d'information.
CAPACITÉS DISPONIBLES:
- Base de connaissances locale (PDFs et documents)
- Recherche web en temps réel
- Calcul d'itinéraires sur cartes OSM
- Analyse d'images et extraction de contenu web
- Traduction et résumé automatiques
- Génération d'images, vidéos, sons et modèles 3D à partir de texte ou images
INSTRUCTIONS IMPORTANTES:
1. Utilise TOUJOURS la base locale en premier pour les questions sur des documents spécifiques
2. Combine les sources locales ET web pour des réponses complètes
3. Pour les actualités ou infos récentes, privilégie la recherche web
4. Cite tes sources et indique leur provenance (locale vs web)
5. Si les informations se contredisent, mentionne les deux perspectives
6. Reste concis mais informatif
7. Pour les générations, sauvegarde les fichiers et retourne le chemin
Tu as accès aux outils suivants: {tools}
Utilise le format suivant:
Question: la question d'entrée
Thought: réfléchis à ce que tu dois faire
Action: l'action à entreprendre, doit être l'un de [{tool_names}]
Action Input: l'entrée à l'action
Observation: le résultat de l'action
... (ce Thought/Action/Action Input/Observation peut se répéter N fois)
Thought: Je connais maintenant la réponse finale
Final Answer: la réponse finale à la question d'entrée
Commence!
Question: {input}
Thought: {agent_scratchpad}"""
        agent = initialize_agent(
            tools,
            llm,
            agent="zero-shot-react-description",
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        print(f"✅ Agent créé avec {len(tools)} outils disponibles")
        return agent
    except Exception as e:
        print(f"❌ Erreur création agent: {e}")
        return None
# Alias pour compatibilité
def create_agent(model_name, vectordb, graph, pois):
    """Version simplifiée pour compatibilité ascendante"""
    return create_enhanced_agent(model_name, vectordb, graph, pois)
# ===============================================
# Fonctions OSM et Graphe Routier
# ===============================================
def haversine(lon1, lat1, lon2, lat2):
    """Calcul distance haversine en mètres"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
class RoadPOIHandler(osmium.SimpleHandler):
    """Handler pour extraire routes et POIs depuis OSM"""
    def __init__(self):
        super().__init__()
        self.graph = nx.Graph()
        self.pois = []
    def node(self, n):
        """Extraire les POIs (points d'intérêt)"""
        if n.location.valid() and n.tags:
            name = n.tags.get('name', '')
            amenity = n.tags.get('amenity', '')
            if name or amenity:
                self.pois.append({
                    'name': name,
                    'amenity': amenity,
                    'lon': n.location.lon,
                    'lat': n.location.lat,
                    'tags': dict(n.tags)
                })
    def way(self, w):
        """Extraire les routes"""
        if 'highway' in w.tags:
            coords = []
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.location.lon, n.location.lat))
            for i in range(len(coords)-1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i+1]
                n1, n2 = (lon1, lat1), (lon2, lat2)
                dist = haversine(lon1, lat1, lon2, lat2)
                self.graph.add_node(n1, x=lon1, y=lat1)
                self.graph.add_node(n2, x=lon2, y=lat2)
                self.graph.add_edge(n1, n2, length=dist, highway=w.tags.get("highway"))
def trouver_noeud_plus_proche(lon, lat, graph):
    """Trouve le nœud du graphe le plus proche"""
    min_dist = float("inf")
    closest_node = None
    for node, data in graph.nodes(data=True):
        nlon, nlat = float(data["x"]), float(data["y"])
        dist = haversine(lon, lat, nlon, nlat)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node
def chercher_poi_par_nom(nom, pois_list):
    """Recherche un POI par nom"""
    nom_lower = nom.lower()
    for poi in pois_list:
        if nom_lower in poi['name'].lower() or nom_lower in poi['amenity'].lower():
            return poi
    return None
def generer_carte_trajet(graph, path, pois_list, start_poi=None, end_poi=None):
    """Génère une carte 2D du trajet"""
    fig, ax = plt.subplots(figsize=(12, 10))
    # Dessiner le graphe en arrière-plan
    for edge in list(graph.edges())[:1000]: # Limiter pour la performance
        node1, node2 = edge
        x1, y1 = node1[0], node1[1]
        x2, y2 = node2[0], node2[1]
        ax.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.3, linewidth=0.5)
    # Dessiner le trajet
    if path and len(path) > 1:
        path_x = [node[0] for node in path]
        path_y = [node[1] for node in path]
        ax.plot(path_x, path_y, 'red', linewidth=3, label='Trajet')
        # Marquer début et fin
        ax.scatter(path_x[0], path_y[0], color='green', s=100, label='Départ', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], color='red', s=100, label='Arrivée', zorder=5)
    # Ajouter quelques POIs
    for poi in pois_list[:20]:
        if poi['name']:
            ax.scatter(poi['lon'], poi['lat'], color='blue', s=20, alpha=0.6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Trajet calculé sur la carte OSM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    # Sauvegarder en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf
def calculer_trajet(question, graph, pois_list):
    """Calcule un trajet basé sur une question textuelle"""
    if not graph or not pois_list:
        return None, "❌ Graphe ou POIs non disponibles", None
    # Utiliser LLM pour extraire départ et arrivée
    try:
        client = create_client()
        prompt = f"""Extraie le lieu de départ et le lieu d'arrivée de cette question de trajet.
Question: {question}
Réponds au format exact:
Départ: [nom du lieu de départ]
Arrivée: [nom du lieu d'arrivée]"""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=WORKING_MODELS["Qwen2.5 7B (Rapide & excellent)"],
            messages=messages,
            max_tokens=100,
            temperature=0.1
        )
        extraction = response.choices[0].message.content
        start_line = [line for line in extraction.split('\n') if line.startswith('Départ: ')]
        end_line = [line for line in extraction.split('\n') if line.startswith('Arrivée: ')]
        if start_line and end_line:
            start_place = start_line[0].replace('Départ: ', '').strip()
            end_place = end_line[0].replace('Arrivée: ', '').strip()
        else:
            return None, "❌ Impossible d'extraire les lieux de la question.", None
    except Exception as e:
        print(f"❌ Erreur extraction LLM: {e}")
        return None, "❌ Erreur lors de l'extraction des lieux.", None
    start_poi = chercher_poi_par_nom(start_place, pois_list)
    end_poi = chercher_poi_par_nom(end_place, pois_list)
    if not start_poi or not end_poi:
        return None, f"❌ Impossible de trouver les lieux: {start_place} ou {end_place}.", None
    # Trouver les nœuds dans le graphe
    start_node = trouver_noeud_plus_proche(start_poi['lon'], start_poi['lat'], graph)
    end_node = trouver_noeud_plus_proche(end_poi['lon'], end_poi['lat'], graph)
    if not start_node or not end_node:
        return None, "❌ Impossible de trouver les nœuds dans le graphe routier.", None
    try:
        # Calculer le chemin
        path = nx.shortest_path(graph, source=start_node, target=end_node, weight="length")
        # Calculer la distance
        distance_totale = 0
        for i in range(len(path)-1):
            distance_totale += graph[path[i]][path[i+1]]['length']
        # Générer la carte
        carte_buf = generer_carte_trajet(graph, path, pois_list, start_poi, end_poi)
        # Réponse textuelle
        reponse = f"""🗺️ **Trajet calculé**
📍 **Départ**: {start_poi['name']} ({start_poi['amenity']})
🎯 **Arrivée**: {end_poi['name']} ({end_poi['amenity']})
📏 **Distance**: {distance_totale/1000:.2f} km
⏱️ **Temps estimé**: {int(distance_totale/83.33):.0f} min à pied | {int(distance_totale/833.33):.0f} min en voiture
🛣️ **Étapes**: {len(path)} points"""
        return carte_buf, reponse, {
            'start': start_poi,
            'end': end_poi,
            'distance': distance_totale,
            'path_length': len(path)
        }
    except nx.NetworkXNoPath:
        return None, f"❌ Aucun chemin trouvé entre {start_poi['name']} et {end_poi['name']}", None
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}", None
# ===============================================
# Fonctions utilitaires pour images
# ===============================================
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
def df_to_html(df, max_rows=10):
    # Réduire le tableau si trop long
    if len(df) > max_rows:
        summary_row = pd.DataFrame({col: ['...'] for col in df.columns})
        df = pd.concat([df.head(max_rows // 2), summary_row, df.tail(max_rows // 2)])
    return df.to_html(index=False, escape=False)
# ===============================================
# Fonctions Image Analysis
# ===============================================
def classify_soil(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:,:,0])
    mean_sat = np.mean(hsv[:,:,1])
    mean_val = np.mean(hsv[:,:,2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray)
    soil_type = "Inconnu"
    possible_contents = "Inconnu"
    possible_minerals = "Inconnu"
    if mean_val < 100 and texture_variance > 5000:
        soil_type = "Argileux (riche en matière organique)"
        possible_contents = "Peut contenir de l'eau, nutriments, adapté aux cultures racines"
        possible_minerals = "Argiles comme kaolinite, illite; possible fer, aluminium"
    elif mean_sat > 100 and texture_variance < 3000:
        soil_type = "Sableux (drainant)"
        possible_contents = "Peut contenir peu d'eau, adapté aux plantes résistantes à la sécheresse"
        possible_minerals = "Quartz, feldspath; silice abondante"
    elif mean_hue > 20 and mean_hue < 40:
        soil_type = "Limoneux (équilibré)"
        possible_contents = "Peut contenir minéraux, bon pour l'agriculture générale"
        possible_minerals = "Silt avec mica, quartz; calcium, potassium"
    # Graphisme : Histogramme des couleurs HSV
    fig, ax = plt.subplots()
    ax.hist(hsv[:,:,0].ravel(), bins=50, color='b', alpha=0.5, label='Hue')
    ax.hist(hsv[:,:,1].ravel(), bins=50, color='g', alpha=0.5, label='Saturation')
    ax.hist(hsv[:,:,2].ravel(), bins=50, color='r', alpha=0.5, label='Value')
    ax.set_title('Histogramme des Composantes HSV')
    ax.legend()
    hist_img = fig_to_pil(fig)
    # Tableau des metrics
    metrics_df = pd.DataFrame({
        'Métrique': ['Hue Moyenne', 'Saturation Moyenne', 'Valeur Moyenne', 'Variance Texture'],
        'Valeur': [mean_hue, mean_sat, mean_val, texture_variance],
        'Explication': ['Moyenne de la teinte', 'Moyenne de la saturation des couleurs', 'Moyenne de la luminosité', 'Variance de la texture pour rugosité']
    })
    metrics_html = df_to_html(metrics_df)
    return {
        "soil_type": soil_type,
        "possible_contents": possible_contents,
        "possible_minerals": possible_minerals
    }, hist_img, metrics_html
def simulate_infrared(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ir_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
    ax.set_title('Simulation Infrarouge (Colormap JET)')
    ax.axis('off')
    ir_pil = fig_to_pil(fig)
    # Analyse simple (fake temp based on intensity)
    mean_intensity = np.mean(gray)
    ir_analysis = f"Simulation IR: Intensité moyenne {mean_intensity:.2f} (plus rouge = plus chaud, bleu = plus froid)"
    return ir_pil, ir_analysis
def detect_objects(image: np.ndarray, scale_factor=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = image.copy()
    dimensions = []
    types = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10: continue  # skip small
        cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
        w_m = w * scale_factor
        h_m = h * scale_factor
        aspect = w / h if h != 0 else 0
        if aspect > 5: obj_type = 'Route'
        elif aspect < 0.2: obj_type = 'Clôture'
        elif 0.5 < aspect < 2: obj_type = 'Bâtiment'
        else: obj_type = 'Autre'
        dimensions.append((w_m, h_m))
        types.append(obj_type)
        cv2.putText(img_with_contours, f"{obj_type}: {w_m:.4f}m x {h_m:.4f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    num_objects = len(contours)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Objets Détectés avec Contours ({num_objects})")
    ax.axis('off')
    obj_img = fig_to_pil(fig)
    if dimensions:
        dim_df = pd.DataFrame({
            'Type': types,
            'Largeur (m)': [d[0] for d in dimensions],
            'Hauteur (m)': [d[1] for d in dimensions],
            'Explication': ['Dimension estimée avec contours OpenCV' for _ in types]
        })
        dim_html = df_to_html(dim_df)
    else:
        dim_html = ""
    return num_objects, obj_img, dim_html
def detect_fences(image: np.ndarray, scale_factor=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    img_with_lines = image.copy()
    lengths = []
    if lines is not None:
        line_list = [line[0] for line in lines]
        filtered_lines = [l for l in line_list if abs(l[0] - l[2]) < 10 or abs(l[1] - l[3]) < 10 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) < 0.1 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) > 10]
        line_lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for x1,y1,x2,y2 in filtered_lines]
        sorted_indices = np.argsort(line_lengths)[::-1]
        sorted_lines = [filtered_lines[i] for i in sorted_indices]
        for x1,y1,x2,y2 in sorted_lines:
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * scale_factor
            lengths.append(length)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(img_with_lines, f"{length:.4f}m", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Clôtures/Bordures Détectées ({len(lengths)})")
    ax.axis('off')
    fence_img = fig_to_pil(fig)
    if lengths:
        fence_df = pd.DataFrame({
            'Longueur (m)': lengths,
            'Explication': ['Longueur de bordure filtrée et triée pour précision' for _ in lengths]
        })
        fence_html = df_to_html(fence_df)
    else:
        fence_html = ""
    return len(lengths), fence_img, fence_html
def detect_anomalies(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    num_edges = np.sum(edges > 0)
    mean_variance = np.mean(cv2.Laplacian(gray, cv2.CV_64F).var())
    anomalies = []
    if num_edges > 10000:
        anomalies.append("Anomalies structurelles détectées (ex. : fissures, défauts)")
    if mean_variance > 500:
        anomalies.append("Textures inhabituelles (ex. : zones irrégulières)")
    # Simulation photogrammétrie basique SANS Open3D (désactivé pour Python 3.13)
    # Open3D n'est pas disponible pour Python 3.13, donc on simule autrement
    num_points = gray.shape[0] * gray.shape[1]  # Nombre de pixels comme approximation
    
    # Graphisme : Histogramme des variances
    fig, ax = plt.subplots()
    ax.hist(cv2.Laplacian(gray, cv2.CV_64F).ravel(), bins=50)
    ax.set_title('Histogramme des Variances Locales (Anomalies)')
    var_hist_img = fig_to_pil(fig)
    # Tableau des metrics anomalies
    anomaly_df = pd.DataFrame({
        'Métrique': ['Nombre de Bords', 'Variance Moyenne', 'Points Estimés'],
        'Valeur': [num_edges, mean_variance, num_points],
        'Explication': ['Indique complexité structurelle (haut = anomalies)', 'Mesure irrégularités texture', 'Estimation de densité pour volume']
    })
    anomaly_html = df_to_html(anomaly_df)
    anomaly_desc_df = pd.DataFrame({
        'Anomalie': anomalies if anomalies else ['Aucune anomalie détectée'],
        'Explication': ['Défauts potentiels dans le terrain ou structures' if anomalies else 'Image sans anomalies significatives']
    })
    anomaly_desc_html = df_to_html(anomaly_desc_df)
    return anomalies, var_hist_img, anomaly_html, anomaly_desc_html
def advanced_analyses(image: np.ndarray):
    analyses = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)
    adv_images = []
    adv_tables = []
    # 1. Analyse Géologique
    kmeans = KMeans(n_clusters=3).fit(gray.reshape(-1, 1))
    clustered = kmeans.labels_.reshape(gray.shape)
    analyses['Géologique'] = 'Clusters de textures : ' + str(np.unique(kmeans.labels_))
    fig, ax = plt.subplots()
    ax.imshow(clustered, cmap='viridis')
    ax.set_title('Analyse Géologique: Clustering Textures')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    geo_df = pd.DataFrame({'Cluster': np.unique(kmeans.labels_), 'Compte': np.bincount(kmeans.labels_), 'Explication': ['Groupe de texture géologique' for _ in np.unique(kmeans.labels_)]})
    adv_tables.append(df_to_html(geo_df))
    # 2. Analyse Hydrologique
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    water_area = np.sum(blue_mask > 0) / blue_mask.size * 100
    analyses['Hydrologique'] = f'Pourcentage eau : {water_area:.2f}%'
    fig, ax = plt.subplots()
    ax.imshow(blue_mask, cmap='gray')
    ax.set_title('Analyse Hydrologique: Masque Eau')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    hydro_df = pd.DataFrame({'Métrique': ['Pourcentage Eau'], 'Valeur': [water_area], 'Explication': ['Zone potentielle pour ressources hydriques']})
    adv_tables.append(df_to_html(hydro_df))
    return analyses, {}, adv_images, adv_tables
def process_image(uploaded_file):
    image = Image.open(BytesIO(uploaded_file))
    img_array = np.array(image)
    proc_images = [image]
    captions = ['Image Originale']
    tables_html = []
    # IR
    ir_pil, ir_analysis = simulate_infrared(img_array)
    proc_images.append(ir_pil)
    captions.append('Simulation Infrarouge')
    tables_html.append('<h3>Analyse IR</h3><p>' + ir_analysis + '</p>')
    # Soil
    soil, hist_img, metrics_html = classify_soil(img_array)
    proc_images.append(hist_img)
    captions.append('Histogramme HSV')
    tables_html.append('<h3>Métriques Sol</h3>' + metrics_html)
    # Objects
    num_objects, obj_img, dim_html = detect_objects(img_array)
    proc_images.append(obj_img)
    captions.append('Objets Détectés')
    if dim_html:
        tables_html.append('<h3>Dimensions Objets</h3>' + dim_html)
    # Fences
    num_fences, fence_img, fence_html = detect_fences(img_array)
    proc_images.append(fence_img)
    captions.append('Clôtures Détectées')
    if fence_html:
        tables_html.append('<h3>Longueurs Clôtures</h3>' + fence_html)
    # Anomalies
    anomalies, var_hist_img, anomaly_html, anomaly_desc_html = detect_anomalies(img_array)
    proc_images.append(var_hist_img)
    captions.append('Histogramme Variances')
    tables_html.append('<h3>Métriques Anomalies</h3>' + anomaly_html)
    # Advanced
    analyses, predictions, adv_images, adv_tables = advanced_analyses(img_array)
    proc_images += adv_images[:5] # Limiter le nombre d'images
    captions += ['Analyse Avancée'] * len(adv_images[:5])
    tables_html += adv_tables[:3] # Limiter le nombre de tableaux
    analysis_data = {
        "soil": soil,
        "ir_analysis": ir_analysis,
        "num_objects": num_objects,
        "num_fences": num_fences,
        "anomalies": anomalies,
        "analyses": analyses,
        "predictions": predictions
    }
    tables_str = '<br>'.join(tables_html)
    return analysis_data, proc_images, tables_str
def improve_analysis_with_llm(analysis_data, model_name):
    prompt = f"""Analyse les données suivantes de l'image et fournis une analyse naturelle améliorée:
DONNÉES:
{json.dumps(analysis_data, indent=2)}
ANALYSE AMÉLIORÉE:"""
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=800,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur: {str(e)}"
# ===============================================
# Interface Streamlit Améliorée
# ===============================================
def main():
    """Interface Streamlit unifiée avec recherche web intégrée"""

    # ===============================================
    # CSS PERSONNALISÉ - DESIGN PUISSANT ET FLUIDE
    # ===============================================
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Variables CSS pour les couleurs Kibali - Version améliorée et moins sombre */
    :root {
        --kibali-green: #00ff88;
        --kibali-yellow: #ffd700;
        --kibali-blue: #0088ff;
        --kibali-dark: #1a1a2e;        /* Plus clair que #0a0a0a */
        --kibali-darker: #0f0f23;     /* Plus clair que #050505 */
        --kibali-light: #2a2a4e;      /* Plus clair que #1a1a1a */
        --kibali-lighter: #3a3a5e;    /* Nouvelle couleur pour les cartes */
        --kibali-accent: #4a4a7e;     /* Accent color */
        --kibali-border: #5a5a8a;     /* Bordures plus douces */
        --kibali-text: #ffffff;
        --kibali-text-secondary: #e0e0e0;  /* Plus clair pour meilleure lisibilité */
        --kibali-text-muted: #b0b0b0;      /* Nouveau pour texte moins important */
        --gradient-primary: linear-gradient(135deg, var(--kibali-green), var(--kibali-blue));
        --gradient-secondary: linear-gradient(135deg, var(--kibali-yellow), var(--kibali-green));
        --gradient-background: linear-gradient(135deg, var(--kibali-dark), var(--kibali-darker));
        --shadow-glow: 0 0 20px rgba(0, 255, 136, 0.3);
        --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.2);
        --shadow-subtle: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* Reset et base - Version améliorée */
    * {
        font-family: 'Inter', sans-serif !important;
    }

    /* Fond principal avec gradient subtil et texture */
    .main {
        background: var(--gradient-background) !important;
        background-attachment: fixed !important;
        position: relative;
    }

    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
            radial-gradient(circle at 20% 80%, rgba(0, 255, 136, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(0, 136, 255, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 215, 0, 0.02) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }

    .stApp {
        background: transparent !important;
    }

    /* Logo Kibali animé */
    .kibali-logo {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0;
        animation: logoGlow 3s ease-in-out infinite alternate;
        text-shadow: var(--shadow-glow);
    }

    /* Nouveau logo K Kibali */
    .kibali-k-logo {
        font-size: 6rem;
        font-weight: 900;
        text-align: center;
        margin: 1rem 0;
        color: black; /* Texte en noir */
        position: relative;
        display: inline-block;
        animation: kGlowBorder 4s ease-in-out infinite alternate, kRotate 8s linear infinite;
        text-shadow: none; /* Supprimer l'ombre du texte */
    }

    .kibali-k-logo::before {
        content: '';
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        background: linear-gradient(45deg, var(--kibali-green), var(--kibali-yellow), var(--kibali-blue));
        border-radius: 20px;
        z-index: -1;
        opacity: 0.8;
        animation: kBackgroundPulse 3s ease-in-out infinite alternate, kBorderShine 2s linear infinite;
    }

    @keyframes kGlow {
        0% {
            filter: brightness(1) drop-shadow(0 0 20px rgba(0, 255, 136, 0.5));
            transform: scale(1);
        }
        50% {
            filter: brightness(1.3) drop-shadow(0 0 40px rgba(255, 215, 0, 0.7));
            transform: scale(1.05);
        }
        100% {
            filter: brightness(1) drop-shadow(0 0 30px rgba(0, 136, 255, 0.6));
            transform: scale(1);
        }
    }

    @keyframes kGlowBorder {
        0% {
            filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.8));
        }
        33% {
            filter: drop-shadow(0 0 15px rgba(255, 215, 0, 0.9));
        }
        66% {
            filter: drop-shadow(0 0 20px rgba(0, 136, 255, 0.8));
        }
        100% {
            filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.8));
        }
    }

    @keyframes kBorderShine {
        0% {
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-yellow), var(--kibali-blue));
            opacity: 0.6;
        }
        25% {
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-blue), var(--kibali-green));
            opacity: 0.8;
        }
        50% {
            background: linear-gradient(45deg, var(--kibali-blue), var(--kibali-green), var(--kibali-yellow));
            opacity: 1.0;
        }
        75% {
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-blue), var(--kibali-yellow));
            opacity: 0.8;
        }
        100% {
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-green), var(--kibali-blue));
            opacity: 0.6;
        }
    }

    @keyframes kRotate {
        0% { transform: rotateY(0deg); }
        25% { transform: rotateY(5deg); }
        50% { transform: rotateY(0deg); }
        75% { transform: rotateY(-5deg); }
        100% { transform: rotateY(0deg); }
    }

    @keyframes kBackgroundPulse {
        0% { opacity: 0.1; transform: scale(1); }
        100% { opacity: 0.3; transform: scale(1.1); }
    }

    @keyframes logoGlow {
        0% { filter: brightness(1) drop-shadow(0 0 10px rgba(0, 255, 136, 0.5)); }
        100% { filter: brightness(1.2) drop-shadow(0 0 20px rgba(0, 255, 136, 0.8)); }
    }

    /* Header principal - Version améliorée */
    .main-header {
        background: linear-gradient(135deg, var(--kibali-light), var(--kibali-accent));
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid var(--kibali-border);
        box-shadow: var(--shadow-subtle);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
        animation: headerShine 4s ease-in-out infinite;
    }

    @keyframes headerShine {
        0%, 100% { transform: translateX(-100%); }
        50% { transform: translateX(100%); }
    }

    /* Cartes flex responsive - Version améliorée */
    .kibali-card {
        background: var(--kibali-lighter);
        border: 1px solid var(--kibali-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-subtle);
        position: relative;
        overflow: hidden;
    }

    .kibali-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.1), transparent);
        transition: left 0.5s;
    }

    .kibali-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-card);
        border-color: var(--kibali-green);
    }

    .kibali-card:hover::before {
        left: 100%;
    }

    /* Boutons stylisés */
    .kibali-btn {
        background: var(--gradient-primary);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 255, 136, 0.3);
        position: relative;
        overflow: hidden;
    }

    .kibali-btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .kibali-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 255, 136, 0.5);
    }

    .kibali-btn:hover::before {
        left: 100%;
    }

    .kibali-btn:active {
        transform: translateY(0);
    }

    /* Onglets personnalisés - Version améliorée */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--kibali-light);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid var(--kibali-border);
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--kibali-accent);
        border-radius: 8px;
        color: var(--kibali-text-secondary);
        transition: all 0.3s ease;
        border: none;
        padding: 12px 20px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--kibali-lighter);
        color: var(--kibali-text);
        transform: translateY(-2px);
        box-shadow: var(--shadow-subtle);
    }

    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }

    /* Chat messages - Version améliorée */
    .chat-message-user {
        background: linear-gradient(135deg, var(--kibali-blue), var(--kibali-green));
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 136, 255, 0.3);
        animation: slideInLeft 0.5s ease-out;
        border: 1px solid rgba(0, 136, 255, 0.2);
    }

    .chat-message-assistant {
        background: var(--kibali-lighter);
        color: var(--kibali-text);
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--kibali-green);
        box-shadow: var(--shadow-subtle);
        animation: slideInRight 0.5s ease-out;
        border: 1px solid var(--kibali-border);
    }

    @keyframes slideInLeft {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes slideInRight {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    /* Input fields - Version améliorée */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background: var(--kibali-accent) !important;
        border: 1px solid var(--kibali-border) !important;
        border-radius: 12px !important;
        color: var(--kibali-text) !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {
        border-color: var(--kibali-green) !important;
        box-shadow: 0 0 0 2px rgba(0, 255, 136, 0.2) !important;
        transform: scale(1.02);
    }

    /* Success/Error messages - Version améliorée */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        animation: fadeIn 0.5s ease-out;
        box-shadow: var(--shadow-subtle);
    }

    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 255, 136, 0.05)) !important;
        border-left: 4px solid var(--kibali-green) !important;
        color: var(--kibali-text) !important;
    }

    .stError {
        background: linear-gradient(135deg, rgba(255, 136, 136, 0.1), rgba(255, 136, 136, 0.05)) !important;
        border-left: 4px solid #ff8888 !important;
        color: var(--kibali-text) !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Spinner personnalisé */
    .stSpinner > div > div {
        border-color: var(--kibali-green) !important;
        border-top-color: transparent !important;
    }

    /* Images avec zoom au survol */
    .zoom-image {
        transition: transform 0.3s ease;
        border-radius: 12px;
        box-shadow: var(--shadow-card);
    }

    .zoom-image:hover {
        transform: scale(1.05);
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: var(--gradient-primary) !important;
    }

    /* Sidebar si utilisée */
    .css-1d391kg, .css-12oz5g7 {
        background: var(--kibali-dark) !important;
    }

    /* Scrollbar personnalisée - Version améliorée */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--kibali-dark);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--gradient-primary);
        border-radius: 4px;
        border: 1px solid var(--kibali-border);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--kibali-green);
        box-shadow: var(--shadow-glow);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .kibali-logo {
            font-size: 2.5rem;
        }

        .kibali-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }

        .main-header {
            padding: 1rem;
        }
    }

    /* Logo K pour les messages de chat */
    .chat-k-logo {
        display: inline-block;
        font-size: 1.2rem;
        font-weight: 900;
        color: black;
        position: relative;
        margin-right: 0.5rem;
        animation: chatKBorderShine 3s ease-in-out infinite;
    }

    .chat-k-logo::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, var(--kibali-green), var(--kibali-yellow), var(--kibali-blue));
        border-radius: 4px;
        z-index: -1;
        opacity: 0.7;
        animation: chatKBackgroundPulse 2s ease-in-out infinite alternate;
    }

    @keyframes chatKBorderShine {
        0% {
            filter: drop-shadow(0 0 3px rgba(0, 255, 136, 0.6));
        }
        33% {
            filter: drop-shadow(0 0 5px rgba(255, 215, 0, 0.7));
        }
        66% {
            filter: drop-shadow(0 0 4px rgba(0, 136, 255, 0.6));
        }
        100% {
            filter: drop-shadow(0 0 3px rgba(0, 255, 136, 0.6));
        }
    }

    @keyframes chatKBackgroundPulse {
        0% {
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-yellow), var(--kibali-blue));
            opacity: 0.5;
        }
        25% {
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-blue), var(--kibali-green));
            opacity: 0.7;
        }
        50% {
            background: linear-gradient(45deg, var(--kibali-blue), var(--kibali-green), var(--kibali-yellow));
            opacity: 0.8;
        }
        75% {
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-blue), var(--kibali-yellow));
            opacity: 0.7;
        }
        100% {
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-green), var(--kibali-blue));
            opacity: 0.5;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.set_page_config(
        page_title="🗺️ Kibali 🌟 - Assistant IA Avancé",
        page_icon="🌟",
        layout="wide"
    )

    # Logo Kibali animé - Grand K avec les couleurs vert, jaune, bleu
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <div class="kibali-k-logo">K</div>
        <div style="color: var(--kibali-text-secondary); font-size: 1.2rem; margin-top: 0.5rem; font-weight: 500;">
            Assistant IA Avancé
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Header principal avec design fluide
    st.markdown("""
    <div class="main-header">
        <h2 style="color: var(--kibali-green); margin: 0; text-align: center;">
            🌟 Assistant IA Avancé avec Recherche Web
        </h2>
        <p style="color: var(--kibali-text-secondary); text-align: center; margin: 1rem 0 0 0;">
            <strong>Nouvelles fonctionnalités:</strong><br>
            🌐 Recherche web intelligente • 🔗 Recherche hybride • 📰 Actualités temps réel<br>
            📄 Extraction de contenu web • 💾 Cache intelligent • 🗺️ Calcul de trajets OSM
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialisation des états
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'pois' not in st.session_state:
        st.session_state.pois = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_traj_info' not in st.session_state:
        st.session_state.last_traj_info = None
    if 'last_reponse' not in st.session_state:
        st.session_state.last_reponse = ""
    # Initialisation du modèle local
    if 'local_mode' not in st.session_state:
        st.session_state.local_mode = False
    if 'local_model_loaded' not in st.session_state:
        st.session_state.local_model_loaded = False
    if 'local_tokenizer' not in st.session_state:
        st.session_state.local_tokenizer = None
    if 'local_model' not in st.session_state:
        st.session_state.local_model = None
    if 'local_qwen_llm' not in st.session_state:
        st.session_state.local_qwen_llm = None
    # Initialisation du système d'outils dynamiques
    if 'tool_manager' not in st.session_state:
        if TOOLS_SYSTEM_AVAILABLE:
            try:
                st.session_state.tool_manager = ToolManager()
                print("✅ Gestionnaire d'outils initialisé")
            except Exception as e:
                print(f"⚠️ Erreur initialisation outils: {e}")
                st.session_state.tool_manager = None
        else:
            st.session_state.tool_manager = None

    # ===============================================
    # Onglets avec design fluide
    # ===============================================
    tab_names = ["⚙️ **Configuration**", "💬 **Chat RAG + Web**", "🗺️ **Trajets**", "📸 **Analyse Image**", "🌐 **Recherche Web**", "📷 **Photogrammétrie**"]
    if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager:
        tab_names.append("🔧 **Outils Dynamiques**")
    
    tabs = st.tabs(tab_names)
    
    # Assignation des onglets
    tab1, tab2, tab3, tab4, tab5, tab_photo = tabs[:6]
    if len(tabs) > 6:
        tab6 = tabs[6]

    # ===============================================
    # Onglet 1: Configuration avec cartes
    # ===============================================
    with tab1:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Gestion des données et configuration")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            pdf_files = st.file_uploader(
                "📤 **Upload PDFs**", 
                type=["pdf"], 
                accept_multiple_files=True,
                key="pdf_upload"
            )
            if pdf_files:
                saved_files = []
                for pdf_file in pdf_files:
                    filepath = os.path.join(PDFS_PATH, pdf_file.name)
                    with open(filepath, 'wb') as f:
                        f.write(pdf_file.getbuffer())
                    saved_files.append(pdf_file.name)
                st.success(f"✅ {len(saved_files)} PDFs uploadés")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            pbf_file = st.file_uploader(
                "📤 **Upload fichier OSM (.pbf)**", 
                type=["pbf"],
                key="pbf_upload"
            )
            if pbf_file:
                pbf_path = os.path.join(GRAPHS_PATH, pbf_file.name)
                with open(pbf_path, 'wb') as f:
                    f.write(pbf_file.getbuffer())
                G, pois, status = upload_and_process_pbf(pbf_path)
                st.session_state.graph = G
                st.session_state.pois = pois
                st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("🔄 **Traiter PDFs**", key="process_pdfs", help="Traite les PDFs uploadés pour créer la base vectorielle"):
                with st.spinner("Traitement des PDFs..."):
                    vectordb, status = process_pdfs()
                    st.session_state.vectordb = vectordb
                    st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("📂 **Charger graphe existant**", key="load_graph", help="Charge un graphe OSM existant"):
                with st.spinner("Chargement du graphe..."):
                    G, pois, status = load_existing_graph()
                    st.session_state.graph = G
                    st.session_state.pois = pois
                    st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("📂 **Charger base vectorielle**", key="load_vectordb", help="Charge la base vectorielle existante"):
                with st.spinner("Chargement de la base..."):
                    vectordb, status = load_vectordb()
                    st.session_state.vectordb = vectordb
                    st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        col6, col7 = st.columns(2)
        with col6:
            if st.button("🗑️ **Vider cache web**", key="clear_cache", help="Nettoie le cache web des anciennes recherches"):
                status = handle_clear_cache()
                st.success(status)
        
        with col7:
            if st.button("📊 **Voir statistiques**", key="show_stats", help="Affiche les statistiques détaillées du système"):
                stats = get_system_status()
                st.json(stats)
        st.markdown('</div>', unsafe_allow_html=True)

    # ===============================================
    # Onglet 2: Chat RAG Amélioré avec cartes
    # ===============================================
    with tab2:
        # Contrôles compacts dans un expander
        with st.expander("⚙️ Configuration", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                model_choice = st.selectbox(
                    "🎯 **Modèle IA**",
                    options=list(WORKING_MODELS.keys()),
                    index=0,
                    key="model_select",
                    help="Choisis le modèle d'IA pour tes réponses"
                )
            with col2:
                web_enabled = st.checkbox("🌐 **Recherche web activée**", value=True, help="Active la recherche web pour des réponses plus complètes")
                # Toggle pour le mode local
                local_mode_toggle = st.checkbox(
                    "🏠 **Mode Local (Qwen 1.5B)**", 
                    value=st.session_state.local_mode, 
                    help="Active le modèle local Qwen 1.5B pour les tâches complexes quand l'API est surchargée"
                )
                if local_mode_toggle != st.session_state.local_mode:
                    st.session_state.local_mode = local_mode_toggle
                    if local_mode_toggle and not st.session_state.local_model_loaded:
                        # Charger le modèle local
                        with st.spinner("🔄 Chargement du modèle local Qwen 1.5B..."):
                            try:
                                tokenizer, model, device, gpu_info = load_local_llm_model()
                                st.session_state.local_tokenizer = tokenizer
                                st.session_state.local_model = model
                                st.session_state.local_qwen_llm = QwenChatModel(tokenizer, model)
                                st.session_state.local_model_loaded = True
                                st.success(f"✅ Modèle local chargé sur {device.upper()}")
                            except Exception as e:
                                st.error(f"❌ Erreur chargement modèle local: {e}")
                                st.session_state.local_mode = False
                    st.rerun()
        
        # Analyser automatiquement les médias uploadés (avant le chat_input)
        media_analysis_results = []
        
        # Zone d'upload de médias compacte (style ChatGPT)
        st.markdown("**📎 Ajouter des médias:**")
        col_attach1, col_attach2, col_attach3 = st.columns([1, 1, 1])
        
        with col_attach1:
            uploaded_images = st.file_uploader(
                "📎", label_visibility="collapsed",
                type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
                accept_multiple_files=True,
                key="image_uploader",
                help="🖼️ Ajouter des images"
            )
        
        with col_attach2:
            uploaded_audios = st.file_uploader(
                "📎", label_visibility="collapsed",
                type=["mp3", "wav", "ogg", "m4a", "flac"],
                accept_multiple_files=True,
                key="audio_uploader",
                help="🎵 Ajouter des fichiers audio"
            )
        
        with col_attach3:
            uploaded_videos = st.file_uploader(
                "📎", label_visibility="collapsed",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                accept_multiple_files=True,
                key="video_uploader",
                help="🎥 Ajouter des vidéos"
            )
        
        # Initialiser le tracking des fichiers traités
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        
        # Variable pour tracker si de nouveaux médias ont été analysés
        new_media_analyzed = False
        
        # Analyse des images avec Vision AI (s'affiche dans le chat)
        if uploaded_images:
            # Ajouter l'analyse directement dans le chat
            for idx, img_file in enumerate(uploaded_images):
                # Vérifier si déjà traité
                file_key = f"img_{img_file.name}_{img_file.size}"
                if file_key in st.session_state.processed_files:
                    continue  # Skip si déjà traité
                
                # Marquer comme en cours de traitement
                st.session_state.processed_files.add(file_key)
                new_media_analyzed = True
                
                # Message utilisateur avec miniature
                img_file.seek(0)
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"📷 Image uploadée: {img_file.name}"
                })
                
                # Analyse en arrière-plan
                with st.spinner(f"🤖 Analyse de {img_file.name}..."):
                    try:
                        import tempfile
                        import base64
                        from io import BytesIO
                        
                        # Convertir l'image en base64 pour l'API
                        img_file.seek(0)
                        image_bytes = img_file.read()
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # Sauvegarder temporairement pour métadonnées
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_file.name)[1]) as tmp_file:
                            tmp_file.write(image_bytes)
                            tmp_path = tmp_file.name
                        
                        # Obtenir les métadonnées de base
                        from PIL import Image as PILImage
                        img = PILImage.open(tmp_path)
                        width, height = img.size
                        img_format = img.format
                        
                        # Analyse avec modèle de vision local (CLIP)
                        try:
                            # Charger les modèles locaux
                            if 'vision_models' not in st.session_state:
                                with st.spinner("📦 Chargement des modèles de vision locaux..."):
                                    st.session_state.vision_models = load_vision_models()
                            
                            # Charger l'OCR
                            if 'ocr_reader' not in st.session_state:
                                with st.spinner("📦 Chargement du modèle OCR..."):
                                    st.session_state.ocr_reader = load_ocr_reader()
                            
                            vision_success = False
                            image_caption = ""
                            analysis_details = []
                            extracted_text_info = ""
                            
                            # Extraction du texte avec OCR
                            if st.session_state.ocr_reader or pytesseract:
                                st.info("📝 Extraction du texte de l'image...")
                                extracted_texts = extract_text_from_image(tmp_path, st.session_state.ocr_reader)
                                if extracted_texts:
                                    extracted_text_info = organize_extracted_text(extracted_texts)
                                    st.success(f"✅ {len(extracted_texts)} éléments de texte détectés!")
                                else:
                                    extracted_text_info = "Aucun texte détecté dans l'image."
                            
                            # Analyse visuelle avec CLIP
                            if st.session_state.vision_models:
                                st.info("🔍 Analyse avec CLIP local...")
                                caption, details = analyze_image_with_clip(tmp_path, st.session_state.vision_models)
                                
                                if caption:
                                    image_caption = caption
                                    analysis_details = details
                                    vision_success = True
                                    st.success("✅ Analyse CLIP réussie!")
                                else:
                                    st.error(f"❌ Erreur: {details}")
                            else:
                                st.error("❌ Modèles de vision non disponibles")
                            
                            if vision_success:
                                # Enrichir avec recherche web sur le type d'image
                                try:
                                    web_results = enhanced_web_search(f"analyse détaillée de: {image_caption}", max_results=3)
                                    web_context = "\n\n".join([f"• {r.get('title', '')}: {r.get('body', '')[:200]}..." for r in web_results]) if web_results else ""
                                except:
                                    web_context = ""
                                
                                # Générer analyse complète avec LLM textuel
                                details_str = "\n".join([f"- {d['label']}: {d['confidence']:.1%}" for d in analysis_details]) if analysis_details else "Non disponibles"
                                
                                analysis_prompt = f"""Voici une image nommée "{img_file.name}" ({width}x{height}px, {img_format}).

📸 Analyse automatique (modèle CLIP local):
{image_caption}

🎯 Classifications détaillées:
{details_str}

📝 TEXTE EXTRAIT DE L'IMAGE (OCR):
{extracted_text_info}

🌐 Informations complémentaires du web:
{web_context if web_context else "Non disponibles"}

🎯 Ta mission: Fournis une analyse COMPLÈTE et DÉTAILLÉE comme ChatGPT:

1. **Description générale approfondie**:
   - Interprète ce que représente vraiment l'image
   - Donne le contexte général
   - Prends en compte le texte extrait pour enrichir ton analyse

2. **Analyse du texte détecté**:
   - Si du texte a été détecté, explique son contexte et sa signification
   - Identifie s'il s'agit de légendes, descriptions, titres, annotations, etc.
   - Relie le texte aux éléments visuels de l'image

3. **Éléments identifiables**:
   - Liste tous les objets, structures, éléments visibles
   - Identifie les détails importants
   - Corrèle avec le texte extrait si pertinent

4. **Analyse du contexte**:
   - Quel type d'image? (photo terrain, schéma technique, scan, graphique, document, etc.)
   - Où et quand pourrait-elle avoir été prise?
   - Le texte donne-t-il des indices supplémentaires?

5. **Analyse technique et scientifique**:
   - Si c'est une image géologique: identifie les roches, minéraux, structures
   - Si c'est technique: explique les éléments techniques
   - Si c'est un document: synthétise les informations textuelles
   - Donne des détails professionnels

6. **Applications pratiques**:
   - À quoi cette image peut-elle servir?
   - Quelles informations peut-on en extraire?
   - Comment le texte complète-t-il l'analyse visuelle?

7. **Observations spécifiques**:
   - Détails uniques ou remarquables
   - Éléments qui méritent attention
   - Cohérence entre le texte et l'image

Sois TRÈS précis, TRÈS détaillé et professionnel. Rédige au moins 250 mots."""

                                text_client = create_client()
                                analysis_response = text_client.chat.completions.create(
                                    model=WORKING_MODELS[model_choice],
                                    messages=[{"role": "user", "content": analysis_prompt}],
                                    max_tokens=1200,
                                    temperature=0.7
                                )
                                
                                enriched_analysis = analysis_response.choices[0].message.content
                                
                                # Préparer l'affichage avec le texte extrait
                                analysis_display = f"**🖼️ Analyse de {img_file.name}**\n\n📏 Résolution: {width}x{height}px | Format: {img_format}\n\n"
                                
                                # Ajouter le texte extrait si disponible
                                if extracted_text_info and "Aucun texte" not in extracted_text_info:
                                    analysis_display += f"## 📝 Texte Extrait:\n\n{extracted_text_info}\n\n---\n\n"
                                
                                analysis_display += f"## 🤖 Analyse IA Complète:\n\n{enriched_analysis}"
                                
                                # Ajouter l'analyse au chat
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": analysis_display
                                })
                                
                                media_analysis_results.append({
                                    'type': 'image',
                                    'name': img_file.name,
                                    'resolution': f"{width}x{height}",
                                    'format': img_format,
                                    'caption': image_caption,
                                    'ai_analysis': enriched_analysis,
                                    'web_context': web_context,
                                    'extracted_text': extracted_text_info
                                })
                            
                            else:
                                # Si aucun modèle n'a fonctionné
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"❌ Impossible d'analyser {img_file.name}. Vérifiez votre token HuggingFace ou réessayez plus tard."
                                })
                                
                        except Exception as analysis_error:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"❌ Erreur lors de l'analyse de {img_file.name}: {str(analysis_error)}"
                            })
                            
                    except Exception as img_error:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Erreur de traitement de {img_file.name}: {str(img_error)}"
                        })
        
        # Transcription des audios (affichée dans le chat)
        if uploaded_audios:
            for idx, audio_file in enumerate(uploaded_audios):
                # Vérifier si déjà traité
                file_key = f"audio_{audio_file.name}_{audio_file.size}"
                if file_key in st.session_state.processed_files:
                    continue
                
                st.session_state.processed_files.add(file_key)
                new_media_analyzed = True
                
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"🎵 Audio uploadé: {audio_file.name}"
                })
                
                with st.spinner(f"Analyse de {audio_file.name}..."):
                    try:
                        import tempfile
                        import librosa
                        
                        audio_file.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                            tmp_file.write(audio_file.read())
                            tmp_path = tmp_file.name
                        
                        y, sr = librosa.load(tmp_path)
                        duration = librosa.get_duration(y=y, sr=sr)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**🎵 Analyse de {audio_file.name}**\n\n⏱️ Durée: {duration:.2f}s | Fréquence: {sr} Hz\n\n🎙️ *Transcription: Utilisez Whisper API pour la transcription en production*"
                        })
                        
                        media_analysis_results.append({
                            'type': 'audio',
                            'name': audio_file.name,
                            'duration': duration,
                            'sample_rate': sr
                        })
                        
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Erreur lors de l'analyse de {audio_file.name}: {str(e)}"
                        })
        
        # Analyse des vidéos (affichée dans le chat)
        if uploaded_videos:
            for idx, video_file in enumerate(uploaded_videos):
                # Vérifier si déjà traité
                file_key = f"video_{video_file.name}_{video_file.size}"
                if file_key in st.session_state.processed_files:
                    continue
                
                st.session_state.processed_files.add(file_key)
                new_media_analyzed = True
                
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"🎥 Vidéo uploadée: {video_file.name}"
                })
                
                with st.spinner(f"Analyse de {video_file.name}..."):
                    try:
                        import tempfile
                        import cv2
                        
                        video_file.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_file:
                            tmp_file.write(video_file.read())
                            tmp_path = tmp_file.name
                        
                        cap = cv2.VideoCapture(tmp_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**🎥 Analyse de {video_file.name}**\n\n📐 Résolution: {width}x{height}px | ⏱️ Durée: {duration:.2f}s | 🎞️ FPS: {fps:.2f}\n\n🎬 *Analyse avancée: Extrayez les frames clés pour une analyse visuelle approfondie*"
                        })
                        
                        media_analysis_results.append({
                            'type': 'video',
                            'name': video_file.name,
                            'duration': duration,
                            'resolution': f"{width}x{height}",
                            'fps': fps
                        })
                        
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Erreur lors de l'analyse de {video_file.name}: {str(e)}"
                        })
        
        # Recharger uniquement si de nouveaux médias ont été analysés
        if 'new_media_analyzed' in locals() and new_media_analyzed:
            st.rerun()
        
        # Zone de chat avec design amélioré
        st.markdown('<div class="kibali-card" style="min-height: 400px;">', unsafe_allow_html=True)
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message-user"><strong>👤 Vous:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-assistant"><strong><span class="chat-k-logo">K</span> Kibali:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input de chat stylisé
        if prompt := st.chat_input("💭 Pose ta question ici...", key="chat_input"):
            # Enrichir le prompt avec les résultats d'analyse de médias IA si disponibles
            enriched_prompt = prompt
            if media_analysis_results:
                media_context = "\n\n📎 **Médias analysés avec IA:**\n"
                for media in media_analysis_results:
                    if media['type'] == 'image':
                        media_context += f"\n🖼️ **Image: {media['name']}**\n"
                        media_context += f"  📏 Résolution: {media.get('resolution', 'N/A')}\n"
                        media_context += f"  🎨 Format: {media.get('format', 'N/A')}\n"
                        
                        # Ajouter l'analyse IA complète
                        if 'ai_analysis' in media:
                            media_context += f"\n  🤖 **Analyse IA détaillée:**\n"
                            # Indenter l'analyse pour la lisibilité
                            ai_lines = media['ai_analysis'].split('\n')
                            for line in ai_lines[:15]:  # Limiter pour ne pas surcharger
                                media_context += f"  {line}\n"
                        
                        if 'caption' in media:
                            media_context += f"  📝 Description: {media['caption']}\n"
                        
                        if 'web_context' in media and media['web_context']:
                            media_context += f"  🌐 Infos complémentaires disponibles\n"
                    
                    elif media['type'] == 'audio':
                        media_context += f"\n🎵 **Audio: {media['name']}**\n"
                        media_context += f"  ⏱️ Durée: {media.get('duration', 0):.2f}s\n"
                        media_context += f"  📊 Fréquence: {media.get('sample_rate', 'N/A')} Hz\n"
                    
                    elif media['type'] == 'video':
                        media_context += f"\n🎥 **Vidéo: {media['name']}**\n"
                        media_context += f"  ⏱️ Durée: {media.get('duration', 0):.2f}s\n"
                        media_context += f"  📐 Résolution: {media.get('resolution', 'N/A')}\n"
                        media_context += f"  🎞️ FPS: {media.get('fps', 'N/A')}\n"
                
                enriched_prompt = f"{prompt}\n{media_context}"
            
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Animation de chargement
            with st.spinner("🤔 Kibali réfléchit..."):
                # Vérifier si on utilise le mode local
                if st.session_state.local_mode and st.session_state.local_model_loaded:
                    # Utiliser le modèle local Qwen
                    try:
                        # Recherche RAG si disponible
                        rag_context = ""
                        if st.session_state.vectordb:
                            rag_docs = rag_search(prompt, st.session_state.vectordb, k=3)
                            if rag_docs:
                                rag_context = "\n\n".join([f"Document: {doc.page_content[:500]}..." for doc in rag_docs])
                        
                        # Recherche web si activée
                        web_context = ""
                        if web_enabled:
                            try:
                                web_results = enhanced_web_search(prompt, max_results=3)
                                if web_results:
                                    web_context = "\n\n".join([f"Web: {r.get('title', '')} - {r.get('body', '')[:300]}" for r in web_results])
                            except Exception as e:
                                web_context = f"Erreur recherche web: {e}"
                        
                        # Construire le contexte enrichi avec médias
                        full_context = f"CONTEXTE DISPONIBLE:\n{rag_context}{web_context}\n\nQUESTION: {enriched_prompt}"
                        
                        # Générer avec le modèle local
                        response = st.session_state.local_qwen_llm._generate(
                            [{"role": "user", "content": full_context}], 
                            stop=None, 
                            run_manager=None
                        ).content
                        
                        # Ajouter mention du mode local
                        response = f"🏠 **Mode Local (Qwen 1.5B)** - Réponse générée localement\n\n{response}"
                        
                    except Exception as e:
                        response = f"❌ Erreur modèle local: {e}\n\nBasculement vers mode API..."
                        # Fallback vers mode API
                        st.session_state.local_mode = False
                        st.rerun()
                
                else:
                    # Mode API normal (code existant)
                    # Utilisation du système d'outils dynamiques si disponible
                    if st.session_state.tool_manager and TOOLS_SYSTEM_AVAILABLE:
                        try:
                            # Préparer le contexte pour les outils (avec médias analysés)
                            tool_context = {
                                'has_pdfs': st.session_state.vectordb is not None,
                                'vectordb_available': st.session_state.vectordb is not None,
                                'web_enabled': web_enabled,
                                'media_analysis': media_analysis_results if media_analysis_results else None
                            }
                            
                            # Analyse de la requête et sélection des outils appropriés
                            selected_tools = st.session_state.tool_manager.get_relevant_tools(enriched_prompt, tool_context)
                            
                            if selected_tools:
                                # Affichage popup élégant des outils sélectionnés
                                tools_names = [f"**{tool.name}**" for tool in selected_tools]
                                tools_descriptions = [tool.description for tool in selected_tools]
                                
                                st.markdown("""
                                <div style="
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 15px;
                                    border-radius: 10px;
                                    margin: 10px 0;
                                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                    border-left: 5px solid #FFD700;
                                ">
                                    <h4 style="color: white; margin: 0 0 10px 0;">
                                        🔧 Outils IA Détectés
                                    </h4>
                                    <p style="color: #f0f0f0; margin: 5px 0; font-size: 14px;">
                                        """ + " • ".join([tool.name for tool in selected_tools]) + """
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Recherche RAG si un outil PDF est détecté
                                rag_context = ""
                                pdf_tool_used = any('pdf' in tool.name.lower() or 'document' in tool.name.lower() for tool in selected_tools)
                                
                                if pdf_tool_used and st.session_state.vectordb:
                                    with st.expander("🔍 Recherche dans les documents PDF...", expanded=True):
                                        rag_docs = rag_search(prompt, st.session_state.vectordb, k=10)
                                        if rag_docs:
                                            rag_context = "\n\n".join([f"📄 {doc.metadata.get('source', 'Document')}:\n{doc.page_content}" for doc in rag_docs])
                                            st.success(f"✅ {len(rag_docs)} passages trouvés")
                                            # Afficher un aperçu
                                            for i, doc in enumerate(rag_docs[:3], 1):
                                                st.markdown(f"**Document {i}:** `{doc.metadata.get('source', 'Inconnu')}`")
                                        else:
                                            st.info("Aucun résultat dans les PDFs")
                                elif pdf_tool_used and not st.session_state.vectordb:
                                    st.warning("⚠️ Outil PDF détecté mais aucune base vectorielle chargée. Ajoutez des PDFs dans l'onglet Configuration.")
                                
                                # Exécution des autres outils avec affichage
                                tool_results = []
                                for tool in selected_tools:
                                    if 'pdf' not in tool.name.lower() and 'document' not in tool.name.lower():
                                        try:
                                            with st.expander(f"⚙️ Exécution de {tool.name}...", expanded=False):
                                                result = tool.execute(prompt, tool_context)
                                                st.json(result)
                                            tool_results.append(f"**{tool.name}:** {result}")
                                        except Exception as e:
                                            st.error(f"❌ Erreur {tool.name}: {str(e)}")
                                            tool_results.append(f"**{tool.name} (erreur):** {str(e)}")
                                
                                # Recherche web si activée
                                web_context = ""
                                if web_enabled:
                                    try:
                                        web_results = enhanced_web_search(prompt, max_results=3)
                                        if web_results:
                                            web_context = "\n\n".join([f"🌐 {r.get('title', '')}: {r.get('body', '')}" for r in web_results])
                                    except Exception as e:
                                        web_context = f"Erreur recherche web: {e}"
                                
                                # Construction du prompt final avec médias
                                full_context = ""
                                
                                # Ajouter contexte des médias analysés avec IA
                                if media_analysis_results:
                                    full_context += "═══════════════════════════════════════\n"
                                    full_context += "📎 MÉDIAS ANALYSÉS PAR IA AVANCÉE\n"
                                    full_context += "═══════════════════════════════════════\n\n"
                                    
                                    for media in media_analysis_results:
                                        if media['type'] == 'image':
                                            full_context += f"🖼️ **IMAGE: {media['name']}**\n"
                                            full_context += f"📏 Résolution: {media.get('resolution', 'N/A')}\n"
                                            full_context += f"🎨 Format: {media.get('format', 'N/A')}\n\n"
                                            
                                            if 'ai_analysis' in media:
                                                full_context += "🤖 ANALYSE IA DÉTAILLÉE:\n"
                                                full_context += f"{media['ai_analysis']}\n\n"
                                            
                                            if 'caption' in media:
                                                full_context += f"📝 Description: {media['caption']}\n\n"
                                            
                                            if 'web_context' in media and media['web_context']:
                                                full_context += "🌐 Informations complémentaires du web:\n"
                                                full_context += f"{media['web_context'][:500]}...\n\n"
                                        
                                        elif media['type'] == 'audio':
                                            full_context += f"🎵 **AUDIO: {media['name']}**\n"
                                            full_context += f"⏱️ Durée: {media.get('duration', 0):.2f}s\n"
                                            full_context += f"📊 Fréquence: {media.get('sample_rate', 'N/A')} Hz\n\n"
                                        
                                        elif media['type'] == 'video':
                                            full_context += f"🎥 **VIDÉO: {media['name']}**\n"
                                            full_context += f"📐 Résolution: {media.get('resolution', 'N/A')}\n"
                                            full_context += f"⏱️ Durée: {media.get('duration', 0):.2f}s\n"
                                            full_context += f"🎞️ FPS: {media.get('fps', 'N/A')}\n\n"
                                    
                                    full_context += "═══════════════════════════════════════\n\n"
                                
                                if rag_context:
                                    full_context += f"📚 DOCUMENTS PDF TROUVÉS:\n{rag_context}\n\n"
                                if tool_results:
                                    full_context += f"🔧 RÉSULTATS DES OUTILS:\n" + "\n\n".join(tool_results) + "\n\n"
                                if web_context:
                                    full_context += f"🌐 INFORMATIONS WEB:\n{web_context}\n\n"
                                
                                final_prompt = f"""Tu es Kibali, un assistant IA expert avec capacités d'analyse multimodale (images, texte, documents).

{full_context}

❓ QUESTION DE L'UTILISATEUR: {prompt}

📋 INSTRUCTIONS POUR TA RÉPONSE:

1. **Si une image a été analysée:**
   - Base-toi PRIORITAIREMENT sur l'analyse IA détaillée fournie ci-dessus
   - Réponds de manière précise et contextuelle en fonction de ce qui est visible dans l'image
   - Cite les éléments spécifiques identifiés par l'IA
   - Si l'image est technique/scientifique, utilise les informations web complémentaires

2. **Pour les autres médias:**
   - Intègre naturellement les informations d'audio/vidéo dans ta réponse
   - Mentionne les métadonnées pertinentes si nécessaire

3. **Pour les documents:**
   - Si des PDFs sont trouvés, cite les sources et extrais les informations clés
   - Combine les informations des médias avec celles des documents

4. **Style de réponse:**
   - Sois précis, professionnel et détaillé
   - Structure ta réponse avec des emojis appropriés
   - Ne mentionne PAS "je ne peux pas voir l'image" car l'analyse IA l'a déjà fait
   - Réponds comme si tu avais directement accès à l'image grâce à l'analyse fournie

🎯 RÉPONDS MAINTENANT:"""
                                
                                # Génération de la réponse finale
                                client = create_client()
                                messages = [{"role": "user", "content": final_prompt}]
                                response_obj = client.chat.completions.create(
                                    model=WORKING_MODELS[model_choice],
                                    messages=messages,
                                    max_tokens=800,
                                    temperature=0.3
                                )
                                response = response_obj.choices[0].message.content
                                
                            else:
                                # Aucun outil spécifique trouvé, utiliser l'approche classique
                                if not web_enabled:
                                    docs = rag_search(prompt, st.session_state.vectordb, k=3)
                                    response = generate_answer_enhanced(
                                        prompt, docs, WORKING_MODELS[model_choice], include_sources=True
                                    )
                                else:
                                    docs = hybrid_search_enhanced(prompt, st.session_state.vectordb, k=3, web_search_enabled=True)
                                    response = generate_answer_enhanced(
                                        prompt, docs, WORKING_MODELS[model_choice], include_sources=True
                                    )
                        
                        except Exception as e:
                            st.error(f"Erreur système d'outils: {e}")
                            # Fallback vers l'approche classique
                            if not web_enabled:
                                docs = rag_search(prompt, st.session_state.vectordb, k=3)
                                response = generate_answer_enhanced(
                                    prompt, docs, WORKING_MODELS[model_choice], include_sources=True
                                )
                            else:
                                docs = hybrid_search_enhanced(prompt, st.session_state.vectordb, k=3, web_search_enabled=True)
                                response = generate_answer_enhanced(
                                    prompt, docs, WORKING_MODELS[model_choice], include_sources=True
                                )
                    
                    else:
                        # Système d'outils non disponible, utiliser l'approche classique
                        if st.session_state.agent is None:
                            st.session_state.current_model = WORKING_MODELS[model_choice]
                            st.session_state.agent = create_enhanced_agent(
                                st.session_state.current_model, 
                                st.session_state.vectordb, 
                                st.session_state.graph, 
                                st.session_state.pois
                            )
                        
                        try:
                            if not web_enabled:
                                docs = rag_search(prompt, st.session_state.vectordb, k=3)
                                response = generate_answer_enhanced(
                                    prompt, docs, st.session_state.current_model, include_sources=True
                                )
                            else:
                                response = st.session_state.agent.run(prompt)
                        except Exception as e:
                            response = f"❌ Erreur: {e}\n\nTentative avec recherche locale..."
                            try:
                                docs = rag_search(prompt, st.session_state.vectordb, k=3)
                                response = generate_answer_enhanced(
                                    prompt, docs, st.session_state.current_model
                                )
                            except:
                                response = f"❌ Erreur complète: {e}"
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Boutons d'action stylisés
        col6, col7, col8 = st.columns(3)
        with col6:
            if st.button("🗑️ **Effacer conversation**", key="clear_chat", help="Supprime tout l'historique du chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col7:
            if st.button("💡 **Questions d'exemple**", key="show_examples", help="Affiche des exemples de questions"):
                st.markdown("""
                <div class="kibali-card">
                <h4>💡 Exemples de questions :</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                <strong>📚 Documents locaux :</strong><br>
                • "Résume le contenu de mes PDFs"<br>
                • "Informations principales dans mes documents"
                </div>
                <div>
                <strong>🌐 Avec recherche web :</strong><br>
                • "Dernières actualités en IA"<br>
                • "Compare mes documents avec tendances actuelles"
                </div>
                <div>
                <strong>🗺️ Navigation :</strong><br>
                • "Comment aller de l'école à l'hôpital ?"
                </div>
                <div>
                <strong>📸 Analyse :</strong><br>
                • "Analyse cette image"
                </div>
                </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col8:
            if st.button("📊 **Statut système**", key="system_status", help="Affiche l'état du système Kibali"):
                status = get_system_status()
                st.markdown(f"""
                <div class="kibali-card">
                <h4>📊 État du système</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div><strong>📁 PDFs:</strong> {status['counts']['pdfs']}</div>
                <div><strong>🗺️ Graphes:</strong> {status['counts']['graphs']}</div>
                <div><strong>💾 Base vectorielle:</strong> {'✅' if status['files']['vectordb'] else '❌'}</div>
                <div><strong>🌐 Cache web:</strong> {'✅' if status['files']['web_cache'] else '❌'}</div>
                </div>
                </div>
                """, unsafe_allow_html=True)

    # ===============================================
    # Onglet 3: Trajets avec cartes
    # ===============================================
    with tab3:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🗺️ Calcul de trajets
        **Exemples:** "Comment aller de l'école à l'hôpital ?"
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        trajectory_question = st.text_area(
            "🗺️ **Question de trajet**",
            placeholder="Ex: Comment aller de l'école à l'hôpital ?",
            height=100,
            help="Décris ton trajet en langage naturel"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_traj1, col_traj2 = st.columns([1, 1])
        with col_traj1:
            if st.button("🚀 **Calculer trajet**", key="calculate_route", help="Calcule l'itinéraire demandé"):
                if not st.session_state.graph or not st.session_state.pois:
                    st.error("❌ Graphe non chargé - Upload un fichier OSM d'abord")
                else:
                    with st.spinner("🗺️ Calcul du trajet en cours..."):
                        carte_buf, reponse, traj_info = calculer_trajet(
                            trajectory_question, 
                            st.session_state.graph, 
                            st.session_state.pois
                        )
                        if carte_buf:
                            carte_buf.seek(0)
                            image = Image.open(carte_buf)
                            st.markdown('<div class="kibali-card zoom-image">', unsafe_allow_html=True)
                            st.image(image, caption="🗺️ Carte du trajet calculé", use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.session_state.last_traj_info = traj_info
                        st.session_state.last_reponse = reponse
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.text_area("📋 **Détails du trajet**", value=st.session_state.last_reponse, height=200, disabled=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with col_traj2:
            if st.button("💾 **Sauvegarder trajet**", key="save_trajectory", help="Sauvegarde ce trajet dans la base de connaissances"):
                if st.session_state.last_traj_info:
                    status = save_trajectory(
                        trajectory_question, 
                        st.session_state.last_reponse, 
                        st.session_state.last_traj_info
                    )
                    st.success(f"✅ Trajet sauvegardé: {status}")
                else:
                    st.warning("⚠️ Aucun trajet à sauvegarder - Calcule un trajet d'abord")

    # ===============================================
    # Onglet 4: Analyse Image avec IA avancée
    # ===============================================
    with tab4:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📸 Analyse d'images avancée avec IA
        Upload une image pour analyse complète: Vision AI (CLIP), OCR, détection d'objets, annotations et graphiques.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "📤 **Upload Image**", 
            type=["jpg", "png", "jpeg"],
            key="image_upload",
            help="Formats supportés: JPG, PNG, JPEG"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 **Analyser l'image**", key="analyze_image", help="Lance l'analyse complète de l'image avec IA"):
            if uploaded_image:
                with st.spinner("🔬 Analyse IA en cours..."):
                    # 1. Analyse traditionnelle (OpenCV)
                    analysis_data, proc_images, tables_str = process_image(uploaded_image.getvalue())
                    
                    # 2. Analyse avancée avec Vision AI (CLIP) + OCR
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 🤖 Analyse Vision AI + OCR")
                    
                    try:
                        # Charger les modèles Vision
                        with st.spinner("📥 Chargement des modèles Vision AI..."):
                            clip_model, clip_processor = load_vision_models()
                        
                        # Sauvegarder l'image temporairement
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(uploaded_image.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Charger le reader OCR
                        with st.spinner("📝 Chargement du moteur OCR..."):
                            ocr_reader = load_ocr_reader()
                        
                        # Extraire le texte (OCR)
                        with st.spinner("🔍 Extraction du texte..."):
                            extracted_texts = extract_text_from_image(tmp_path, ocr_reader)
                        
                        # Afficher le texte extrait
                        if extracted_texts:
                            st.markdown("#### 📝 Texte extrait de l'image")
                            organized_text = organize_extracted_text(extracted_texts)
                            st.text_area("Texte détecté", value=organized_text, height=200, disabled=True)
                        else:
                            st.info("ℹ️ Aucun texte détecté dans l'image")
                        
                        # Analyser avec CLIP
                        with st.spinner("🎨 Analyse sémantique (CLIP)..."):
                            image_pil = Image.open(tmp_path)
                            clip_analysis = analyze_image_with_clip(image_pil, clip_model, clip_processor)
                        
                        # Afficher les résultats CLIP
                        st.markdown("#### 🎯 Analyse sémantique (catégories détectées)")
                        
                        # Créer un DataFrame pour les catégories
                        import pandas as pd
                        categories_df = pd.DataFrame({
                            'Catégorie': list(clip_analysis.keys()),
                            'Score de confiance': [f"{v*100:.1f}%" for v in clip_analysis.values()]
                        })
                        st.dataframe(categories_df, use_container_width=True)
                        
                        # Visualisation graphique
                        fig, ax = plt.subplots(figsize=(10, 6))
                        categories = list(clip_analysis.keys())
                        scores = [v * 100 for v in clip_analysis.values()]
                        colors = ['#2E7D32' if s > 50 else '#FFA726' if s > 30 else '#EF5350' for s in scores]
                        ax.barh(categories, scores, color=colors)
                        ax.set_xlabel('Score de confiance (%)')
                        ax.set_title('Analyse sémantique de l\'image')
                        ax.set_xlim(0, 100)
                        st.pyplot(fig)
                        
                        # Nettoyer le fichier temporaire
                        import os
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse Vision AI: {str(e)}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 3. Afficher les images traditionnelles avec zoom au survol
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Analyses traditionnelles (OpenCV)")
                    if proc_images:
                        cols = st.columns(min(len(proc_images), 3))
                        for i, (img, caption) in enumerate(zip(proc_images, ['Image Originale'] + ['Analyse'] * (len(proc_images)-1))):
                            with cols[i % len(cols)]:
                                st.markdown('<div class="zoom-image">', unsafe_allow_html=True)
                                st.image(img, caption=f"📸 {caption}", use_column_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 4. Afficher les tableaux d'analyse
                    if tables_str:
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.markdown(tables_str, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 5. Génération de rapport complet avec LLM
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 🤖 Rapport d'analyse IA complet")
                    
                    with st.spinner("✍️ Génération du rapport détaillé..."):
                        # Combiner toutes les informations
                        combined_info = f"""
Analyse de l'image uploadée:

### Vision AI (CLIP):
{chr(10).join([f"- {cat}: {score*100:.1f}%" for cat, score in clip_analysis.items()])}

### Texte extrait (OCR):
{organized_text if extracted_texts else "Aucun texte détecté"}

### Analyses techniques (OpenCV):
{analysis_data}
"""
                        
                        improved_analysis = improve_analysis_with_llm(combined_info, st.session_state.current_model)
                        st.text_area("📋 Rapport complet", value=improved_analysis, height=400, disabled=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Veuillez uploader une image d'abord")

    # ===============================================
    # Onglet 5: Recherche Web Avancée avec cartes
    # ===============================================
    with tab5:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🌐 Recherche web avancée avec extraction de contenu
        Recherche intelligente sur Internet avec cache automatique et extraction de contenu détaillé.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col8, col9 = st.columns([3, 1])
        with col8:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            web_query = st.text_area(
                "🔍 **Requête de recherche**",
                placeholder="Ex: actualités intelligence artificielle 2024",
                height=100,
                help="Décris ta recherche en langage naturel"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col9:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            search_type = st.selectbox(
                "🎯 **Type de recherche**",
                options=["text", "news", "both"],
                index=2,
                help="Texte: recherche générale, News: actualités, Both: les deux"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        col_search1, col_search2 = st.columns([1, 1])
        with col_search1:
            if st.button("🔍 **Rechercher**", key="web_search", help="Lance la recherche web"):
                if web_query.strip():
                    with st.spinner("🌐 Recherche en cours..."):
                        results = enhanced_web_search(web_query, max_results=10, search_type=search_type)
                        if results:
                            for result in results:
                                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                st.markdown(f"### 📰 {result.get('title', 'Sans titre')}")
                                st.write(f"📝 {result.get('body', 'Pas de description')}")
                                url = result.get('href') or result.get('url', '#')
                                st.markdown(f"[🔗 **Accéder à la source**]({url})")
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown("---")
                        else:
                            st.error("❌ Aucun résultat trouvé")
                else:
                    st.warning("⚠️ Veuillez entrer une requête de recherche")
        
        with col_search2:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            url_extract = st.text_input(
                "🌐 **URL à extraire**",
                placeholder="https://example.com/article",
                help="Colle une URL pour extraire son contenu détaillé"
            )
            if st.button("📄 **Extraire contenu**", key="extract_content", help="Extrait le contenu détaillé de la page web"):
                if url_extract.strip():
                    if not url_extract.startswith(('http://', 'https://')):
                        url_extract = 'https://' + url_extract
                    with st.spinner("📄 Extraction en cours..."):
                        content = smart_content_extraction(url_extract, max_length=2000)
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.text_area("📄 **Contenu extrait**", value=content, height=300, disabled=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Veuillez entrer une URL valide")
            st.markdown('</div>', unsafe_allow_html=True)

    # ===============================================
    # Onglet 6: Photogrammétrie - Classification Intelligente avec IA
    # ===============================================
    with tab_photo:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📷 Classification Intelligente pour Photogrammétrie avec Vision AI
        **L'IA analyse visuellement chaque photo et les ordonne intelligemment pour une reconstruction 3D optimale.**
        
        🤖 **Analyse IA:**
        - Vision sémantique (CLIP) - Compréhension du contenu
        - Analyse de textures et détails visuels
        - Détection d'angles de vue similaires
        - Ordonnancement optimal pour Dust3R/MeshRoom
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Deux modes: Classification IA ou Optimisation simple
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        mode = st.radio(
            "🎯 Mode de traitement:",
            ["🤖 Classification IA (Analyse visuelle + Ordonnancement)", "⚡ Optimisation Rapide (Réduction simple)"],
            help="Classification IA analyse visuellement chaque photo, Optimisation Rapide réduit juste le nombre"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section d'upload multiple
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Upload de photos")
        
        uploaded_photos = st.file_uploader(
            "Sélectionnez vos photos de photogrammétrie",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
            accept_multiple_files=True,
            help="Uploadez toutes vos photos. L'IA les analysera et les classera intelligemment!",
            key="photo_upload_classifier"
        )
        
        if uploaded_photos and len(uploaded_photos) > 0:
            st.success(f"✅ {len(uploaded_photos)} photos chargées")
            
            if mode.startswith("🤖"):
                # MODE CLASSIFICATION IA
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                st.markdown("#### 🎨 Paramètres de classification IA")
                
                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    ordering_method = st.selectbox(
                        "📐 Méthode d'ordonnancement",
                        ["sequential", "cluster"],
                        format_func=lambda x: "Séquentiel (photos qui se suivent)" if x == "sequential" else "Par groupes d'angles",
                        help="Séquentiel: crée une chaîne d'images similaires. Groupes: classe par angles puis ordonne."
                    )
                
                with col_ai2:
                    generate_viz = st.checkbox(
                        "📊 Générer visualisation",
                        value=True,
                        help="Crée une grille visuelle montrant l'ordre des photos"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Bouton de classification IA
                if st.button("🚀 **Analyser et Classer avec l'IA**", type="primary", use_container_width=True):
                    with st.spinner("🤖 Vision AI en cours d'analyse..."):
                        try:
                            import tempfile
                            import shutil
                            from pathlib import Path
                            from outils.intelligent_photo_classifier import classify_photos_with_ai
                            
                            # Créer dossier temporaire
                            temp_dir = tempfile.mkdtemp(prefix="ai_photo_classifier_")
                            
                            # Sauvegarder les photos
                            st.info(f"💾 Sauvegarde de {len(uploaded_photos)} photos...")
                            photo_paths = []
                            for photo in uploaded_photos:
                                photo_path = Path(temp_dir) / photo.name
                                with open(photo_path, 'wb') as f:
                                    f.write(photo.getbuffer())
                                photo_paths.append(str(photo_path))
                            
                            # Charger le modèle Vision AI (CLIP)
                            st.info("📦 Chargement du modèle Vision AI (CLIP)...")
                            clip_model, clip_processor = load_vision_models()
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            
                            # Classifier avec l'IA
                            st.info(f"🔍 Analyse IA de {len(photo_paths)} photos...")
                            output_dir = temp_dir + "_classified"
                            Path(output_dir).mkdir(exist_ok=True)
                            
                            ordered_paths, report, viz_path = classify_photos_with_ai(
                                photo_paths,
                                clip_model['clip_model'],
                                clip_model['clip_processor'],
                                device=clip_model['device'],
                                method=ordering_method,
                                output_dir=output_dir if generate_viz else None
                            )
                            
                            # Afficher les résultats
                            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                            st.markdown("### 🎯 Résultats de la Classification IA")
                            st.text_area("📋 Rapport détaillé", value=report, height=400, disabled=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Afficher la visualisation
                            if viz_path and Path(viz_path).exists():
                                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                st.markdown("### 📊 Visualisation de l'ordre")
                                st.image(viz_path, caption="Ordre optimisé des photos (gauche → droite, haut → bas)", use_column_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Copier les photos dans l'ordre
                            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                            st.markdown("### 📥 Photos classées et renommées")
                            
                            ordered_dir = Path(output_dir) / "ordered_photos"
                            ordered_dir.mkdir(exist_ok=True)
                            
                            for idx, src_path in enumerate(ordered_paths, 1):
                                src = Path(src_path)
                                # Renommer avec numéro d'ordre + nom original
                                dest = ordered_dir / f"{idx:04d}_{src.name}"
                                shutil.copy2(src, dest)
                            
                            st.success(f"✅ {len(ordered_paths)} photos ordonnées et sauvegardées")
                            
                            # Créer ZIP pour téléchargement
                            import zipfile
                            zip_path = Path(temp_dir) / "photos_classified_ai.zip"
                            
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                # Ajouter les photos ordonnées
                                for photo_file in ordered_dir.glob("*.*"):
                                    zipf.write(photo_file, arcname=f"ordered_photos/{photo_file.name}")
                                
                                # Ajouter le rapport
                                report_path = Path(output_dir) / "classification_report.txt"
                                if report_path.exists():
                                    zipf.write(report_path, arcname="classification_report.txt")
                                
                                # Ajouter la visualisation
                                if viz_path and Path(viz_path).exists():
                                    zipf.write(viz_path, arcname="ordering_visualization.png")
                            
                            # Bouton de téléchargement
                            with open(zip_path, 'rb') as f:
                                st.download_button(
                                    label="📦 Télécharger les photos classées (ZIP)",
                                    data=f.read(),
                                    file_name="photos_classees_ia.zip",
                                    mime="application/zip"
                                )
                            
                            st.info("💡 **Prêt pour Dust3R:** Utilisez les photos dans l'ordre 0001, 0002, 0003... pour une reconstruction optimale!")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la classification: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
            
            else:
                # MODE OPTIMISATION RAPIDE (ancien système)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                st.markdown("#### ⚙️ Paramètres d'optimisation")
                
                col_param1, col_param2 = st.columns(2)
            with col_param1:
                target_count = st.number_input(
                    "🎯 Nombre cible de photos (0 = automatique)",
                    min_value=0,
                    max_value=len(uploaded_photos),
                    value=0,
                    help="Laissez 0 pour une détection automatique du nombre optimal"
                )
            
            with col_param2:
                coverage_threshold = st.slider(
                    "📊 Couverture minimale requise",
                    min_value=0.8,
                    max_value=1.0,
                    value=0.95,
                    step=0.05,
                    help="Pourcentage de la scène qui doit être couvert (0.95 = 95%)"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Bouton d'optimisation
            if st.button("🚀 **Optimiser le dataset**", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyse des photos en cours..."):
                    try:
                        import tempfile
                        import shutil
                        from pathlib import Path
                        
                        # Créer un dossier temporaire pour les photos
                        temp_dir = tempfile.mkdtemp(prefix="photogrammetry_")
                        
                        # Sauvegarder les photos uploadées
                        st.info(f"💾 Sauvegarde de {len(uploaded_photos)} photos...")
                        for idx, photo in enumerate(uploaded_photos):
                            photo_path = Path(temp_dir) / photo.name
                            with open(photo_path, 'wb') as f:
                                f.write(photo.getbuffer())
                        
                        # Importer l'outil
                        from outils.photogrammetry_optimizer_tool import PhotogrammetryOptimizerTool
                        
                        # Exécuter l'optimisation
                        tool = PhotogrammetryOptimizerTool()
                        context = {
                            'input_folder': temp_dir,
                            'target_count': target_count if target_count > 0 else None,
                            'coverage_threshold': coverage_threshold,
                            'similarity_threshold': 0.85
                        }
                        
                        result = tool.execute("", context=context)
                        
                        # Afficher les résultats
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.markdown("### 📊 Résultats de l'optimisation")
                        st.text(result)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Proposer le téléchargement des photos sélectionnées
                        output_folder = Path(temp_dir + "_optimized")
                        if output_folder.exists():
                            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                            st.markdown("### 📥 Téléchargement des photos optimisées")
                            
                            # Créer un ZIP
                            import zipfile
                            zip_path = Path(temp_dir) / "photos_optimized.zip"
                            
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for photo_file in output_folder.glob("*.*"):
                                    if photo_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                                        zipf.write(photo_file, arcname=photo_file.name)
                            
                            # Bouton de téléchargement
                            with open(zip_path, 'rb') as f:
                                st.download_button(
                                    label="📦 Télécharger les photos sélectionnées (ZIP)",
                                    data=f.read(),
                                    file_name="photos_photogrammetrie_optimisees.zip",
                                    mime="application/zip"
                                )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Nettoyage (optionnel, garder pour debug)
                        # shutil.rmtree(temp_dir, ignore_errors=True)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'optimisation: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
        
        else:
            st.info("👆 Uploadez vos photos pour commencer l'optimisation")
            
            # Exemples d'utilisation
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("""
            #### 💡 Exemples d'utilisation
            
            **Photogrammétrie aérienne:**
            - Upload: 500-1000 photos drone
            - Résultat: 15-30 photos essentielles
            - Réduction: ~97%
            
            **Scan d'objet 3D:**
            - Upload: 200 photos multi-angles
            - Résultat: 10-20 photos représentatives
            - Réduction: ~90%
            
            **Cartographie terrain:**
            - Upload: 1000+ photos aériennes
            - Résultat: 20-50 photos de couverture
            - Réduction: ~95%
            
            #### 🎯 Algorithme
            1. **Extraction de features** (ORB, histogrammes, textures)
            2. **Clustering intelligent** (KMeans sur angles de vue)
            3. **Sélection optimale** (représentants par cluster)
            4. **Vérification de couverture** (ajout si zones manquantes)
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # ===============================================
    # Onglet 7: Outils Dynamiques (si disponible)
    # ===============================================
    if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager and len(tabs) > 6:
        with tab6:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 🔧 Système d'outils dynamiques
            **Outils disponibles:** Recherche web, analyse d'images, calculs mathématiques, génération de code, traduction, gestion de fichiers.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Affichage des outils disponibles
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Outils chargés")
            tools = st.session_state.tool_manager.tools
            if tools:
                tool_data = []
                for tool in tools:
                    tool_data.append({
                        "Nom": tool.name,
                        "Description": tool.description,
                        "Capacités": ", ".join(tool.capabilities)
                    })
                st.dataframe(pd.DataFrame(tool_data), use_container_width=True)
            else:
                st.warning("⚠️ Aucun outil chargé")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Test manuel d'un outil
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("#### 🧪 Test d'outil manuel")
            
            col_test1, col_test2 = st.columns([2, 1])
            with col_test1:
                test_query = st.text_area(
                    "🔍 **Requête de test**",
                    placeholder="Ex: Analyse cette image de sol minier",
                    height=100,
                    help="Décris la tâche à effectuer"
                )
            
            with col_test2:
                available_tools = [tool.name for tool in tools] if tools else []
                selected_tool_name = st.selectbox(
                    "🔧 **Outil à tester**",
                    options=["Auto (détection)"] + available_tools,
                    help="Choisis un outil spécifique ou laisse la détection automatique"
                )
            
            if st.button("🚀 **Tester l'outil**", key="test_tool", help="Exécute l'outil sélectionné"):
                if test_query.strip():
                    with st.spinner("🔧 Exécution de l'outil..."):
                        try:
                            if selected_tool_name == "Auto (détection)":
                                # Préparer le contexte pour la détection
                                tool_context = {
                                    'has_pdfs': st.session_state.vectordb is not None,
                                    'vectordb_available': st.session_state.vectordb is not None
                                }
                                
                                # Détection automatique
                                relevant_tools = st.session_state.tool_manager.get_relevant_tools(test_query, tool_context)
                                if relevant_tools:
                                    st.success(f"🔍 Outils détectés: {', '.join([t.name for t in relevant_tools])}")
                                    
                                    # Si outil PDF détecté, faire une vraie recherche
                                    pdf_tool_detected = any('pdf' in t.name.lower() or 'document' in t.name.lower() for t in relevant_tools)
                                    
                                    if pdf_tool_detected:
                                        if st.session_state.vectordb:
                                            st.info("🔍 Recherche dans les documents PDF...")
                                            rag_docs = rag_search(test_query, st.session_state.vectordb, k=5)
                                            if rag_docs:
                                                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                                st.markdown(f"### 📊 Résultats trouvés: {len(rag_docs)} passages")
                                                for i, doc in enumerate(rag_docs, 1):
                                                    st.markdown(f"**Document {i}:** {doc.metadata.get('source', 'Inconnu')}")
                                                    st.text_area(f"Extrait {i}", value=doc.page_content[:500], height=100, disabled=True, key=f"extract_{i}")
                                                    st.markdown("---")
                                                st.markdown('</div>', unsafe_allow_html=True)
                                            else:
                                                st.warning("Aucun résultat trouvé dans les documents")
                                        else:
                                            st.warning("⚠️ Base vectorielle non chargée. Ajoutez des PDFs dans l'onglet Configuration.")
                                    
                                    # Exécuter les autres outils
                                    results = []
                                    for tool in relevant_tools:
                                        if 'pdf' not in tool.name.lower() and 'document' not in tool.name.lower():
                                            result = tool.execute(test_query, tool_context)
                                            results.append(f"**{tool.name}:**\n{result}")
                                    
                                    if results:
                                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                        st.markdown("### 📊 Résultats des autres outils")
                                        for result in results:
                                            st.markdown(result)
                                            st.markdown("---")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.warning("⚠️ Aucun outil pertinent détecté pour cette requête")
                            
                            else:
                                # Outil spécifique
                                tool = next((t for t in tools if t.name == selected_tool_name), None)
                                if tool:
                                    result = tool.execute(test_query)
                                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                    st.markdown(f"### 📊 Résultat de {tool.name}")
                                    st.text_area("Résultat", value=str(result), height=200, disabled=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.error("❌ Outil non trouvé")
                        
                        except Exception as e:
                            st.error(f"❌ Erreur lors du test: {e}")
                else:
                    st.warning("⚠️ Veuillez entrer une requête de test")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistiques des outils
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Statistiques du système d'outils")
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Outils chargés", len(tools) if tools else 0)
            
            with col_stats2:
                total_capabilities = sum(len(tool.capabilities) for tool in tools) if tools else 0
                st.metric("Capacités totales", total_capabilities)
            
            with col_stats3:
                status = "✅ Actif" if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager else "❌ Inactif"
                st.metric("Statut système", status)
            
            st.markdown('</div>', unsafe_allow_html=True)

# ===============================================
# Lancement principal
# ===============================================
if __name__ == "__main__":
    print("🚀 Kibali 🌟 - Assistant IA Avancé avec Recherche Web")
    print("=" * 60)
    # Initialisation
    setup_drive()
    print(f"📁 Dossier unifié: {CHATBOT_DIR}")
    print(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")
    print(f"🌐 Recherche web intégrée")
    # Vérifier les fichiers existants
    existing_graphs = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
    existing_pdfs = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
    print(f"📊 État initial:")
    print(f" 🗺️ Graphes OSM: {len(existing_graphs)}")
    print(f" 📄 PDFs: {len(existing_pdfs)}")
    print(f" 💾 Base vectorielle: {'✅' if os.path.exists(VECTORDB_PATH) else '⚠️ À créer (upload PDFs + traiter)'}")
    print(f" 🌐 Cache web: {'✅' if os.path.exists(WEB_CACHE_PATH) else '📝 Sera créé automatiquement'}")
    # Afficher les statistiques du cache si disponible
    print(f" 📈 {get_cache_stats()}")
    # Lancer l'interface Streamlit
    print("\n🌟 Lancement de l'interface Streamlit...")
    print("📝 Fonctionnalités disponibles:")
    print(" 💬 Chat RAG avec recherche web intelligent")
    print(" 🗺️ Calcul de trajets OSM")
    print(" 📸 Analyse d'images avancée")
    print(" 🌐 Recherche web avec extraction de contenu")
    print(" 💾 Cache intelligent pour optimiser les performances")
    print(" 📊 Gestion unifiée des fichiers")
    main()
else:
    print("📦 Module Kibali importé avec succès")
    print("🔧 Fonctions disponibles:")
    print(" - main() : Interface Streamlit complète")
    print(" - enhanced_web_search() : Recherche web avancée")
    print(" - hybrid_search_enhanced() : Recherche hybride")
    print(" - smart_content_extraction() : Extraction de contenu web")
    print(" - process_image() : Analyse d'images")
    print(" - calculer_trajet() : Calcul de trajets OSM")
# ===============================================
# Fonctions utilitaires supplémentaires
# ===============================================
def cleanup_old_cache():
    """Nettoie les entrées expirées du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        original_count = len(cache)
        cleaned_cache = {}
        for key, entry in cache.items():
            if not is_cache_expired(entry):
                cleaned_cache[key] = entry
        save_web_cache(cleaned_cache)
        removed_count = original_count - len(cleaned_cache)
        return f"✅ Cache nettoyé: {removed_count} entrées expirées supprimées, {len(cleaned_cache)} conservées"
    except Exception as e:
        return f"❌ Erreur nettoyage cache: {e}"
def export_system_config():
    """Exporte la configuration système pour debug"""
    config = {
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "chatbot_dir": CHATBOT_DIR,
            "vectordb_path": VECTORDB_PATH,
            "pdfs_path": PDFS_PATH,
            "graphs_path": GRAPHS_PATH,
            "maps_path": MAPS_PATH
        },
        "models": WORKING_MODELS,
        "status": get_system_status(),
        "features": {
            "web_search": True,
            "osm_routing": True,
            "image_analysis": True,
            "pdf_processing": True,
            "caching": True
        }
    }
    config_path = os.path.join(CHATBOT_DIR, "system_config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return f"✅ Configuration exportée: {config_path}"
    except Exception as e:
        return f"❌ Erreur export: {e}"
def test_all_features():
    """Teste toutes les fonctionnalités principales"""
    results = {}
    # Test HuggingFace
    results["huggingface"] = test_hf_connection()
    # Test recherche web
    try:
        test_results = enhanced_web_search("test", max_results=1)
        results["web_search"] = len(test_results) > 0
    except:
        results["web_search"] = False
    # Test recherche web
    results["specialized_models"] = {}
    for model_name, model in SPECIALIZED_MODELS.items():
        results["specialized_models"][model_name] = model is not None
    # Test base vectorielle
    try:
        vectordb, _ = load_vectordb()
        results["vectordb"] = vectordb is not None
    except:
        results["vectordb"] = False
    # Test graphe OSM
    try:
        graph, pois, _ = load_existing_graph()
        results["osm_graph"] = graph is not None
    except:
        results["osm_graph"] = False
    return results
# ===============================================
# Fonctions de maintenance avancées
# ===============================================
def optimize_vectordb():
    """Optimise la base vectorielle en supprimant les doublons"""
    try:
        vectordb, status = load_vectordb()
        if not vectordb:
            return "❌ Aucune base vectorielle à optimiser"
        # Cette fonction nécessiterait une implémentation plus complexe
        # pour détecter et supprimer les doublons dans FAISS
        return "✅ Base vectorielle optimisée (fonctionnalité à implémenter)"
    except Exception as e:
        return f"❌ Erreur optimisation: {e}"
def backup_all_data():
    """Crée une sauvegarde de toutes les données"""
    try:
        import shutil
        import zipfile
        backup_name = f"kibali_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        backup_path = os.path.join(CHATBOT_DIR, backup_name)
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            # Sauvegarder tous les fichiers du dossier chatbot
            for root, dirs, files in os.walk(CHATBOT_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, CHATBOT_DIR)
                    backup_zip.write(file_path, arcname)
        return f"✅ Sauvegarde créée: {backup_path}"
    except Exception as e:
        return f"❌ Erreur sauvegarde: {e}"
def restore_from_backup(backup_path):
    """Restaure les données depuis une sauvegarde"""
    try:
        import zipfile
        if not os.path.exists(backup_path):
            return "❌ Fichier de sauvegarde non trouvé"
        with zipfile.ZipFile(backup_path, 'r') as backup_zip:
            backup_zip.extractall(CHATBOT_DIR)
        return f"✅ Données restaurées depuis: {backup_path}"
    except Exception as e:
        return f"❌ Erreur restauration: {e}"
# ===============================================
# Version API pour utilisation externe
# ===============================================
class KibaliAPI:
    """API simplifiée pour utiliser Kibali depuis du code externe"""
    def __init__(self):
        self.vectordb = None
        self.graph = None
        self.pois = []
        self.client = None
        self.model_name = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
        # Initialisation automatique
        self._initialize()
    def _initialize(self):
        """Initialisation automatique"""
        try:
            setup_drive()
            self.vectordb, _ = load_vectordb()
            self.graph, self.pois, _ = load_existing_graph()
            self.client = create_client()
        except Exception as e:
            print(f"⚠️ Initialisation partielle: {e}")
    def ask(self, question, use_web=True):
        """Pose une question simple"""
        try:
            if use_web:
                docs = hybrid_search_enhanced(question, self.vectordb, web_search_enabled=True)
            else:
                docs = rag_search(question, self.vectordb)
            return generate_answer_enhanced(question, docs, self.model_name)
        except Exception as e:
            return f"❌ Erreur: {e}"
    def search_web(self, query, max_results=5):
        """Recherche web simple"""
        try:
            results = enhanced_web_search(query, max_results)
            return [{"title": r.get("title"), "url": r.get("href", r.get("url")), "snippet": r.get("body")} for r in results]
        except Exception as e:
            return [{"error": str(e)}]
    def calculate_route(self, from_place, to_place):
        """Calcule un itinéraire"""
        try:
            question = f"Comment aller de {from_place} à {to_place}"
            _, response, info = calculer_trajet(question, self.graph, self.pois)
            return {"response": response, "info": info}
        except Exception as e:
            return {"error": str(e)}
    def get_status(self):
        """Retourne le statut du système"""
        return get_system_status()
# Instance globale de l'API
kibali_api = KibaliAPI()
# ===============================================
# Messages de fin et documentation
# ===============================================
print("\n" + "="*60)
print("🎉 KIBALI 🌟 - SYSTÈME CHARGÉ AVEC SUCCÈS")
print("="*60)
print(f"📅 Version: 2.0.0 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔑 Token HF: {'✅ Configuré' if HF_TOKEN else '❌ Manquant'}")
print(f"📁 Dossier: {CHATBOT_DIR}")
print(f"🌐 Recherche web: ✅ Activée")
print(f"💾 Cache intelligent: ✅ Activé")
print("\n📚 FONCTIONNALITÉS PRINCIPALES:")
print(" 💬 Chat RAG avec recherche web intelligent")
print(" 🗺️ Calcul de trajets OSM")
print(" 📸 Analyse d'images avec IA")
print(" 🌐 Extraction de contenu web")
print(" 💾 Gestion unifiée des données")
print("\n🚀 UTILISATION:")
print(" Interface: Exécutez les cellules suivantes")
print(" API: kibali_api.ask('votre question')")
print(" Tests: test_all_features()")
print("\n⚙️ MAINTENANCE:")
print(" Status: get_system_status()")
print(" Nettoyage: cleanup_old_cache()")
print(" Sauvegarde: backup_all_data()")
print("="*60)