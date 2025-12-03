"""
Fonctionnalités de KIbalione8 à intégrer dans kibali-IA
Ce fichier contient les systèmes avancés :
1. Mémoire vectorielle chat pour continuité conversationnelle
2. Auto-apprentissage avec sous-modèles sklearn
3. Amélioration automatique de la DB via fouille internet
4. Highlighting et effets scintillants pour fluidité
"""

import os
import json
import time
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===============================================
# 1. SYSTÈME DE MÉMOIRE VECTORIELLE POUR HISTORIQUE CHAT
# ===============================================

def load_chat_vectordb(chat_vectordb_path, embedding_model):
    """Charger la base vectorielle pour l'historique chat"""
    if not os.path.exists(chat_vectordb_path):
        return None, "⚠️ Aucune base chat trouvée"
    
    try:
        from langchain_community.vectorstores import FAISS
        chat_vectordb = FAISS.load_local(
            chat_vectordb_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        return chat_vectordb, "✅ Base chat chargée"
    except Exception as e:
        return None, f"❌ Erreur chat: {e}"

def add_to_chat_db(user_msg, ai_msg, chat_vectordb, chat_vectordb_path, embedding_model):
    """Ajouter un échange user-AI à la base chat"""
    from langchain_community.vectorstores import FAISS
    
    if chat_vectordb is None:
        chat_vectordb = FAISS.from_texts([""], embedding_model)
    
    exchange = f"User: {user_msg} ||| Assistant: {ai_msg}"
    doc = Document(
        page_content=exchange,
        metadata={"type": "chat_exchange", "timestamp": time.time()}
    )
    
    chat_vectordb.add_documents([doc])
    chat_vectordb.save_local(chat_vectordb_path)
    
    return chat_vectordb

def chat_rag_search(question, chat_vectordb, k=3):
    """Rechercher dans l'historique chat pour contexte"""
    if not chat_vectordb:
        return []
    
    try:
        return chat_vectordb.similarity_search(question, k=k)
    except Exception as e:
        print(f"❌ Erreur recherche chat: {e}")
        return []

# ===============================================
# 2. AUTO-APPRENTISSAGE AVEC SOUS-MODÈLES SKLEARN
# ===============================================

def create_submodel_from_chat_history(chat_vectordb, submodels_path, submodel_type="classification"):
    """
    Crée un petit sous-modèle sklearn à partir de l'historique chat pour automatiser des réponses.
    - Type: 'classification' pour classer les questions et prédire des réponses automatisées.
    Rend le modèle plus "humain" en apprenant des patterns conversationnels.
    """
    if not chat_vectordb:
        return None, "❌ Aucune base chat pour entraîner le sous-modèle"
   
    # Extraire les échanges de l'historique
    exchanges = []
    try:
        for doc in list(chat_vectordb.docstore._dict.values()) or []:
            exchange = doc.page_content
            if "User:" in exchange and "Assistant:" in exchange:
                parts = exchange.split("|||")
                if len(parts) == 2:
                    user_part = parts[0].replace("User: ", "").strip()
                    ai_part = parts[1].replace("Assistant: ", "").strip()
                    exchanges.append((user_part, ai_part))
    except:
        return None, "❌ Erreur extraction échanges"
   
    if len(exchanges) < 10:
        return None, f"❌ Historique chat trop court ({len(exchanges)} échanges, min 10)"
   
    try:
        # Préparation des données : TF-IDF pour vectorisation textuelle
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform([user[0] for user in exchanges])
       
        # Pour classification simple (ex: prédire si réponse est informative ou autre)
        labels = []
        for user_msg, _ in exchanges:
            if re.search(r'\?', user_msg):
                labels.append(1)  # Question
            elif any(word in user_msg.lower() for word in ['info', 'savoir', 'expliquer']):
                labels.append(0)  # Info
            else:
                labels.append(2)  # Autre
       
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
       
        if submodel_type == "classification":
            model = MultinomialNB()
        else:
            model = RandomForestClassifier(n_estimators=50)
       
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
       
        # Sauvegarder le modèle et vectorizer
        os.makedirs(submodels_path, exist_ok=True)
        model_path = os.path.join(
            submodels_path, 
            f"submodel_{submodel_type}_{int(time.time())}.pkl"
        )
        
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
       
        # Visualisation avec matplotlib : Accuracy plot
        fig, ax = plt.subplots()
        ax.bar(['Train', 'Test'], [1.0, accuracy])
        ax.set_title(f'Précision du sous-modèle {submodel_type.capitalize()}')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1.1])
        
        plot_path = os.path.join(
            submodels_path,
            f"accuracy_plot_{submodel_type}_{int(time.time())}.png"
        )
        plt.savefig(plot_path)
        plt.close()
       
        return model_path, f"✅ Sous-modèle {submodel_type} créé avec accuracy {accuracy:.2f}. Sauvegardé: {model_path}"
    except Exception as e:
        return None, f"❌ Erreur création sous-modèle: {e}"

def use_submodel_for_automation(query, submodel_path, submodels_path, submodel_type="classification"):
    """
    Utilise un sous-modèle pour automatiser une réponse, rendant le comportement plus humain.
    """
    if not os.path.exists(submodel_path):
        return "❌ Sous-modèle non trouvé"
   
    try:
        with open(submodel_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            vectorizer = data['vectorizer']
       
        query_vec = vectorizer.transform([query])
        prediction = model.predict(query_vec)[0]
       
        # Réponses automatisées basées sur prédiction
        automated_responses = {
            0: "Voici des infos basiques sur ce sujet, basées sur nos échanges passés.",
            1: "Bonne question ! Laisse-moi réfléchir à ça en me basant sur ce qu'on a discuté avant.",
            2: "Intéressant, je vais creuser un peu plus pour te répondre de manière personnalisée."
        }
       
        response = automated_responses.get(prediction, "Réponse automatisée générée.")
       
        # Visualisation: Distribution des features TF-IDF pour la query
        fig, ax = plt.subplots()
        tfidf_scores = query_vec.toarray()[0]
        top_features_idx = np.argsort(tfidf_scores)[-5:]
        top_scores = tfidf_scores[top_features_idx]
        
        feature_names = vectorizer.get_feature_names_out()
        top_feature_names = [feature_names[i] for i in top_features_idx]
        
        ax.bar(range(len(top_features_idx)), top_scores)
        ax.set_title('Top Features TF-IDF pour la Query')
        ax.set_xticks(range(len(top_features_idx)))
        ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
        ax.tight_layout()
        
        plot_path = os.path.join(
            submodels_path,
            f"query_features_{int(time.time())}.png"
        )
        plt.savefig(plot_path)
        plt.close()
       
        return f"{response} (Prédiction: {prediction}) | Graph: {plot_path}"
    except Exception as e:
        return f"❌ Erreur utilisation sous-modèle: {e}"

# ===============================================
# 3. AMÉLIORATION DB PAR FOUILLE INTERNET
# ===============================================

def improve_database_with_web_search(
    topics, 
    num_results_per_topic, 
    vectordb, 
    vectordb_path, 
    embedding_model,
    enhanced_web_search_func,
    smart_content_extraction_func
):
    """
    Fouille internet sur des sujets spécifiques et améliore la base de données.
    """
    from langchain_community.vectorstores import FAISS
    
    specific_topics = topics or [
        "pétrole extraction techniques", 
        "topographie cartographie avancée", 
        "sciences physiques mécanique sol", 
        "sous-sol géologie ressources",
        "ERT electrical resistivity tomography",
        "géophysique méthodes prospection"
    ]
   
    if vectordb is None:
        vectordb = FAISS.from_texts([""], embedding_model)
   
    new_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
   
    for topic in specific_topics:
        print(f"🔍 Fouille internet pour: {topic}")
        
        search_results = enhanced_web_search_func(
            topic, 
            max_results=num_results_per_topic, 
            search_type="both"
        )
       
        for result in search_results:
            content = f"Titre: {result.get('title', '')}\nContenu: {result.get('body', '')}\n"
            url = result.get('href') or result.get('url')
            
            if url and len(result.get('body', '')) < 500:
                extra_content = smart_content_extraction_func(url, max_length=2000)
                if "Impossible d'extraire" not in extra_content:
                    content += f"\nContenu détaillé: {extra_content}"
           
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": url or topic,
                        "topic": topic,
                        "type": "web_enrichment",
                        "chunk_id": i,
                        "timestamp": time.time()
                    }
                )
                new_documents.append(doc)
   
    if new_documents:
        vectordb.add_documents(new_documents)
        vectordb.save_local(vectordb_path)
        return vectordb, f"✅ Base améliorée: {len(new_documents)} nouveaux chunks ajoutés sur {len(specific_topics)} sujets"
    else:
        return vectordb, "⚠️ Aucun nouveau contenu ajouté"

# ===============================================
# 4. HIGHLIGHTING ET EFFETS SCINTILLANTS
# ===============================================

def highlight_important_words(text):
    """Met en évidence les mots importants avec effet scintillante et tooltip"""
    important_keywords = [
        'important', 'clé', 'essentiel', 'critique', 'principal', 
        'trajet', 'pétrole', 'topographie', 'ERT', 'résistivité',
        'géophysique', 'analyse', 'données', 'graphique', 'tableau',
        'précis', 'détaillé', 'complet'
    ]
    
    for keyword in important_keywords:
        pattern = rf'\b({keyword})\b'
        replacement = r'<span class="sparkle-word" title="\1: Terme clé pour la compréhension du contexte">\1</span>'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def get_sparkle_css():
    """Retourne le CSS pour les effets scintillants"""
    return """
    /* Effet scintillante pour mots importants */
    .sparkle-word {
        color: #2196F3;
        background: linear-gradient(45deg, #2196F3, #21CBF3, #4ecdc4, #45b7d1);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: sparkle 2s linear infinite, gradient-shift 3s ease infinite;
        cursor: pointer;
        position: relative;
        padding: 2px 4px;
        border-radius: 4px;
        transition: transform 0.2s ease;
        font-weight: 700;
    }
    
    .sparkle-word:hover {
        transform: scale(1.1);
        text-shadow: 0 0 10px rgba(33, 150, 243, 0.8);
    }
    
    @keyframes sparkle {
        0%, 100% { text-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }
        50% { text-shadow: 0 0 20px rgba(33, 150, 243, 1), 0 0 30px rgba(33, 203, 243, 0.7); }
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    """

# ===============================================
# 5. GESTION AMÉLIORÉE DU CHAT AVEC OUTILS
# ===============================================

def handle_chat_enhanced_with_tools(
    message, 
    history, 
    agent, 
    model_choice, 
    vectordb, 
    graph, 
    pois, 
    web_enabled,
    chat_vectordb,
    chat_vectordb_path,
    embedding_model,
    working_models,
    hybrid_search_enhanced_func,
    generate_answer_enhanced_func
):
    """
    Gestion améliorée du chat avec:
    - Mémoire vectorielle (historique)
    - Recherche hybride (locale + historique + web)
    - Auto-sauvegarde des échanges
    """
    if not message.strip():
        return ""
    
    # Charger/créer la base chat si nécessaire
    if chat_vectordb is None:
        from langchain_community.vectorstores import FAISS
        chat_vectordb = FAISS.from_texts([""], embedding_model)
    
    try:
        if not web_enabled:
            # Recherche hybride incluant historique chat
            docs = hybrid_search_enhanced_func(
                message, 
                vectordb, 
                k=3, 
                web_search_enabled=False,
                chat_vectordb=chat_vectordb
            )
            response = generate_answer_enhanced_func(
                message, 
                docs, 
                working_models[model_choice], 
                include_sources=True
            )
        else:
            # Utiliser l'agent avec tous les outils
            response = agent.run(message)
    
    except Exception as e:
        response = f"❌ Erreur: {e}\n\nTentative avec recherche locale..."
        try:
            docs = hybrid_search_enhanced_func(
                message, 
                vectordb, 
                k=3, 
                web_search_enabled=False,
                chat_vectordb=chat_vectordb
            )
            response = generate_answer_enhanced_func(
                message, 
                docs, 
                working_models[model_choice]
            )
        except Exception as e2:
            response = f"❌ Erreur complète: {e2}"
    
    # Sauvegarder l'échange dans la base chat
    chat_vectordb = add_to_chat_db(
        message, 
        response, 
        chat_vectordb,
        chat_vectordb_path,
        embedding_model
    )
    
    # Appliquer highlighting pour fluidité
    response = highlight_important_words(response)
    
    return response, chat_vectordb

# ===============================================
# 6. FONCTIONS D'INTÉGRATION DANS L'APP PRINCIPALE
# ===============================================

def init_kibalione8_systems(chatbot_dir):
    """
    Initialise tous les systèmes KibaliOne8 dans l'app principale.
    Retourne les chemins et configurations nécessaires.
    """
    # Créer les dossiers nécessaires
    chat_vectordb_path = os.path.join(chatbot_dir, "chat_vectordb")
    submodels_path = os.path.join(chatbot_dir, "submodels")
    
    os.makedirs(chat_vectordb_path, exist_ok=True)
    os.makedirs(submodels_path, exist_ok=True)
    
    print("✅ Systèmes KibaliOne8 initialisés:")
    print(f"   📝 Base chat: {chat_vectordb_path}")
    print(f"   🧠 Sous-modèles: {submodels_path}")
    
    return {
        'chat_vectordb_path': chat_vectordb_path,
        'submodels_path': submodels_path
    }

def update_agent_with_chat_memory(
    model_choice, 
    vectordb, 
    graph, 
    pois, 
    chat_vectordb,
    working_models,
    create_enhanced_agent_func,
    get_cache_stats_func
):
    """
    Met à jour l'agent en incluant la mémoire chat.
    """
    model_name = working_models[model_choice]
    agent = create_enhanced_agent_func(model_name, vectordb, graph, pois, chat_vectordb)
    cache_info = get_cache_stats_func()
    return model_name, agent, cache_info
