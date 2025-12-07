"""
Autodible UI - Interface Streamlit Simple
Contrôle et monitoring de l'assistant audio
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent))

from autodible import AutodibleEngine
from autodible_config import (
    get_config_summary, ASSISTANCE_MODES,
    SYSTEM_CONFIG, AUDIO_CONFIG, LLM_CONFIG
)
from loguru import logger

# Configuration page
st.set_page_config(
    page_title="Autodible - Assistant Audio",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00ff88 0%, #00ffff 50%, #0088ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .status-active {
        color: #00ff88;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    
    .status-inactive {
        color: #ff4444;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid rgba(0, 255, 136, 0.3);
        margin: 0.5rem 0;
    }
    
    .transcript-box {
        background: #16213e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
        margin: 0.5rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALISATION SESSION STATE
# ============================================================

if "engine" not in st.session_state:
    st.session_state.engine = None
    st.session_state.is_running = False
    st.session_state.transcripts = []
    st.session_state.responses = []
    st.session_state.start_time = None

# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Mode d'assistance
    mode = st.selectbox(
        "Mode d'assistance",
        options=list(ASSISTANCE_MODES.keys()),
        format_func=lambda x: f"{ASSISTANCE_MODES[x]['name']} - {ASSISTANCE_MODES[x]['description']}"
    )
    
    st.markdown("---")
    
    # Informations système
    st.markdown("### 📊 Système")
    
    config_summary = get_config_summary()
    for key, value in config_summary.items():
        st.metric(key, value)
    
    st.markdown("---")
    
    # Paramètres audio
    st.markdown("### 🎤 Audio")
    
    volume = st.slider("Volume oreillette", 0.0, 1.0, 0.7, 0.1)
    speed = st.slider("Vitesse voix", 0.8, 1.5, 1.15, 0.05)
    
    st.markdown("---")
    
    # Aide
    with st.expander("❓ Comment utiliser"):
        st.markdown("""
        **Étapes:**
        1. Cliquez sur "🚀 Démarrer Autodible"
        2. Mettez votre casque/oreillette
        3. Parlez normalement
        4. Écoutez les suggestions dans l'oreillette
        
        **Conseils:**
        - Parlez clairement
        - Attendez la suggestion avant de répondre
        - Ajustez le volume si besoin
        """)

# ============================================================
# MAIN - INTERFACE PRINCIPALE
# ============================================================

st.markdown('<div class="main-title">🎧 Autodible</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #888;">Assistant Audio Intelligent en Temps Réel</p>', unsafe_allow_html=True)

# Zone de statut
col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    if st.session_state.is_running:
        st.markdown('<div class="status-active">🟢 ACTIF</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-inactive">⚪ INACTIF</div>', unsafe_allow_html=True)

with col_status2:
    if st.session_state.is_running and st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        st.metric("Temps actif", f"{elapsed}s")
    else:
        st.metric("Temps actif", "0s")

with col_status3:
    st.metric("Transcriptions", len(st.session_state.transcripts))

st.markdown("---")

# Boutons de contrôle
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("🚀 Démarrer Autodible", disabled=st.session_state.is_running, use_container_width=True):
        with st.spinner("Initialisation..."):
            try:
                # Créer engine si nécessaire
                if st.session_state.engine is None:
                    st.session_state.engine = AutodibleEngine()
                
                # Configurer mode
                st.session_state.engine.set_mode(mode)
                
                # Démarrer
                st.session_state.engine.start()
                st.session_state.is_running = True
                st.session_state.start_time = time.time()
                
                st.success("✅ Autodible démarré ! Parlez dans votre micro.")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                logger.error(f"Erreur démarrage: {e}", exc_info=True)

with col_btn2:
    if st.button("⏹️ Arrêter", disabled=not st.session_state.is_running, use_container_width=True):
        if st.session_state.engine:
            st.session_state.engine.stop()
            st.session_state.is_running = False
            st.success("✅ Autodible arrêté")
            st.rerun()

with col_btn3:
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.engine = None
        st.session_state.is_running = False
        st.session_state.transcripts = []
        st.session_state.responses = []
        st.session_state.start_time = None
        st.rerun()

st.markdown("---")

# ============================================================
# MONITORING EN TEMPS RÉEL
# ============================================================

st.markdown("### 📊 Monitoring en Temps Réel")

# Tabs
tab_transcripts, tab_context, tab_stats, tab_logs = st.tabs([
    "🗣️ Transcriptions",
    "🧠 Contexte",
    "📈 Statistiques",
    "📋 Logs"
])

with tab_transcripts:
    st.markdown("**Historique des transcriptions et réponses**")
    
    if st.session_state.engine and st.session_state.engine.context_history:
        for i, entry in enumerate(reversed(st.session_state.engine.context_history[-10:])):
            with st.container():
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="transcript-box">
                        <strong>👤 Vous:</strong><br>
                        {entry['query']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="transcript-box" style="border-left-color: #0088ff;">
                        <strong>🤖 Autodible:</strong><br>
                        {entry['response']}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"<small style='color: #666;'>Il y a {int(time.time() - entry['timestamp'])}s</small>", unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("Aucune transcription pour le moment. Démarrez Autodible et parlez !")

with tab_context:
    st.markdown("**Contexte de conversation actuel**")
    
    if st.session_state.engine and st.session_state.engine.context_history:
        st.json({
            "Mode actuel": st.session_state.engine.current_mode,
            "Nombre d'échanges": len(st.session_state.engine.context_history),
            "Dernier échange": st.session_state.engine.context_history[-1] if st.session_state.engine.context_history else None
        })
    else:
        st.info("Aucun contexte pour le moment")

with tab_stats:
    st.markdown("**Statistiques de performance**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Latence STT cible",
            f"{SYSTEM_CONFIG['max_latency_stt']}s"
        )
    
    with col2:
        st.metric(
            "Latence LLM cible",
            f"{SYSTEM_CONFIG['max_latency_llm']}s"
        )
    
    with col3:
        st.metric(
            "Latence TTS cible",
            f"{SYSTEM_CONFIG['max_latency_tts']}s"
        )
    
    st.markdown("---")
    
    st.markdown("**Modèles chargés:**")
    st.code(f"""
LLM: {LLM_CONFIG['primary']['model_name']}
STT: faster-whisper (base)
TTS: Coqui XTTS v2
Audio: {AUDIO_CONFIG['sample_rate']}Hz
    """)

with tab_logs:
    st.markdown("**Logs système**")
    
    if Path("autodible.log").exists():
        with open("autodible.log", "r") as f:
            logs = f.readlines()
            st.code("".join(logs[-50:]))  # 50 dernières lignes
    else:
        st.info("Aucun log disponible")

# ============================================================
# AUTO-REFRESH si actif
# ============================================================

if st.session_state.is_running:
    time.sleep(2)
    st.rerun()
