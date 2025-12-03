"""
Gestionnaire de mode offline pour Kibali
Détecte automatiquement la connexion et bascule entre online/offline
"""

import requests
import time
from pathlib import Path
import streamlit as st
from typing import Tuple, Optional

class OfflineManager:
    """Gestionnaire du mode offline avec détection automatique"""
    
    def __init__(self):
        self.last_check_time = 0
        self.check_interval = 30  # Vérifier toutes les 30 secondes
        self.timeout = 3  # Timeout de 3 secondes pour les tests
        
    def check_internet_connection(self) -> bool:
        """Vérifie si une connexion internet est disponible"""
        
        # Ne pas vérifier trop souvent
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            # Utiliser le dernier résultat en cache
            return st.session_state.get('internet_available', False)
        
        self.last_check_time = current_time
        
        # Tester plusieurs endpoints pour plus de fiabilité
        test_urls = [
            'https://www.google.com',
            'https://huggingface.co',
            'https://api.github.com'
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    st.session_state['internet_available'] = True
                    st.session_state['last_online_check'] = current_time
                    return True
            except (requests.ConnectionError, requests.Timeout, Exception):
                continue
        
        # Aucune connexion disponible
        st.session_state['internet_available'] = False
        return False
    
    def get_mode(self) -> str:
        """Retourne le mode actuel (online/offline)"""
        
        # Mode manuel forcé
        if 'force_offline_mode' in st.session_state and st.session_state['force_offline_mode']:
            return 'offline'
        
        # Détection automatique
        if self.check_internet_connection():
            return 'online'
        else:
            return 'offline'
    
    def force_offline(self, enabled: bool = True):
        """Force le mode offline manuellement"""
        st.session_state['force_offline_mode'] = enabled
    
    def get_status_info(self) -> dict:
        """Retourne les informations de statut détaillées"""
        mode = self.get_mode()
        
        info = {
            'mode': mode,
            'is_online': mode == 'online',
            'is_offline': mode == 'offline',
            'forced': st.session_state.get('force_offline_mode', False),
            'internet_available': st.session_state.get('internet_available', False),
            'last_check': st.session_state.get('last_online_check', 0)
        }
        
        return info

# Instance globale
offline_manager = OfflineManager()

def get_model_loading_params(mode: str = None) -> dict:
    """Retourne les paramètres de chargement selon le mode"""
    
    if mode is None:
        mode = offline_manager.get_mode()
    
    if mode == 'offline':
        return {
            'local_files_only': True,
            'resume_download': False,
            'force_download': False
        }
    else:
        return {
            'local_files_only': False,
            'resume_download': True,
            'force_download': False
        }

def load_model_with_fallback(model_name: str, cache_dir: Path, loading_function, **kwargs):
    """
    Charge un modèle avec fallback automatique online -> offline
    
    Args:
        model_name: Nom du modèle HuggingFace
        cache_dir: Dossier de cache
        loading_function: Fonction de chargement (from_pretrained)
        **kwargs: Arguments supplémentaires pour le chargement
    
    Returns:
        Le modèle chargé
    """
    
    mode = offline_manager.get_mode()
    params = get_model_loading_params(mode)
    
    # Fusionner les paramètres
    loading_kwargs = {**kwargs, **params, 'cache_dir': str(cache_dir)}
    
    try:
        # Tentative de chargement avec le mode actuel
        model = loading_function(model_name, **loading_kwargs)
        return model, mode
        
    except Exception as e:
        # Si échec en mode online, essayer offline
        if mode == 'online':
            print(f"⚠️ Échec chargement online, tentative offline...")
            offline_params = get_model_loading_params('offline')
            loading_kwargs.update(offline_params)
            
            try:
                model = loading_function(model_name, **loading_kwargs)
                # Basculer automatiquement en mode offline
                offline_manager.force_offline(True)
                st.warning("🔌 Connexion internet instable - Basculement automatique en mode offline")
                return model, 'offline'
            except Exception as offline_error:
                raise Exception(f"Impossible de charger le modèle: {str(e)} (offline: {str(offline_error)})")
        else:
            raise e

def render_mode_indicator():
    """Affiche l'indicateur de mode dans la sidebar"""
    
    status = offline_manager.get_status_info()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐")
    
    # Indicateur visuel avec icônes uniquement
    if status['is_online']:
        if status['forced']:
            st.sidebar.warning("🔌")
        else:
            st.sidebar.success("🌐")
    else:
        if status['forced']:
            st.sidebar.info("🔌")
        else:
            st.sidebar.error("🔌")
    
    # Boutons avec UNIQUEMENT des icônes
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🌐", disabled=status['is_online'] and not status['forced'], 
                     help="Mode Online - Téléchargements autorisés",
                     key="btn_online",
                     use_container_width=True):
            offline_manager.force_offline(False)
            st.rerun()
    
    with col2:
        if st.button("🔌", disabled=status['is_offline'] and status['forced'],
                     help="Mode Offline - Cache local uniquement",
                     key="btn_offline",
                     use_container_width=True):
            offline_manager.force_offline(True)
            st.rerun()
    
    # Informations détaillées (expander)
    with st.sidebar.expander("ℹ️"):
        mode_icon = "🌐" if status['mode'] == 'online' else "🔌"
        internet_icon = "✅" if status['internet_available'] else "❌"
        forced_icon = "🔒" if status['forced'] else "🔓"
        
        st.markdown(f"{mode_icon} {internet_icon} {forced_icon}")
        
        if status['last_check'] > 0:
            time_since = time.time() - status['last_check']
            st.caption(f"⏱️ {int(time_since)}s")
    
    st.sidebar.markdown("---")

def get_offline_capabilities() -> dict:
    """Retourne les capacités disponibles en mode offline"""
    
    from pathlib import Path
    cache_dir = Path(__file__).parent.parent / "kibali_data" / "models" / "huggingface_cache"
    
    # Vérifier quels modèles sont disponibles localement
    available_models = []
    
    if cache_dir.exists():
        model_folders = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith('models--')]
        
        # Mapper les dossiers aux capacités
        capabilities = {
            'llm': False,
            'vision': False,
            'embeddings': False,
            'code_generation': False,
            'image_generation': False,
            'translation': False,
            'summarization': False,
            'ner': False
        }
        
        for folder in model_folders:
            name = folder.name.lower()
            
            if 'qwen' in name or 'mistral' in name or 'phi' in name:
                capabilities['llm'] = True
            if 'clip' in name:
                capabilities['vision'] = True
            if 'sentence-transformers' in name or 'all-minilm' in name:
                capabilities['embeddings'] = True
            if 'codegen' in name or 'phi-1' in name:
                capabilities['code_generation'] = True
            if 'stable-diffusion' in name or 'dreamshaper' in name:
                capabilities['image_generation'] = True
            if 'opus-mt' in name or 'helsinki' in name:
                capabilities['translation'] = True
            if 'bart' in name:
                capabilities['summarization'] = True
            if 'bert' in name and 'ner' in name:
                capabilities['ner'] = True
        
        return capabilities
    
    return {}

def render_offline_status_card():
    """Affiche une carte de statut des capacités offline"""
    
    status = offline_manager.get_status_info()
    
    if status['is_offline']:
        caps = get_offline_capabilities()
        
        st.info("""
        ### 🔌 Mode Offline Actif
        
        Kibali fonctionne avec les modèles locaux en cache. 
        Performance optimale garantie !
        """)
        
        # Afficher les capacités disponibles
        st.markdown("**Capacités disponibles:**")
        
        cols = st.columns(4)
        
        cap_icons = {
            'llm': '🧠',
            'vision': '👁️',
            'embeddings': '🔤',
            'code_generation': '💻',
            'image_generation': '🎨',
            'translation': '🌍',
            'summarization': '📝',
            'ner': '🏷️'
        }
        
        cap_names = {
            'llm': 'LLM',
            'vision': 'Vision AI',
            'embeddings': 'Embeddings',
            'code_generation': 'Code Gen',
            'image_generation': 'Images',
            'translation': 'Traduction',
            'summarization': 'Résumé',
            'ner': 'NER'
        }
        
        for idx, (cap, available) in enumerate(caps.items()):
            with cols[idx % 4]:
                if available:
                    st.success(f"{cap_icons.get(cap, '✅')} {cap_names.get(cap, cap)}")
                else:
                    st.error(f"❌ {cap_names.get(cap, cap)}")

# Export des fonctions principales
__all__ = [
    'offline_manager',
    'get_model_loading_params',
    'load_model_with_fallback',
    'render_mode_indicator',
    'render_offline_status_card',
    'get_offline_capabilities'
]
