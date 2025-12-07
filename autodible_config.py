"""
Configuration d'Autodible - Assistant Audio Intelligent en Temps Réel
Utilise les modèles locaux disponibles dans kibali_data/models
"""
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "kibali_data" / "models"
CACHE_DIR = MODELS_DIR / "huggingface_cache"

# ============================================================
# CONFIGURATION LLM - Modèles locaux
# ============================================================

LLM_CONFIG = {
    # Modèle principal: Mistral-7B (puissant, rapide)
    "primary": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "cache_dir": str(CACHE_DIR),
        "device_map": "auto",  # Détection auto GPU/CPU
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "max_new_tokens": 50,  # Réponses courtes pour l'oreillette
        "temperature": 0.3,  # Déterministe = plus rapide
        "top_p": 0.9,
        "do_sample": True,
        "streaming": True,  # CRUCIAL pour latence < 1s
    },
    
    # Modèle de secours: Qwen (ultra-rapide)
    "fallback": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "cache_dir": str(CACHE_DIR),
        "device_map": "cpu",
        "max_new_tokens": 40,
        "temperature": 0.3,
        "streaming": True,
    },
    
    # Modèle alternatif: Phi-3 (équilibre)
    "alternative": {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "cache_dir": str(CACHE_DIR),
        "max_new_tokens": 45,
        "temperature": 0.3,
        "streaming": True,
    }
}

# ============================================================
# CONFIGURATION SPEECH-TO-TEXT (Transcription)
# ============================================================

STT_CONFIG = {
    "engine": "faster-whisper",  # Plus rapide que Whisper standard
    "model_size": "base",  # tiny|base|small|medium|large
    "device": "auto",  # auto|cuda|cpu
    "compute_type": "float16",  # float16|int8 (GPU) ou int8 (CPU)
    "language": "fr",  # Langue principale
    "beam_size": 1,  # 1 = plus rapide, 5 = meilleure qualité
    "vad_filter": True,  # Voice Activity Detection
    "vad_threshold": 0.5,
    "min_silence_duration_ms": 500,
}

# ============================================================
# CONFIGURATION TEXT-TO-SPEECH (Voix dans l'oreillette)
# ============================================================

TTS_CONFIG = {
    "engine": "coqui-tts",  # coqui-tts|pyttsx3|gtts
    "model": "tts_models/multilingual/multi-dataset/xtts_v2",
    "language": "fr",
    "speaker_wav": None,  # Chemin vers fichier voix custom (optionnel)
    "speed": 1.15,  # Légèrement accéléré pour l'oreille
    "pitch": 0.95,  # Voix légèrement plus grave = discret
    "volume": 0.7,  # Volume modéré
    "streaming": True,  # Commence à parler avant fin génération
}

# ============================================================
# CONFIGURATION AUDIO (Micro & Casque)
# ============================================================

AUDIO_CONFIG = {
    # Capture microphone
    "input_device": None,  # None = défaut système, ou index device
    "sample_rate": 16000,  # Hz (Whisper optimisé pour 16kHz)
    "channels": 1,  # Mono
    "chunk_duration": 0.5,  # Secondes par chunk
    "buffer_duration": 5.0,  # Buffer circulaire 5s
    
    # Sortie casque/oreillette
    "output_device": None,  # None = défaut système
    "output_sample_rate": 22050,  # Hz pour TTS
    
    # Detection voix
    "energy_threshold": 300,  # Seuil détection voix
    "pause_threshold": 0.8,  # Pause entre phrases (secondes)
}

# ============================================================
# MODES D'ASSISTANCE
# ============================================================

ASSISTANCE_MODES = {
    "general": {
        "name": "Général",
        "description": "Aide générale sur tous sujets",
        "prompt_template": """Tu es un assistant vocal discret qui aide l'utilisateur en temps réel.
Réponds de manière TRÈS COURTE (max 2-3 phrases) et DIRECTE.
Context: {context}
Question/Besoin: {query}
Réponse courte:"""
    },
    
    "meeting": {
        "name": "Réunion Pro",
        "description": "Assistance pendant réunions professionnelles",
        "prompt_template": """Tu assistes discrètement pendant une réunion professionnelle.
Fournis des FAITS, CHIFFRES ou ARGUMENTS courts et précis.
Context réunion: {context}
Point discuté: {query}
Suggestion (1 phrase):"""
    },
    
    "interview": {
        "name": "Entretien",
        "description": "Aide pour entretiens d'embauche",
        "prompt_template": """Tu aides pendant un entretien d'embauche.
Suggère une réponse COURTE, PROFESSIONNELLE et CONCRÈTE.
Profil candidat: {context}
Question posée: {query}
Suggestion rapide:"""
    },
    
    "debate": {
        "name": "Débat/Négociation",
        "description": "Arguments et contre-arguments",
        "prompt_template": """Tu fournis des arguments ou contre-arguments courts.
Contexte débat: {context}
Argument adverse: {query}
Contre-argument (15 mots max):"""
    },
    
    "learning": {
        "name": "Apprentissage",
        "description": "Aide à l'étude et compréhension",
        "prompt_template": """Tu aides à comprendre un concept complexe.
Explique de manière SIMPLE et COURTE.
Sujet: {context}
Question: {query}
Explication courte:"""
    }
}

# ============================================================
# PARAMÈTRES SYSTÈME
# ============================================================

SYSTEM_CONFIG = {
    # Performance
    "use_gpu": True,  # Utiliser GPU si disponible
    "mixed_precision": True,  # FP16 pour vitesse
    "preload_models": True,  # Charger au démarrage (pas de latence après)
    
    # Cache des réponses fréquentes
    "enable_cache": True,
    "cache_similarity_threshold": 0.85,  # Seuil similarité sémantique
    "cache_max_size": 100,  # Nombre max réponses en cache
    
    # Latences cibles
    "max_latency_stt": 0.3,  # Secondes
    "max_latency_llm": 1.0,  # Secondes
    "max_latency_tts": 0.3,  # Secondes
    "target_total_latency": 1.5,  # Objectif total
    
    # Pipeline
    "parallel_processing": True,  # Traitement parallèle STT+LLM+TTS
    "async_mode": True,  # Opérations asynchrones
    
    # Logging
    "log_level": "INFO",  # DEBUG|INFO|WARNING|ERROR
    "log_file": "autodible.log",
    "log_transcripts": True,  # Garder historique transcriptions
}

# ============================================================
# PROMPTS SYSTÈME
# ============================================================

SYSTEM_PROMPTS = {
    "base": """Tu es Autodible, un assistant vocal intelligent et DISCRET.
Tu fournis des réponses ULTRA-COURTES (max 2-3 phrases) car tu parles dans l'oreillette de l'utilisateur.
Ton style:
- CONCIS et DIRECT
- ACTIONNABLE (conseils pratiques)
- NATUREL (comme un ami qui chuchote)
- PAS de formules de politesse inutiles
- PAS de longues explications

Réponds toujours en 15-30 mots maximum.""",

    "urgent": """URGENT - Réponse en 10 mots MAX. Sois TRÈS bref.""",
    
    "contextual": """Utilise le contexte de la conversation en cours.
Adapte ta réponse à la situation détectée.
Reste CONCIS."""
}

# ============================================================
# INTERFACE UTILISATEUR
# ============================================================

UI_CONFIG = {
    "show_transcription": True,  # Afficher transcription en temps réel
    "show_suggestions": True,  # Afficher suggestions avant audio
    "minimal_mode": False,  # Mode minimal (juste icône)
    "notifications": True,  # Notifications système
    "hotkey": "ctrl+shift+a",  # Raccourci activation/désactivation
    "tray_icon": True,  # Icône dans system tray
}

# ============================================================
# SÉCURITÉ & PRIVACY
# ============================================================

PRIVACY_CONFIG = {
    "save_recordings": False,  # Ne PAS sauvegarder audio
    "save_transcripts": False,  # Ne PAS sauvegarder transcriptions
    "anonymize_data": True,  # Anonymiser données sensibles
    "local_only": True,  # Tout traitement local (pas de cloud)
    "encryption": False,  # Chiffrement (si sauvegarde activée)
}

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def get_config_summary():
    """Résumé de la configuration"""
    return {
        "LLM Principal": LLM_CONFIG["primary"]["model_name"],
        "STT Engine": STT_CONFIG["engine"],
        "TTS Engine": TTS_CONFIG["engine"],
        "GPU Enabled": SYSTEM_CONFIG["use_gpu"],
        "Latence cible": f"{SYSTEM_CONFIG['target_total_latency']}s",
        "Mode privacy": "100% local" if PRIVACY_CONFIG["local_only"] else "Hybride",
    }

def validate_config():
    """Valide la configuration"""
    errors = []
    
    # Vérifier modèles
    if not CACHE_DIR.exists():
        errors.append(f"❌ Cache modèles introuvable: {CACHE_DIR}")
    
    # Vérifier latences
    total = (SYSTEM_CONFIG["max_latency_stt"] + 
             SYSTEM_CONFIG["max_latency_llm"] + 
             SYSTEM_CONFIG["max_latency_tts"])
    if total > SYSTEM_CONFIG["target_total_latency"] * 1.5:
        errors.append(f"⚠️ Latences cumulées ({total}s) dépassent cible")
    
    if errors:
        print("\n".join(errors))
        return False
    
    print("✅ Configuration Autodible validée")
    return True

# ============================================================
# INITIALISATION
# ============================================================

if __name__ == "__main__":
    print("🎧 Configuration Autodible")
    print("="*60)
    
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("="*60)
    validate_config()
