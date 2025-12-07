"""
Autodible - Moteur Principal
Assistant Audio Intelligent en Temps Réel avec Oreillette

Architecture pipeline optimisée:
Micro → Whisper → Mistral-7B → Coqui TTS → Oreillette
Latence cible: < 1.5 secondes
"""

import asyncio
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import torch
import numpy as np

# Configuration
from autodible_config import (
    LLM_CONFIG, STT_CONFIG, TTS_CONFIG, AUDIO_CONFIG,
    SYSTEM_CONFIG, ASSISTANCE_MODES, SYSTEM_PROMPTS,
    PRIVACY_CONFIG
)

# Logging
from loguru import logger
import sys

# Configuration logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
    level=SYSTEM_CONFIG["log_level"]
)
if SYSTEM_CONFIG.get("log_file"):
    logger.add(
        SYSTEM_CONFIG["log_file"],
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )


class AudioCapture:
    """Capture audio du microphone avec VAD"""
    
    def __init__(self):
        import sounddevice as sd
        self.sd = sd
        self.sample_rate = AUDIO_CONFIG["sample_rate"]
        self.channels = AUDIO_CONFIG["channels"]
        self.chunk_duration = AUDIO_CONFIG["chunk_duration"]
        self.buffer_duration = AUDIO_CONFIG["buffer_duration"]
        
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        logger.info(f"🎤 AudioCapture initialisé: {self.sample_rate}Hz, {self.channels} canal(aux)")
    
    def start(self):
        """Démarre la capture audio"""
        if self.is_recording:
            return
        
        chunk_size = int(self.sample_rate * self.chunk_duration)
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            # Convertir en mono si nécessaire
            audio_data = indata.copy()
            if audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
            self.audio_queue.put(audio_data.flatten())
        
        self.stream = self.sd.InputStream(
            device=AUDIO_CONFIG["input_device"],
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=chunk_size,
            callback=audio_callback
        )
        
        self.stream.start()
        self.is_recording = True
        logger.success("✅ Capture audio démarrée")
    
    def stop(self):
        """Arrête la capture"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_recording = False
        logger.info("⏹️ Capture audio arrêtée")
    
    def get_audio_chunk(self, timeout=1.0):
        """Récupère un chunk audio de la queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class TranscriptionEngine:
    """Moteur de transcription avec Faster-Whisper"""
    
    def __init__(self):
        from faster_whisper import WhisperModel
        
        logger.info(f"🗣️ Chargement Faster-Whisper ({STT_CONFIG['model_size']})...")
        
        device = "cuda" if torch.cuda.is_available() and STT_CONFIG["device"] != "cpu" else "cpu"
        
        self.model = WhisperModel(
            STT_CONFIG["model_size"],
            device=device,
            compute_type=STT_CONFIG["compute_type"] if device == "cuda" else "int8"
        )
        
        self.language = STT_CONFIG["language"]
        self.beam_size = STT_CONFIG["beam_size"]
        
        logger.success(f"✅ Faster-Whisper chargé sur {device}")
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcrit l'audio en texte"""
        start_time = time.time()
        
        # Whisper attend float32 entre -1 et 1
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normaliser si nécessaire
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        segments, info = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=STT_CONFIG["vad_filter"],
            vad_parameters={
                "threshold": STT_CONFIG["vad_threshold"],
                "min_silence_duration_ms": STT_CONFIG["min_silence_duration_ms"]
            }
        )
        
        text = " ".join([segment.text for segment in segments])
        latency = time.time() - start_time
        
        logger.debug(f"📝 Transcription ({latency:.2f}s): {text}")
        
        return text.strip()


class LLMEngine:
    """Moteur LLM local avec streaming"""
    
    def __init__(self, mode="primary"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        config = LLM_CONFIG[mode]
        logger.info(f"🤖 Chargement {config['model_name']}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            cache_dir=config["cache_dir"],
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            cache_dir=config["cache_dir"],
            device_map=config.get("device_map", "auto"),
            torch_dtype=torch.float16 if config.get("torch_dtype") == "auto" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=config.get("low_cpu_mem_usage", True)
        )
        
        self.config = config
        self.mode = mode
        
        logger.success(f"✅ LLM chargé: {config['model_name']}")
    
    def generate(self, prompt: str, stream: bool = True):
        """Génère une réponse (streaming ou complète)"""
        start_time = time.time()
        
        # Préparation du prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["base"]},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenization
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)
        
        # Génération
        generation_config = {
            "max_new_tokens": self.config["max_new_tokens"],
            "temperature": self.config["temperature"],
            "top_p": self.config.get("top_p", 0.9),
            "do_sample": self.config.get("do_sample", True),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        if stream:
            # Streaming (retourne tokens progressivement)
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            
            generation_config["streamer"] = streamer
            
            thread = threading.Thread(
                target=self.model.generate,
                kwargs={"input_ids": inputs, **generation_config}
            )
            thread.start()
            
            full_response = ""
            for text_chunk in streamer:
                full_response += text_chunk
                yield text_chunk
            
            thread.join()
            latency = time.time() - start_time
            logger.debug(f"🤖 Génération stream ({latency:.2f}s): {full_response[:50]}...")
        else:
            # Génération complète
            outputs = self.model.generate(inputs, **generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraire seulement la réponse (après le prompt)
            if "assistant" in response.lower():
                response = response.split("assistant")[-1].strip()
            
            latency = time.time() - start_time
            logger.debug(f"🤖 Génération ({latency:.2f}s): {response[:50]}...")
            
            yield response


class TTSEngine:
    """Moteur Text-to-Speech pour l'oreillette"""
    
    def __init__(self):
        if TTS_CONFIG["engine"] == "coqui-tts":
            from TTS.api import TTS
            logger.info(f"🔊 Chargement Coqui TTS ({TTS_CONFIG['model']})...")
            self.tts = TTS(TTS_CONFIG["model"])
            logger.success("✅ Coqui TTS chargé")
        else:
            import pyttsx3
            logger.info("🔊 Chargement pyttsx3...")
            self.tts = pyttsx3.init()
            self.tts.setProperty('rate', int(150 * TTS_CONFIG["speed"]))
            self.tts.setProperty('volume', TTS_CONFIG["volume"])
            logger.success("✅ pyttsx3 chargé")
        
        self.engine_type = TTS_CONFIG["engine"]
    
    def speak(self, text: str, async_mode: bool = True):
        """Parle le texte dans l'oreillette"""
        start_time = time.time()
        
        logger.debug(f"🔊 TTS: {text}")
        
        if self.engine_type == "coqui-tts":
            # Générer audio
            import sounddevice as sd
            import numpy as np
            
            # Coqui TTS retourne un numpy array
            audio = self.tts.tts(text, language=TTS_CONFIG["language"])
            
            # Jouer l'audio
            if async_mode:
                threading.Thread(target=lambda: sd.play(audio, TTS_CONFIG["output_sample_rate"])).start()
            else:
                sd.play(audio, TTS_CONFIG["output_sample_rate"])
                sd.wait()
        else:
            # pyttsx3
            if async_mode:
                threading.Thread(target=lambda: self.tts.say(text) or self.tts.runAndWait()).start()
            else:
                self.tts.say(text)
                self.tts.runAndWait()
        
        latency = time.time() - start_time
        logger.debug(f"✅ TTS terminé ({latency:.2f}s)")


class AutodibleEngine:
    """
    Moteur principal Autodible
    Orchestre: Audio → STT → LLM → TTS
    """
    
    def __init__(self):
        logger.info("🎧 Initialisation Autodible Engine...")
        
        # Composants
        self.audio = AudioCapture()
        self.stt = TranscriptionEngine()
        self.llm = LLMEngine(mode="primary")
        self.tts = TTSEngine()
        
        # État
        self.is_running = False
        self.current_mode = "general"
        self.context_history = []
        
        # Cache réponses
        self.response_cache = {}
        
        logger.success("✅ Autodible Engine prêt!")
    
    def set_mode(self, mode: str):
        """Change le mode d'assistance"""
        if mode in ASSISTANCE_MODES:
            self.current_mode = mode
            logger.info(f"🔄 Mode changé: {ASSISTANCE_MODES[mode]['name']}")
        else:
            logger.warning(f"⚠️ Mode inconnu: {mode}")
    
    def process_audio_realtime(self):
        """Pipeline de traitement en temps réel"""
        logger.info("▶️ Pipeline temps réel démarré")
        
        audio_buffer = []
        silence_duration = 0
        
        while self.is_running:
            # 1. Capturer audio
            chunk = self.audio.get_audio_chunk(timeout=0.1)
            if chunk is None:
                continue
            
            audio_buffer.append(chunk)
            
            # Détecter silence (fin de phrase)
            energy = np.sqrt(np.mean(chunk**2))
            if energy < AUDIO_CONFIG["energy_threshold"] / 10000:
                silence_duration += AUDIO_CONFIG["chunk_duration"]
            else:
                silence_duration = 0
            
            # Si silence détecté après parole
            if silence_duration > AUDIO_CONFIG["pause_threshold"] and len(audio_buffer) > 3:
                # 2. Transcrire
                audio_data = np.concatenate(audio_buffer)
                try:
                    text = self.stt.transcribe(audio_data)
                    
                    if text and len(text) > 5:  # Éviter transcriptions vides
                        logger.info(f"👤 Utilisateur: {text}")
                        
                        # 3. Générer réponse LLM (streaming)
                        prompt = self._build_prompt(text)
                        response_parts = []
                        sentence_buffer = ""
                        
                        for token in self.llm.generate(prompt, stream=True):
                            sentence_buffer += token
                            response_parts.append(token)
                            
                            # Détecter fin de phrase
                            if any(p in sentence_buffer for p in ['. ', '! ', '? ', '\n']):
                                # 4. Parler immédiatement (streaming TTS)
                                self.tts.speak(sentence_buffer.strip(), async_mode=True)
                                sentence_buffer = ""
                        
                        # Parler dernier bout si nécessaire
                        if sentence_buffer.strip():
                            self.tts.speak(sentence_buffer.strip(), async_mode=True)
                        
                        # Sauvegarder contexte
                        full_response = "".join(response_parts)
                        self.context_history.append({
                            "query": text,
                            "response": full_response,
                            "timestamp": time.time()
                        })
                        
                        logger.success(f"🤖 Réponse: {full_response}")
                
                except Exception as e:
                    logger.error(f"❌ Erreur pipeline: {e}")
                
                # Reset buffer
                audio_buffer = []
                silence_duration = 0
    
    def _build_prompt(self, query: str) -> str:
        """Construit le prompt pour le LLM"""
        mode_config = ASSISTANCE_MODES[self.current_mode]
        
        # Context récent
        context = ""
        if self.context_history:
            last_exchanges = self.context_history[-3:]  # 3 derniers échanges
            context = "\n".join([
                f"Q: {ex['query']}\nR: {ex['response']}"
                for ex in last_exchanges
            ])
        
        prompt = mode_config["prompt_template"].format(
            context=context or "Aucun contexte précédent",
            query=query
        )
        
        return prompt
    
    def start(self):
        """Démarre Autodible"""
        logger.info("🚀 Démarrage Autodible...")
        
        self.is_running = True
        self.audio.start()
        
        # Pipeline dans thread séparé
        self.pipeline_thread = threading.Thread(target=self.process_audio_realtime, daemon=True)
        self.pipeline_thread.start()
        
        logger.success("✅ Autodible actif - Écoutez dans votre oreillette!")
    
    def stop(self):
        """Arrête Autodible"""
        logger.info("⏹️ Arrêt Autodible...")
        
        self.is_running = False
        self.audio.stop()
        
        if hasattr(self, 'pipeline_thread'):
            self.pipeline_thread.join(timeout=2.0)
        
        logger.success("✅ Autodible arrêté")


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def main():
    """Point d'entrée principal"""
    logger.info("="*60)
    logger.info("🎧 AUTODIBLE - Assistant Audio Intelligent")
    logger.info("="*60)
    
    try:
        # Initialiser
        engine = AutodibleEngine()
        
        # Démarrer
        engine.start()
        
        logger.info("💡 Parlez dans votre micro, les suggestions arriveront dans votre oreillette")
        logger.info("   Appuyez sur Ctrl+C pour arrêter")
        
        # Garder actif
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\n⏹️ Interruption utilisateur")
        engine.stop()
    
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}", exc_info=True)
    
    finally:
        logger.info("👋 Au revoir!")


if __name__ == "__main__":
    main()
