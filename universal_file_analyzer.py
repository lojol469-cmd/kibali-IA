"""
🔬 Analyseur Universel de Fichiers Binaires
Détecte automatiquement la nature de tout fichier et applique les traitements appropriés
"""

import os
import magic
import struct
import hashlib
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Tuple
import json

class UniversalFileAnalyzer:
    """Analyseur intelligent de fichiers avec détection automatique"""
    
    def __init__(self):
        self.file_signatures = {
            # Formats scientifiques
            b'ERT': {'type': 'geophysics_ert', 'handler': 'analyze_ert_data'},
            b'RIFF': {'type': 'audio_wav', 'handler': 'analyze_audio'},
            b'ID3': {'type': 'audio_mp3', 'handler': 'analyze_audio'},
            b'\x89PNG': {'type': 'image_png', 'handler': 'analyze_image'},
            b'\xFF\xD8\xFF': {'type': 'image_jpeg', 'handler': 'analyze_image'},
            b'GIF8': {'type': 'image_gif', 'handler': 'analyze_image'},
            b'%PDF': {'type': 'document_pdf', 'handler': 'analyze_pdf'},
            
            # Formats géophysiques spécifiques
            b'SEG-Y': {'type': 'seismic_segy', 'handler': 'analyze_seismic'},
            b'LAS': {'type': 'well_log_las', 'handler': 'analyze_well_log'},
            
            # Formats 3D
            b'STL': {'type': 'model_3d_stl', 'handler': 'analyze_3d_model'},
            b'OBJ': {'type': 'model_3d_obj', 'handler': 'analyze_3d_model'},
            b'PLY': {'type': 'pointcloud_ply', 'handler': 'analyze_pointcloud'},
            
            # Formats de données brutes
            b'HDF': {'type': 'scientific_hdf5', 'handler': 'analyze_hdf5'},
            b'\x00\x01\x00\x00': {'type': 'binary_data', 'handler': 'analyze_binary'},
        }
        
    def detect_file_type(self, file_path: str) -> Dict[str, Any]:
        """Détecte le type de fichier via signatures binaires et extension"""
        
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'extension': Path(file_path).suffix.lower(),
            'detected_type': 'unknown',
            'handler': None,
            'confidence': 0.0,
            'magic_type': None,
            'hex_signature': None,
            'metadata': {}
        }
        
        # 1. Lecture des premiers octets (signature)
        with open(file_path, 'rb') as f:
            header = f.read(512)
            result['hex_signature'] = header[:16].hex()
            
            # Vérifier les signatures connues
            for signature, info in self.file_signatures.items():
                if header.startswith(signature) or signature in header[:100]:
                    result['detected_type'] = info['type']
                    result['handler'] = info['handler']
                    result['confidence'] = 0.95
                    break
        
        # 2. Utiliser libmagic si disponible
        try:
            mime = magic.from_file(file_path, mime=True)
            result['magic_type'] = mime
            
            # Mapper MIME vers nos types
            mime_mapping = {
                'application/x-ert': 'geophysics_ert',
                'application/segy': 'seismic_segy',
                'application/octet-stream': 'binary_data',
                'text/plain': 'text_ascii',
                'application/x-hdf': 'scientific_hdf5',
            }
            
            if result['detected_type'] == 'unknown' and mime in mime_mapping:
                result['detected_type'] = mime_mapping[mime]
                result['confidence'] = 0.7
                
        except Exception as e:
            print(f"⚠️ Magic detection failed: {e}")
        
        # 3. Détection par extension (fallback)
        if result['detected_type'] == 'unknown':
            ext_mapping = {
                '.dat': 'binary_data',
                '.ert': 'geophysics_ert',
                '.bin': 'binary_data',
                '.sgy': 'seismic_segy',
                '.las': 'well_log_las',
                '.xyz': 'pointcloud_xyz',
                '.csv': 'tabular_csv',
                '.nc': 'netcdf',
                '.grd': 'grid_data',
            }
            
            if result['extension'] in ext_mapping:
                result['detected_type'] = ext_mapping[result['extension']]
                result['handler'] = f"analyze_{ext_mapping[result['extension']].split('_')[-1]}"
                result['confidence'] = 0.5
        
        # 4. Calculer le hash MD5
        result['md5'] = self.calculate_hash(file_path)
        
        return result
    
    def calculate_hash(self, file_path: str) -> str:
        """Calcule le hash MD5 du fichier"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def analyze_ert_data(self, file_path: str, detection_info: Dict) -> Dict[str, Any]:
        """Analyse spécifique pour données ERT (Tomographie de Résistivité Électrique)"""
        
        result = {
            'type': 'geophysics_ert',
            'analysis': {},
            'visualizations': [],
            'recommendations': [],
            'metadata': detection_info
        }
        
        try:
            # Tenter de lire le fichier ERT
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Recherche de patterns communs dans les fichiers ERT
            # Format typique: nombre de mesures, positions électrodes, résistivités
            
            # Essayer de parser comme texte ASCII d'abord
            try:
                text_data = data.decode('utf-8', errors='ignore')
                lines = text_data.strip().split('\n')
                
                result['analysis']['total_lines'] = len(lines)
                result['analysis']['format'] = 'ascii_text'
                
                # Extraire statistiques basiques
                numeric_values = []
                for line in lines:
                    try:
                        values = [float(x) for x in line.split() if x.replace('.', '').replace('-', '').isdigit()]
                        numeric_values.extend(values)
                    except:
                        continue
                
                if numeric_values:
                    result['analysis']['num_measurements'] = len(numeric_values)
                    result['analysis']['min_value'] = float(np.min(numeric_values))
                    result['analysis']['max_value'] = float(np.max(numeric_values))
                    result['analysis']['mean_value'] = float(np.mean(numeric_values))
                    result['analysis']['std_value'] = float(np.std(numeric_values))
                    
                    result['recommendations'].append({
                        'action': 'ert_inversion',
                        'description': 'Effectuer une inversion de résistivité pour obtenir un modèle 2D/3D',
                        'tool': 'PyGIMLi ou ResIPy',
                        'priority': 'high'
                    })
                    
                    result['recommendations'].append({
                        'action': 'quality_check',
                        'description': 'Vérifier la qualité des données (outliers, erreurs de contact)',
                        'tool': 'Statistical analysis',
                        'priority': 'medium'
                    })
                    
            except UnicodeDecodeError:
                # Format binaire
                result['analysis']['format'] = 'binary'
                result['analysis']['file_size_bytes'] = len(data)
                
                # Essayer de détecter des patterns binaires
                # Les fichiers ERT binaires ont souvent des floats (4 ou 8 bytes)
                if len(data) % 4 == 0:
                    # Essayer de lire comme floats 32-bit
                    try:
                        num_floats = len(data) // 4
                        floats = struct.unpack(f'{num_floats}f', data)
                        valid_floats = [f for f in floats if not (np.isnan(f) or np.isinf(f))]
                        
                        if valid_floats:
                            result['analysis']['num_values'] = len(valid_floats)
                            result['analysis']['value_range'] = [float(np.min(valid_floats)), float(np.max(valid_floats))]
                    except:
                        pass
            
            # Recommandations de traitement
            result['recommendations'].append({
                'action': 'web_search',
                'description': f'Rechercher des informations sur le format ERT et les outils de traitement',
                'query': f'ERT tomography data format {detection_info["extension"]} processing tools',
                'priority': 'high'
            })
            
            result['visualizations'].append({
                'type': 'resistivity_profile',
                'description': 'Profil de résistivité apparente',
                'requires': ['matplotlib', 'numpy']
            })
            
        except Exception as e:
            result['error'] = str(e)
            result['recommendations'].append({
                'action': 'manual_inspection',
                'description': f'Inspection manuelle requise: {str(e)}',
                'priority': 'high'
            })
        
        return result
    
    def analyze_binary(self, file_path: str, detection_info: Dict) -> Dict[str, Any]:
        """Analyse générique pour données binaires"""
        
        result = {
            'type': 'binary_data',
            'analysis': {},
            'patterns': [],
            'recommendations': []
        }
        
        with open(file_path, 'rb') as f:
            data = f.read(min(10000, os.path.getsize(file_path)))  # Lire premiers 10KB
        
        # Statistiques basiques
        result['analysis']['entropy'] = self.calculate_entropy(data)
        result['analysis']['null_bytes_ratio'] = data.count(b'\x00') / len(data)
        result['analysis']['printable_ratio'] = sum(1 for b in data if 32 <= b < 127) / len(data)
        
        # Détecter des patterns
        if result['analysis']['printable_ratio'] > 0.7:
            result['patterns'].append('mostly_ascii')
            result['recommendations'].append({
                'action': 'try_text_parsing',
                'description': 'Le fichier contient beaucoup de texte ASCII, essayer de le lire comme texte',
                'priority': 'high'
            })
        
        if result['analysis']['entropy'] > 7.5:
            result['patterns'].append('high_entropy_compressed_or_encrypted')
            result['recommendations'].append({
                'action': 'check_compression',
                'description': 'Entropie élevée, possiblement compressé ou chiffré',
                'priority': 'medium'
            })
        
        # Chercher des nombres (floats/ints)
        try:
            # Essayer différents formats
            for fmt, size, name in [('f', 4, 'float32'), ('d', 8, 'float64'), ('i', 4, 'int32'), ('q', 8, 'int64')]:
                if len(data) % size == 0:
                    count = len(data) // size
                    try:
                        values = struct.unpack(f'{count}{fmt}', data[:count*size])
                        valid = [v for v in values if not (np.isnan(v) if 'float' in name else False)]
                        if len(valid) > count * 0.8:  # >80% valeurs valides
                            result['patterns'].append(f'numeric_{name}')
                            result['recommendations'].append({
                                'action': f'parse_as_{name}',
                                'description': f'Fichier semble contenir des {name}',
                                'count': len(valid),
                                'priority': 'high'
                            })
                            break
                    except:
                        pass
        except Exception as e:
            pass
        
        # Recommandation de recherche web
        result['recommendations'].append({
            'action': 'web_search',
            'description': 'Rechercher des informations sur ce type de fichier',
            'query': f'binary file format {detection_info["extension"]} {detection_info["hex_signature"]}',
            'priority': 'high'
        })
        
        return result
    
    def calculate_entropy(self, data: bytes) -> float:
        """Calcule l'entropie de Shannon"""
        if not data:
            return 0
        
        # Fréquence de chaque byte
        frequencies = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = frequencies / len(data)
        probabilities = probabilities[probabilities > 0]
        
        return float(-np.sum(probabilities * np.log2(probabilities)))
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Point d'entrée principal pour analyser n'importe quel fichier"""
        
        # 1. Détecter le type
        detection = self.detect_file_type(file_path)
        
        # 2. Appeler le handler approprié
        if detection['handler'] and hasattr(self, detection['handler']):
            handler_method = getattr(self, detection['handler'])
            analysis = handler_method(file_path, detection)
        else:
            # Handler générique binaire
            analysis = self.analyze_binary(file_path, detection)
        
        # 3. Ajouter des métadonnées générales
        analysis['detection'] = detection
        analysis['timestamp'] = __import__('datetime').datetime.now().isoformat()
        
        return analysis


# ============================================
# Fonction helper pour intégration Streamlit
# ============================================

def analyze_uploaded_file(uploaded_file, web_search_func=None, llm_client=None):
    """
    Analyse un fichier uploadé dans Streamlit et retourne un rapport complet
    
    Args:
        uploaded_file: Objet file uploadé par Streamlit
        web_search_func: Fonction de recherche web (optionnel)
        llm_client: Client LLM pour analyse IA (optionnel)
    
    Returns:
        Dict avec analyse complète et recommandations
    """
    import tempfile
    
    # Sauvegarder temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Analyser
        analyzer = UniversalFileAnalyzer()
        analysis = analyzer.analyze_file(tmp_path)
        
        # Enrichir avec recherche web si disponible
        if web_search_func and analysis.get('recommendations'):
            web_recs = [r for r in analysis['recommendations'] if r.get('action') == 'web_search']
            for rec in web_recs:
                query = rec.get('query', '')
                if query:
                    web_results = web_search_func(query, max_results=3)
                    rec['web_results'] = web_results
        
        # Enrichir avec analyse IA si disponible
        if llm_client:
            analysis['ai_interpretation'] = generate_ai_interpretation(analysis, llm_client)
        
        return analysis
        
    finally:
        # Nettoyer
        try:
            os.unlink(tmp_path)
        except:
            pass


def generate_ai_interpretation(analysis: Dict, llm_client) -> str:
    """Génère une interprétation IA de l'analyse"""
    
    prompt = f"""Analyse ce fichier et fournis une interprétation experte:

Type détecté: {analysis['detection']['detected_type']}
Confiance: {analysis['detection']['confidence']*100:.1f}%
Taille: {analysis['detection']['file_size'] / 1024:.1f} KB
Extension: {analysis['detection']['extension']}

Analyse technique:
{json.dumps(analysis.get('analysis', {}), indent=2)}

Recommandations:
{json.dumps(analysis.get('recommendations', []), indent=2)}

Fournis:
1. Une explication claire de ce type de fichier
2. Les outils/logiciels recommandés pour le traiter
3. Les étapes de traitement suggérées
4. Les précautions à prendre

Réponds en français, de manière détaillée et professionnelle."""

    try:
        response = llm_client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur génération IA: {str(e)}"


if __name__ == "__main__":
    # Test rapide
    analyzer = UniversalFileAnalyzer()
    print("🔬 Analyseur Universel de Fichiers - Test")
    print("Signatures supportées:", len(analyzer.file_signatures))
