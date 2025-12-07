"""
Module de gestion et visualisation de fichiers binaires scientifiques
Supporte tous les formats binaires avec visualiseur hex/ASCII intégré
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import io
import json

# ===============================================
# FORMATS BINAIRES SCIENTIFIQUES SUPPORTÉS
# ===============================================

SCIENTIFIC_BINARY_FORMATS = {
    # Biologie & Médecine
    '.dcm': 'DICOM - Medical Imaging',
    '.nii': 'NIfTI - Neuroimaging',
    '.nii.gz': 'Compressed NIfTI',
    '.mgh': 'MGH - FreeSurfer Format',
    '.mnc': 'MINC - Medical Image NetCDF',
    '.edf': 'European Data Format - EEG/Sleep',
    '.fcs': 'Flow Cytometry Standard',
    '.ab1': 'Applied Biosystems Sequencing',
    '.scn': 'Leica SCN - Whole Slide Imaging',
    '.svs': 'Aperio SVS - Pathology',
    '.czi': 'Carl Zeiss Image Format',
    '.lif': 'Leica Image File',
    '.nd2': 'Nikon NIS-Elements',
    '.oib': 'Olympus Image Binary',
    '.zvi': 'Zeiss Vision Image',
    
    # Physique & Astronomie
    '.fits': 'Flexible Image Transport System',
    '.fit': 'FITS variant',
    '.fts': 'FITS variant',
    '.root': 'ROOT - CERN Data Format',
    '.hdf': 'Hierarchical Data Format',
    '.hdf5': 'HDF5',
    '.h5': 'HDF5 short',
    '.nexus': 'NeXus scientific data',
    '.nxs': 'NeXus short',
    '.spec': 'SPEC - Certified Scientific Software',
    
    # Géosciences & Géophysique
    '.sgy': 'SEG-Y - Seismic Data',
    '.segy': 'SEG-Y variant',
    '.seg': 'Seismic data',
    '.grd': 'Grid/Raster Data',
    '.nc': 'NetCDF - Network Common Data',
    '.cdf': 'Common Data Format',
    '.grb': 'GRIB - Meteorological data',
    '.grib': 'GRIB variant',
    '.grib2': 'GRIB version 2',
    '.bufr': 'Binary Universal Form',
    '.las': 'LAS - Well Log Data',
    '.dat': 'Generic binary data',
    '.bin': 'Generic binary',
    '.raw': 'Raw binary data',
    
    # Chimie & Matériaux
    '.cif': 'Crystallographic Information File',
    '.pdb': 'Protein Data Bank',
    '.mol2': 'Tripos Mol2',
    '.xyz': 'XYZ Chemical Format',
    '.sdf': 'Structure Data File',
    '.mtz': 'MTZ - Crystallography',
    '.ccp4': 'CCP4 - Protein Crystallography',
    '.map': 'Electron Density Map',
    '.mrc': 'MRC - Electron Microscopy',
    '.dm3': 'Gatan Digital Micrograph',
    '.dm4': 'Gatan DM4',
    '.ser': 'TIA Series',
    '.emi': 'FEI Titan',
    
    # Spectroscopie & Analyse
    '.spc': 'Galactic SPC Spectroscopy',
    '.sp': 'PerkinElmer Spectrum',
    '.dpt': 'Bruker OPUS',
    '.dx': 'JCAMP-DX',
    '.jdx': 'JCAMP-DX variant',
    '.asc': 'ASCII Spectrum',
    '.0': 'Bruker NMR',
    '.1r': 'Bruker NMR processed',
    '.2rr': 'Bruker 2D NMR',
    '.fid': 'Varian/Agilent NMR',
    '.nmr': 'Generic NMR',
    
    # Microscopie & Imagerie
    '.tif': 'Tagged Image File',
    '.tiff': 'TIFF variant',
    '.lsm': 'Zeiss LSM - Confocal',
    '.stk': 'MetaMorph Stack',
    '.ims': 'Imaris Image',
    '.vsi': 'Olympus VSI',
    '.scn': 'Leica Aperio',
    '.bif': 'Ventana BIF',
    
    # Électronique & Mesure
    '.wfm': 'Tektronix Waveform',
    '.isf': 'Tektronix ISF',
    '.tdms': 'National Instruments TDMS',
    '.mat': 'MATLAB Binary',
    '.labview': 'LabVIEW Data',
    
    # Base de données & Archives
    '.db': 'SQLite Database',
    '.sqlite': 'SQLite',
    '.sqlite3': 'SQLite3',
    '.mdb': 'Microsoft Access',
    '.accdb': 'Access 2007+',
    
    # Formats compressés
    '.gz': 'Gzip compressed',
    '.bz2': 'Bzip2 compressed',
    '.xz': 'XZ compressed',
    '.lz4': 'LZ4 compressed',
    '.zst': 'Zstandard compressed',
    
    # Formats propriétaires scientifiques
    '.wsp': 'Whole Slide Panoramic',
    '.mrxs': 'MIRAX Slide',
    '.ndpi': 'Hamamatsu NanoZoomer',
    '.vms': 'Hamamatsu VMS',
    '.rrd': 'Round Robin Database',
    '.pcap': 'Packet Capture',
    '.pcapng': 'PCAP Next Generation',
}

# ===============================================
# VISUALISEUR HEXADÉCIMAL/ASCII
# ===============================================

class BinaryFileViewer:
    """Visualiseur hexadécimal et ASCII pour fichiers binaires"""
    
    def __init__(self, file_data: bytes, filename: str = ""):
        self.data = file_data
        self.filename = filename
        self.size = len(file_data)
        
    def hex_dump(self, start: int = 0, length: int = 512, bytes_per_line: int = 16) -> str:
        """
        Génère un dump hexadécimal avec ASCII à côté
        Format: OFFSET | HEX BYTES | ASCII
        """
        end = min(start + length, self.size)
        lines = []
        
        lines.append(f"📄 Fichier: {self.filename}")
        lines.append(f"📊 Taille: {self.size:,} bytes ({self._format_size(self.size)})")
        lines.append(f"📍 Affichage: {start:08X} → {end:08X} ({end-start} bytes)")
        lines.append("=" * 80)
        lines.append(f"{'OFFSET':<10} | {'HEX':<48} | {'ASCII':<16}")
        lines.append("-" * 80)
        
        for offset in range(start, end, bytes_per_line):
            # Offset
            line_offset = f"{offset:08X}"
            
            # Hex bytes
            chunk_size = min(bytes_per_line, end - offset)
            hex_bytes = ' '.join(f"{b:02X}" for b in self.data[offset:offset+chunk_size])
            hex_bytes = hex_bytes.ljust(48)  # Padding pour alignement
            
            # ASCII representation
            ascii_chars = ''.join(
                chr(b) if 32 <= b < 127 else '.'
                for b in self.data[offset:offset+chunk_size]
            )
            
            lines.append(f"{line_offset} | {hex_bytes} | {ascii_chars}")
        
        if end < self.size:
            lines.append(f"\n... ({self.size - end:,} bytes restants)")
            
        return '\n'.join(lines)
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyse la structure du fichier binaire"""
        analysis = {
            'size': self.size,
            'size_formatted': self._format_size(self.size),
            'header': self._analyze_header(),
            'statistics': self._compute_statistics(),
            'entropy': self._compute_entropy(),
            'patterns': self._detect_patterns(),
            'magic_bytes': self._detect_magic_bytes()
        }
        return analysis
    
    def _analyze_header(self, header_size: int = 64) -> Dict[str, Any]:
        """Analyse les premiers bytes (header)"""
        header_bytes = self.data[:min(header_size, self.size)]
        
        # Détection de texte ASCII
        ascii_count = sum(1 for b in header_bytes if 32 <= b < 127)
        is_text_like = ascii_count > len(header_bytes) * 0.7
        
        # Première ligne de texte si applicable
        first_line = ""
        if is_text_like:
            try:
                first_line = header_bytes.decode('utf-8', errors='ignore').split('\n')[0][:80]
            except:
                pass
        
        return {
            'bytes': header_bytes.hex()[:128],
            'size': len(header_bytes),
            'is_text_like': is_text_like,
            'ascii_ratio': ascii_count / len(header_bytes) if header_bytes else 0,
            'first_line': first_line
        }
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Calcule des statistiques sur les bytes"""
        if self.size == 0:
            return {}
        
        # Convertir en numpy pour statistiques
        data_array = np.frombuffer(self.data[:min(10000, self.size)], dtype=np.uint8)
        
        return {
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'std': float(np.std(data_array)),
            'min': int(np.min(data_array)),
            'max': int(np.max(data_array)),
            'unique_bytes': len(np.unique(data_array))
        }
    
    def _compute_entropy(self, sample_size: int = 10000) -> float:
        """Calcule l'entropie de Shannon (désordre)"""
        sample = self.data[:min(sample_size, self.size)]
        if not sample:
            return 0.0
        
        # Fréquence de chaque byte
        byte_counts = np.bincount(np.frombuffer(sample, dtype=np.uint8), minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(sample)
        
        # Entropie de Shannon
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    
    def _detect_patterns(self) -> List[str]:
        """Détecte des patterns communs"""
        patterns = []
        sample = self.data[:1024]
        
        # Vérifier des patterns connus
        if b'<?xml' in sample:
            patterns.append('XML header detected')
        if b'{' in sample and b'"' in sample:
            patterns.append('Possible JSON data')
        if sample.startswith(b'\x89PNG'):
            patterns.append('PNG image')
        if sample.startswith(b'\xff\xd8\xff'):
            patterns.append('JPEG image')
        if sample.startswith(b'PK'):
            patterns.append('ZIP/Office document')
        if sample.startswith(b'%PDF'):
            patterns.append('PDF document')
        if b'\x00' * 10 in sample:
            patterns.append('Contains null byte sequences')
        if sample.startswith(b'SIMPLE'):
            patterns.append('FITS astronomical data')
        if sample.startswith(b'HDF'):
            patterns.append('HDF5 scientific data')
            
        return patterns
    
    def _detect_magic_bytes(self) -> str:
        """Détecte le magic number du fichier"""
        magic_numbers = {
            b'\x89PNG\r\n\x1a\n': 'PNG Image',
            b'\xff\xd8\xff': 'JPEG Image',
            b'GIF87a': 'GIF Image (87a)',
            b'GIF89a': 'GIF Image (89a)',
            b'BM': 'BMP Image',
            b'II*\x00': 'TIFF Image (little endian)',
            b'MM\x00*': 'TIFF Image (big endian)',
            b'RIFF': 'RIFF Container (AVI/WAV)',
            b'%PDF': 'PDF Document',
            b'PK\x03\x04': 'ZIP Archive',
            b'PK\x05\x06': 'ZIP Empty Archive',
            b'\x1f\x8b': 'GZIP Compressed',
            b'SIMPLE': 'FITS Astronomical',
            b'\x89HDF\r\n\x1a\n': 'HDF5 Scientific Data',
            b'CDF': 'NetCDF',
            b'SQLite format 3': 'SQLite Database',
            b'\x00\x00\x00\x0cjP  ': 'JPEG 2000',
            b'OggS': 'Ogg Media',
            b'\x1a\x45\xdf\xa3': 'Matroska/WebM',
        }
        
        for magic, description in magic_numbers.items():
            if self.data.startswith(magic):
                return f"{description} (0x{magic.hex()})"
        
        # Magic bytes génériques
        first_bytes = self.data[:8].hex()
        return f"Unknown (0x{first_bytes})"
    
    def _format_size(self, size: int) -> str:
        """Formate la taille en unités lisibles"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"
    
    def extract_text_strings(self, min_length: int = 4) -> List[str]:
        """Extrait les chaînes de texte ASCII lisibles"""
        strings = []
        current = []
        
        for byte in self.data:
            if 32 <= byte < 127:  # Caractère ASCII imprimable
                current.append(chr(byte))
            else:
                if len(current) >= min_length:
                    strings.append(''.join(current))
                current = []
        
        if len(current) >= min_length:
            strings.append(''.join(current))
        
        return strings[:100]  # Limiter à 100 premières chaînes
    
    def to_numpy_array(self) -> Optional[np.ndarray]:
        """Tente de convertir en array numpy"""
        try:
            # Essayer différents types de données
            for dtype in [np.float32, np.float64, np.int32, np.int16, np.uint8]:
                try:
                    arr = np.frombuffer(self.data, dtype=dtype)
                    if len(arr) > 0:
                        return arr
                except:
                    continue
        except:
            pass
        return None

# ===============================================
# GESTIONNAIRE DE FICHIERS BINAIRES
# ===============================================

class BinaryFileHandler:
    """Gestionnaire principal pour tous les fichiers binaires"""
    
    @staticmethod
    def identify_format(filename: str, data: bytes) -> Dict[str, Any]:
        """Identifie le format d'un fichier binaire"""
        ext = Path(filename).suffix.lower()
        
        info = {
            'filename': filename,
            'extension': ext,
            'format_name': SCIENTIFIC_BINARY_FORMATS.get(ext, 'Unknown Binary Format'),
            'size': len(data),
            'is_scientific': ext in SCIENTIFIC_BINARY_FORMATS,
        }
        
        # Créer viewer pour analyse
        viewer = BinaryFileViewer(data, filename)
        info['magic_bytes'] = viewer._detect_magic_bytes()
        info['analysis'] = viewer.analyze_structure()
        
        return info
    
    @staticmethod
    def process_file(uploaded_file) -> Dict[str, Any]:
        """Traite un fichier uploadé (Streamlit ou bytes)"""
        if hasattr(uploaded_file, 'read'):
            # Streamlit UploadedFile
            filename = uploaded_file.name
            data = uploaded_file.read()
            uploaded_file.seek(0)  # Reset pour réutilisation
        else:
            # Bytes directs
            filename = "unknown.bin"
            data = uploaded_file
        
        # Identifier le format
        file_info = BinaryFileHandler.identify_format(filename, data)
        
        # Créer le viewer
        viewer = BinaryFileViewer(data, filename)
        
        return {
            'info': file_info,
            'viewer': viewer,
            'data': data
        }
    
    @staticmethod
    def get_supported_formats() -> str:
        """Retourne la liste des formats supportés"""
        categories = {
            'Biologie & Médecine': [k for k in SCIENTIFIC_BINARY_FORMATS.keys() if k in ['.dcm', '.nii', '.edf', '.fcs', '.ab1']],
            'Physique & Astronomie': [k for k in SCIENTIFIC_BINARY_FORMATS.keys() if k in ['.fits', '.root', '.hdf5']],
            'Géosciences': [k for k in SCIENTIFIC_BINARY_FORMATS.keys() if k in ['.sgy', '.nc', '.grib', '.las']],
            'Chimie': [k for k in SCIENTIFIC_BINARY_FORMATS.keys() if k in ['.cif', '.pdb', '.mol2']],
            'Spectroscopie': [k for k in SCIENTIFIC_BINARY_FORMATS.keys() if k in ['.spc', '.dx', '.fid']],
            'Microscopie': [k for k in SCIENTIFIC_BINARY_FORMATS.keys() if k in ['.lsm', '.dm3', '.mrc']],
        }
        
        text = "### 📚 Formats Scientifiques Supportés\n\n"
        for category, formats in categories.items():
            if formats:
                text += f"**{category}**: {', '.join(formats[:5])}\n"
        text += f"\n**Total: {len(SCIENTIFIC_BINARY_FORMATS)} formats**"
        return text

# ===============================================
# FONCTIONS UTILITAIRES
# ===============================================

def create_hex_viewer_ui(viewer: BinaryFileViewer, start_offset: int = 0, length: int = 512):
    """Génère l'interface du visualiseur hexadécimal"""
    return viewer.hex_dump(start=start_offset, length=length)

def analyze_binary_file(data: bytes, filename: str = "") -> str:
    """Analyse complète d'un fichier binaire"""
    viewer = BinaryFileViewer(data, filename)
    analysis = viewer.analyze_structure()
    
    report = f"""
## 🔬 Analyse Complète du Fichier Binaire

**📄 Fichier**: {filename}
**📊 Taille**: {analysis['size_formatted']}
**🔢 Magic Bytes**: {analysis['magic_bytes']}
**📈 Entropie**: {analysis['entropy']:.2f} / 8.0 (compression/chiffrement si > 7.5)

### 📋 Statistiques des Bytes
- **Moyenne**: {analysis['statistics'].get('mean', 0):.2f}
- **Médiane**: {analysis['statistics'].get('median', 0):.2f}
- **Écart-type**: {analysis['statistics'].get('std', 0):.2f}
- **Min/Max**: {analysis['statistics'].get('min', 0)} / {analysis['statistics'].get('max', 0)}
- **Bytes uniques**: {analysis['statistics'].get('unique_bytes', 0)} / 256

### 🎯 Patterns Détectés
{chr(10).join('- ' + p for p in analysis['patterns']) if analysis['patterns'] else '- Aucun pattern spécifique détecté'}

### 📝 Header (64 premiers bytes)
- **Type**: {'Texte ASCII' if analysis['header']['is_text_like'] else 'Binaire pur'}
- **Ratio ASCII**: {analysis['header']['ascii_ratio']:.1%}
{f"- **Première ligne**: {analysis['header']['first_line']}" if analysis['header']['first_line'] else ''}

### 🔍 Dump Hexadécimal (512 premiers bytes)
```
{viewer.hex_dump(length=512)}
```

### 📝 Chaînes de Texte Extraites
{chr(10).join(f'- {s[:80]}' for s in viewer.extract_text_strings()[:20]) if viewer.extract_text_strings() else '- Aucune chaîne lisible détectée'}
"""
    return report
