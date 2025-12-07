# 🔬 Système d'Analyse de Fichiers Binaires Scientifiques

## ✅ Installation Complète

Kibali peut maintenant analyser **TOUS les types de fichiers binaires** sans restriction, avec un focus spécial sur les formats scientifiques.

### 🎯 Capacités

#### 1. **Upload Universel**
- ✅ **Tous les formats acceptés** (`type=None` sur tous les uploaders)
- ✅ Plus de restrictions d'extensions
- ✅ Fichiers binaires, texte, compressés, propriétaires, etc.

#### 2. **Visualiseur Hexadécimal/ASCII Intégré**
```
OFFSET     | HEX                                              | ASCII           
--------------------------------------------------------------------------------
00000000 | 89 50 4E 47 0D 0A 1A 0A 00 00 00 0D 49 48 44 52 | .PNG........IHDR
00000010 | 00 00 01 90 00 00 01 40 08 06 00 00 00 E1 8C 5F | .......@......._
```

#### 3. **92 Formats Scientifiques Supportés**

##### 🧬 Biologie & Médecine (15 formats)
- `.dcm` - DICOM (imagerie médicale)
- `.nii`, `.nii.gz` - NIfTI (neuroimagerie)
- `.edf` - EEG/Sleep data
- `.fcs` - Flow Cytometry
- `.ab1` - Séquençage ADN
- `.svs`, `.scn` - Pathologie
- `.czi`, `.lif`, `.nd2` - Microscopie

##### 🌌 Physique & Astronomie (10 formats)
- `.fits`, `.fit` - Astronomical data
- `.root` - CERN data
- `.hdf5`, `.h5`, `.hdf` - HDF5
- `.nexus`, `.nxs` - NeXus

##### 🌍 Géosciences (12 formats)
- `.sgy`, `.segy` - Données sismiques SEG-Y
- `.nc`, `.cdf` - NetCDF
- `.grib`, `.grib2` - Météo
- `.las` - Logs de puits
- `.dat`, `.bin`, `.raw` - Binaires génériques

##### ⚗️ Chimie & Matériaux (11 formats)
- `.cif` - Cristallographie
- `.pdb` - Protéines
- `.mol2`, `.sdf` - Molécules
- `.mtz`, `.ccp4` - Cristallographie protéines
- `.mrc` - Microscopie électronique
- `.dm3`, `.dm4` - Gatan

##### 📊 Spectroscopie (10 formats)
- `.spc` - Galactic SPC
- `.dx`, `.jdx` - JCAMP-DX
- `.0`, `.1r`, `.2rr` - Bruker NMR
- `.fid` - Varian/Agilent NMR

##### 🔬 Microscopie (9 formats)
- `.lsm` - Zeiss confocal
- `.ims` - Imaris
- `.vsi` - Olympus
- `.bif` - Ventana

##### 📡 Électronique & Mesure (5 formats)
- `.wfm`, `.isf` - Tektronix
- `.tdms` - National Instruments
- `.mat` - MATLAB

##### 💾 Bases de Données (5 formats)
- `.db`, `.sqlite`, `.sqlite3`
- `.mdb`, `.accdb` - Microsoft Access

##### 📦 Compression (5 formats)
- `.gz`, `.bz2`, `.xz`, `.lz4`, `.zst`

##### 🔧 Autres Formats (10+ formats)
- `.pcap`, `.pcapng` - Captures réseau
- `.rrd` - Round Robin Database
- Tous les formats propriétaires d'imagerie médicale

## 🎨 Interface Utilisateur

### Dans le Chat RAG
1. **Upload** : Zone "📄 Ajouter des documents" accepte TOUT
2. **Analyse automatique** : Fichier détecté → Analyse complète
3. **Panneau interactif** avec :
   - 📊 Informations (taille, type, magic bytes, entropie)
   - 🔍 Visualiseur hex (collapsible)
   - 📊 Statistiques détaillées
   - 📝 Chaînes de texte extraites
   - 🛠️ Actions suggérées

### Exemple d'Affichage
```
🔬 Fichier Binaire Analysé
━━━━━━━━━━━━━━━━━━━━━━━━━
data.fits

📊 Taille: 2.4 MB
🔢 Type: FITS - Astronomical Data
🎯 Magic Bytes: SIMPLE (0x53494d504c45)
📈 Entropie: 7.2/8.0 (données compressées)

🔍 Visualiseur Hexadécimal (512 premiers bytes) ▼
📊 Statistiques Détaillées ▼
📝 Chaînes de Texte Extraites (45 trouvées) ▼

🛠️ Actions Disponibles:
💬 Analyse IA
🔍 Exploration Avancée
📊 Conversion Données
```

## 🤖 Intégration IA

### Contexte Automatique
Quand vous uploadez un fichier binaire, l'IA reçoit automatiquement :
- Structure complète du fichier
- Magic bytes et type détecté
- Statistiques (entropie, distribution des bytes)
- Patterns détectés
- Chaînes de texte extraites
- Premiers bytes en hexadécimal

### Questions Possibles
```
"Analyse ce fichier binaire"
"Que contient ce fichier FITS?"
"Montre-moi les bytes à l'offset 0x1000"
"Convertis les données en array numpy"
"Extrait les métadonnées"
"Quelle est la structure de ce fichier?"
```

## 🔧 Architecture Technique

### Modules
```
kibali-IA/
├── app.py (intégration principale)
└── binary_file_handler.py
    ├── SCIENTIFIC_BINARY_FORMATS (92 formats)
    ├── BinaryFileViewer (visualiseur hex/ASCII)
    ├── BinaryFileHandler (gestionnaire principal)
    └── analyze_binary_file() (analyse complète)
```

### Fonctionnalités du Visualiseur

#### `BinaryFileViewer`
- ✅ Dump hexadécimal avec ASCII
- ✅ Détection de magic bytes
- ✅ Calcul d'entropie de Shannon
- ✅ Statistiques des bytes (moyenne, médiane, écart-type)
- ✅ Détection de patterns (PNG, JPEG, ZIP, FITS, HDF5, etc.)
- ✅ Extraction de chaînes ASCII
- ✅ Conversion vers numpy array
- ✅ Analyse de header

#### Métriques Calculées
1. **Entropie** : 0-8 (0 = répétitif, 8 = aléatoire/chiffré)
2. **Distribution des bytes** : Min, max, moyenne, médiane, écart-type
3. **Bytes uniques** : Combien de valeurs différentes (sur 256)
4. **Ratio ASCII** : Pourcentage de caractères lisibles

## 💡 Cas d'Usage

### 1. Recherche Scientifique
```python
# Upload d'un fichier FITS astronomique
→ Kibali détecte : Format FITS
→ Analyse : Magic bytes, entropie, métadonnées
→ Extraction : Chaînes FITS headers
→ IA répond : "Ce fichier contient des données d'observation..."
```

### 2. Analyse Médical
```python
# Upload d'un DICOM
→ Détection : DICOM Medical Imaging
→ Visualisation hex des headers
→ Extraction des tags DICOM
→ Questions : "Quelle modalité?", "Patient info?"
```

### 3. Géophysique
```python
# Upload d'un SEG-Y sismique
→ Type : SEG-Y Seismic Data
→ Headers : Binary file header + trace headers
→ Analyse : Format, échantillonnage
→ Conversion : Vers numpy pour traitement
```

### 4. Reverse Engineering
```python
# Upload d'un binaire inconnu
→ Magic bytes : Identification automatique
→ Entropie : Détection compression/chiffrement
→ Strings : Extraction chemins, URLs, textes
→ Structure : Hypothèses sur le format
```

## 🚀 Utilisation

### Étape 1 : Upload
- Allez dans l'onglet **💬 Chat RAG + Web**
- Cliquez sur **📄 Ajouter des documents**
- Sélectionnez **n'importe quel fichier binaire**

### Étape 2 : Analyse Automatique
Kibali analyse immédiatement :
- ✅ Identification du format
- ✅ Visualisation hex/ASCII
- ✅ Extraction de métadonnées
- ✅ Affichage dans le chat

### Étape 3 : Interaction
Posez des questions :
- "Que contient ce fichier?"
- "Analyse les headers"
- "Convertis en données lisibles"
- "Affiche plus de hex dump"

## 📚 Exemples de Formats

### Format FITS (Astronomie)
```
Magic Bytes: SIMPLE
Entropie: 5.2/8.0
Patterns: FITS astronomical data
Strings: SIMPLE, BITPIX, NAXIS, EXTEND, DATE-OBS
```

### Format HDF5 (Scientifique)
```
Magic Bytes: \x89HDF\r\n\x1a\n
Entropie: 6.8/8.0
Patterns: HDF5 scientific data
Structure: Hiérarchique avec groupes/datasets
```

### Format DICOM (Médical)
```
Magic Bytes: DICM à offset 128
Patterns: Medical imaging, tags
Strings: Patient info, acquisition params
```

### Format SEG-Y (Sismique)
```
Structure: 3200 bytes text header + 400 bytes binary header + traces
Patterns: Seismic data
Détection: Offset fixes, formats IBM float
```

## 🎯 Avantages

### 1. **Universel**
- Aucune restriction d'extension
- Tous les binaires acceptés
- Pas de "format non supporté"

### 2. **Intelligent**
- Détection automatique de format
- Magic bytes recognition
- Pattern matching avancé

### 3. **Scientifique**
- 92 formats spécialisés
- Métadonnées extraites
- Conversion données possibles

### 4. **Sécurisé**
- Pas d'exécution de code
- Analyse passive uniquement
- Visualisation contrôlée

### 5. **Intégré**
- Dans le chat principal
- IA comprend le contexte
- Réponses intelligentes

## 🔐 Sécurité

- ✅ **Lecture seule** : Aucune modification du fichier
- ✅ **Pas d'exécution** : Analyse statique uniquement
- ✅ **Sandbox** : Traitement isolé
- ✅ **Limite mémoire** : Chunks pour gros fichiers

## 🛠️ Extension Future

### Prévues
- [ ] Parsers spécialisés pour chaque format
- [ ] Extraction automatique de métadonnées
- [ ] Conversion format-to-format
- [ ] Visualisation graphique des données
- [ ] Export vers formats standards
- [ ] Comparaison de fichiers binaires
- [ ] Détection d'anomalies

### Facilement Ajoutables
- Plus de formats scientifiques
- Parsers de formats propriétaires
- Outils de reconstruction de structure
- Décompression automatique
- Décryptage (avec clé)

## 📞 Support

Pour ajouter un nouveau format :
1. Ajoutez l'extension dans `SCIENTIFIC_BINARY_FORMATS`
2. (Optionnel) Ajoutez les magic bytes dans `_detect_magic_bytes()`
3. (Optionnel) Créez un parser spécialisé

## 🎉 Conclusion

Kibali est maintenant capable d'analyser **TOUT fichier binaire**, avec une expertise particulière sur **92 formats scientifiques**. Le visualiseur hexadécimal/ASCII intégré permet une exploration complète, et l'IA peut répondre intelligemment grâce au contexte enrichi.

**Aucun fichier ne lui résiste ! 🚀**
