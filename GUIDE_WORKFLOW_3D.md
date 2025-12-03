# 📸 Guide Complet: Workflow Classification IA → Reconstruction 3D

## 🎯 Vue d'ensemble

Pipeline professionnel pour transformer vos photos drone en modèle 3D texturé:
1. **Classification IA** avec CLIP (36 critères multi-angles)
2. **Raffinement interactif** via chat intelligent
3. **Reconstruction 3D** avec COLMAP (photogrammétrie professionnelle)

---

## 🚀 Étape 1: Classification Intelligente

### Upload & Analyse
```
Interface Streamlit → Section "📸 Photos à analyser"
↓
Uploader 50-200 photos drone (format JPG/PNG)
↓
L'IA analyse automatiquement chaque image selon 36 critères:
  • 15 angles de vue (frontal, latéral, oblique, zénith...)
  • 11 qualités (sharp, detailed, well-exposed...)
  • 10 distances (close-up, medium, far, aerial...)
```

### Résultat
- Images triées automatiquement par pertinence
- Scores de confiance pour chaque catégorie
- Galerie interactive avec drag & drop

---

## 💬 Étape 2: Raffinement via Chat

### Commandes disponibles

| Commande | Action | Exemple |
|----------|--------|---------|
| `regroup similaire` | Grouper images similaires | "Regroupe les images ensemble" |
| `sombre début` | Trier du + sombre au + clair | "Mets les sombres au début" |
| `inverse` | Inverser l'ordre actuel | "Inverse l'ordre" |
| `floue` | Retirer images floues | "Enlève les photos floues" |

### Fonctionnement
```
Chat Kibali (en bas de la galerie)
↓
Tapez votre demande en langage naturel
↓
L'IA détecte le pattern et applique immédiatement
↓
La galerie se met à jour en temps réel
```

**Astuce:** Pas besoin de formulation parfaite, le système comprend les variantes!

---

## 🎨 Étape 3: Reconstruction 3D avec COLMAP

### Configuration

1. **Qualité** (select slider):
   - `low`: rapide, mesh simple (5-10 min)
   - `medium`: bon compromis (15-20 min) 
   - `high`: qualité pro ⭐ recommandée (25-40 min)
   - `extreme`: maximum détails (1-2h)

2. **GPU** (checkbox):
   - ✅ Activer pour MVS (Multi-View Stereo)
   - ⚠️ SIFT sur CPU en WSL (limitation OpenGL)

### Pipeline COLMAP (8 étapes)

```
┌─────────────────────────────────────────────────┐
│ 1. Préparation images (copie workspace)        │
│    📸 102 images → /tmp/colmap_workspace        │
│    ⏱️  ~1 minute                                 │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 2. Feature Extraction (SIFT)                   │
│    🔍 Détection keypoints sur chaque image      │
│    ⚙️  CPU mode (WSL)                           │
│    ⏱️  ~3-5 minutes (102 images)                │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 3. Feature Matching                            │
│    🔗 Correspondances entre images              │
│    ⚙️  CPU mode                                 │
│    ⏱️  ~2-4 minutes                             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 4. Sparse Reconstruction (SfM)                 │
│    📐 Structure-from-Motion                     │
│    📷 Estimation poses caméras + nuage sparse   │
│    ⏱️  ~3-6 minutes                             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 5. Image Undistortion                          │
│    🎯 Correction distorsions caméra             │
│    ⏱️  ~1-2 minutes                             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 6. Dense Reconstruction (MVS)                  │
│    ☁️  Multi-View Stereo depth maps            │
│    ⚡ GPU accelerated (RTX 5090)                │
│    ⏱️  ~10-20 minutes (quality=high)            │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 7. Stereo Fusion                               │
│    🔀 Fusion depth maps → dense point cloud     │
│    ⏱️  ~2-3 minutes                             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 8. Poisson Meshing                             │
│    🎨 Génération mesh 3D + textures             │
│    📦 Export PLY, OBJ                           │
│    ⏱️  ~1-2 minutes                             │
└─────────────────────────────────────────────────┘
                    ↓
              ✅ TERMINÉ !
```

**Temps total estimé:** 25-45 minutes (quality=high)

### Résultats

Vous obtenez :
- ☁️  **Dense Point Cloud** : `fused.ply` (millions de points)
- 🎨 **Mesh 3D** : `meshed-poisson.ply` (surfaces + textures)
- 📷 **Cameras** : Positions/orientations estimées
- 📊 **Stats** : Nombre de points, images utilisées

### Fichiers générés

```
/tmp/colmap_kibali_XXXXX/
├── images/              # 102 photos copiées
├── database.db          # Features SIFT + matches
├── sparse/
│   └── 0/               # Reconstruction sparse (cameras + points)
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── dense/
│   ├── images/          # Images undistorted
│   ├── stereo/          # Depth maps
│   │   ├── depth_maps/
│   │   └── normal_maps/
│   ├── fused.ply        # ⭐ Point cloud dense
│   └── meshed-poisson.ply  # ⭐ Mesh final
└── colmap_output.json   # Statistiques

Taille totale: ~500 MB - 2 GB
```

---

## 📊 Visualisation & Export

### Visualiser le Mesh

**Option 1: MeshLab** (recommandé)
```bash
sudo apt install meshlab
meshlab /tmp/colmap_kibali_XXX/dense/meshed-poisson.ply
```

**Option 2: Blender**
```bash
blender
# File → Import → PLY → Sélectionner meshed-poisson.ply
```

**Option 3: CloudCompare** (pour point cloud)
```bash
sudo apt install cloudcompare
cloudcompare.CloudCompare /tmp/colmap_kibali_XXX/dense/fused.ply
```

### Exporter vers d'autres formats

**PLY → OBJ (avec textures)**
```python
import open3d as o3d
mesh = o3d.io.read_triangle_mesh("meshed-poisson.ply")
o3d.io.write_triangle_mesh("model.obj", mesh, write_vertex_colors=True)
```

**Optimiser pour le Web** (glTF/GLB)
```bash
# Via Blender
blender --background --python - << EOF
import bpy
bpy.ops.import_mesh.ply(filepath="meshed-poisson.ply")
bpy.ops.export_scene.gltf(filepath="model.glb", export_format='GLB')
EOF
```

---

## ⚙️ Paramètres Avancés

### Modifier la qualité COLMAP

Dans `outils/colmap_photogrammetry.py`:

```python
quality_params = {
    'low': {
        '--PatchMatchStereo.window_radius': '3',
        '--PatchMatchStereo.num_samples': '5',
        '--PatchMatchStereo.num_iterations': '3'
    },
    'high': {
        '--PatchMatchStereo.window_radius': '7',
        '--PatchMatchStereo.num_samples': '15', 
        '--PatchMatchStereo.num_iterations': '5'
    },
    'extreme': {
        '--PatchMatchStereo.window_radius': '11',
        '--PatchMatchStereo.num_samples': '25',
        '--PatchMatchStereo.num_iterations': '8'
    }
}
```

### Forcer GPU pour SIFT (Linux natif)

Retirer la détection WSL dans `feature_extraction()`:

```python
# Commenter ces lignes:
# if 'microsoft' in platform.uname().release.lower():
#     logger.warning("⚠️  WSL: SIFT sur CPU")
#     use_gpu = False
```

---

## 🐛 Troubleshooting

### Erreur: "OpenGL context creation failed"
**Cause:** WSL essaie d'utiliser GPU pour SIFT  
**Solution:** Mode CPU automatique (déjà implémenté)

### Erreur: "SQLite constraint failed"
**Cause:** Base de données COLMAP corrompue  
**Solution:**
```bash
rm -rf /tmp/reconstruction_drone_3d
# Relancer le pipeline
```

### Processus trop lent
**Optimisations:**
1. Réduire nombre d'images (50-80 au lieu de 102)
2. Utiliser `quality='medium'` au lieu de `high`
3. Vérifier: `htop` → Python doit utiliser ~100% CPU pendant SIFT

### Mesh final incomplet
**Causes possibles:**
- Images trop différentes (angles trop variés)
- Chevauchement insuffisant (<60%)
- Éclairage très variable

**Solutions:**
1. Retrier avec chat: "regroupe les similaires"
2. Retirer vues extrêmes
3. Utiliser `quality='extreme'` pour plus de détails

---

## 📜 Licences

| Composant | Licence | Usage commercial |
|-----------|---------|------------------|
| COLMAP | BSD 3-Clause | ✅ Autorisé |
| CLIP (OpenAI) | MIT | ✅ Autorisé |
| EasyOCR | Apache 2.0 | ✅ Autorisé |
| Sentence Transformers | Apache 2.0 | ✅ Autorisé |

**Votre workflow complet est 100% commercial-ready!** 🎉

---

## 🎓 Ressources

- [COLMAP Documentation](https://colmap.github.io/)
- [COLMAP Tutorial](https://colmap.github.io/tutorial.html)
- [Photogrammetry Best Practices](https://www.capturingreality.com/photogrammetry-guidelines)
- [MeshLab Tutorials](https://www.meshlab.net/#tutorials)

---

## 💡 Astuces Pro

1. **Photos optimales:** 
   - 60-70% de chevauchement entre images adjacentes
   - Angles variés mais progressifs
   - Éclairage constant

2. **Classification efficace:**
   - Commencer avec 100-150 photos
   - Utiliser chat pour affiner
   - Viser 50-80 images finales bien espacées

3. **Performance:**
   - SIFT CPU: ~2-3 secondes par image
   - MVS GPU: ~300-500 depth maps/min (RTX 5090)
   - Total 100 images: 25-40 minutes

4. **Qualité maximale:**
   - `quality='extreme'` + GPU
   - Images 12+ megapixels
   - Bracketing HDR pour éclairage uniforme

---

**Auteur:** Kibali AI System  
**Version:** 1.0  
**Date:** Décembre 2025  
**License workflow:** MIT
