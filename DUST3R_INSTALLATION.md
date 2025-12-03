# Installation DUSt3R - État

## ✅ Environnement actuel

- **Python**: 3.13.9
- **PyTorch**: 2.10.0.dev20251122+cu128 (nightly avec CUDA 13.0)
- **CUDA**: Compatible RTX 5090

## ✅ Dépendances installées

### DUSt3R (depuis requirements.txt)
- ✅ torch, torchvision
- ✅ roma, gradio, matplotlib, tqdm
- ✅ opencv-python, scipy, einops
- ✅ trimesh, tensorboard, pyglet<2
- ✅ huggingface-hub[torch]>=0.22

### Application Streamlit
- ✅ streamlit, plotly, pillow
- ✅ scikit-learn, transformers
- ✅ pandas, psutil, pynvml (nvidia-ml-py3)
- ✅ faiss-cpu

### ⚠️ Open3D
- **Statut**: Non installé (incompatible Python 3.13)
- **Impact**: Fonctionnalités de visualisation 3D avancées désactivées
- **Solution**: L'application fonctionne avec Plotly pour la visualisation

## 📁 Fichiers créés

1. **`/home/belikan/kibali-IA/app_dust3r.py`**
   - Application Streamlit complète avec DUSt3R
   - Reconstruction 3D, textures PBR, maillage
   - Support GPU avec monitoring

2. **`/home/belikan/dust3r/`** (lien symbolique)
   - Repository DUSt3R cloné depuis GitHub
   - Modèle: `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`

3. **`/home/belikan/kibali-IA/app.py.backup_before_dust3r`**
   - Backup de l'ancienne application avec COLMAP

## 🚀 Lancement de l'application

```bash
cd /home/belikan/kibali-IA
streamlit run app_dust3r.py
```

## 📊 Informations importantes

### Modèle DUSt3R
- **Nom**: `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`
- **Taille**: ~1.5 GB (sera téléchargé au premier lancement)
- **Source**: HuggingFace Hub
- **Licence**: Vérifier sur https://github.com/naver/dust3r

### Compatibilité Open3D
Pour installer Open3D, il faudrait:
1. Créer un environnement Python 3.11 dédié
2. Ou attendre une version Open3D compatible Python 3.13

### Fonctionnalités actuelles sans Open3D
- ✅ Reconstruction 3D DUSt3R
- ✅ Visualisation Plotly (nuages de points 3D)
- ✅ Textures PBR intelligentes
- ✅ Export PLY
- ❌ Visualisation Open3D (fenêtre externe)
- ❌ Maillage Poisson/Ball Pivoting
- ❌ Rendu Blender automatique

## 🔧 Prochaines étapes

1. **Tester l'application**: `streamlit run app_dust3r.py`
2. **Télécharger 2+ images** pour reconstruction 3D
3. **Vérifier le GPU** via monitoring sidebar
4. **Optionnel**: Créer env Python 3.11 pour Open3D complet

## 📖 Ressources

- DUSt3R GitHub: https://github.com/naver/dust3r
- OpenSfM (alternative COLMAP): https://github.com/mapillary/OpenSfM
- Documentation DUSt3R: https://opensfm.org/docs/
