#!/usr/bin/env python3
"""
Test du pipeline COLMAP avec échantillon d'images
"""

import sys
import glob
from pathlib import Path
from outils.colmap_photogrammetry import ColmapPhotogrammetry

def progress_callback(current, total, message):
    """Callback pour afficher la progression"""
    percentage = (current / total * 100) if total > 0 else 0
    print(f"[{percentage:5.1f}%] {current}/{total} - {message}")

def main():
    print("=" * 70)
    print("🎯 TEST COLMAP - Reconstruction 3D")
    print("=" * 70)
    
    # Chercher des images de test
    test_image_patterns = [
        "/tmp/*.jpg",
        "/tmp/*.JPG",
        "/tmp/*.png",
        "/home/belikan/kibali-IA/*.jpg",
        "/home/belikan/*.jpg"
    ]
    
    image_paths = []
    for pattern in test_image_patterns:
        image_paths.extend(glob.glob(pattern))
    
    if not image_paths:
        print("\n❌ Aucune image trouvée pour le test")
        print("💡 Pour tester, placez quelques images dans /tmp/ ou spécifiez un dossier")
        print("\nUtilisation:")
        print("  python3 test_colmap_reconstruction.py [dossier_images]")
        return 1
    
    # Limiter à 10 images pour le test
    image_paths = image_paths[:10]
    
    print(f"\n📸 Images trouvées: {len(image_paths)}")
    for i, path in enumerate(image_paths[:5], 1):
        print(f"   {i}. {Path(path).name}")
    if len(image_paths) > 5:
        print(f"   ... et {len(image_paths) - 5} autres")
    
    # Créer workspace
    workspace = Path("/tmp/colmap_test_workspace")
    print(f"\n📁 Workspace: {workspace}")
    
    # Initialiser COLMAP
    colmap = ColmapPhotogrammetry(str(workspace))
    
    print("\n" + "=" * 70)
    print("🚀 DÉMARRAGE DU PIPELINE COLMAP")
    print("=" * 70)
    
    # Demander confirmation
    response = input("\n▶️  Lancer la reconstruction 3D ? (o/N): ")
    if response.lower() not in ['o', 'oui', 'y', 'yes']:
        print("❌ Test annulé")
        return 0
    
    try:
        # Lancer le pipeline complet
        results = colmap.run_full_pipeline(
            image_paths=image_paths,
            use_gpu=True,
            quality='medium',  # medium pour test rapide
            progress_callback=progress_callback
        )
        
        print("\n" + "=" * 70)
        print("📊 RÉSULTATS")
        print("=" * 70)
        
        if results['success']:
            print("✅ Reconstruction 3D réussie !\n")
            
            if results.get('images_prepared'):
                print(f"📸 Images traitées: {results['images_prepared']}")
            
            if results.get('sparse_model'):
                print(f"🎯 Modèle sparse: {results['sparse_model']}")
            
            if results.get('dense_pointcloud'):
                print(f"🌟 Nuage dense: {results['dense_pointcloud']}")
            
            if results.get('mesh'):
                print(f"🎨 Mesh 3D: {results['mesh']}")
                print(f"\n💡 Visualiser avec: meshlab {results['mesh']}")
            
            if results.get('errors'):
                print(f"\n⚠️  Avertissements: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"   - {error}")
        else:
            print("❌ Reconstruction échouée\n")
            if results.get('errors'):
                print("Erreurs:")
                for error in results['errors']:
                    print(f"   - {error}")
        
        # Stats
        stats = colmap.get_reconstruction_stats()
        print(f"\n📊 Statistiques:")
        print(f"   • Images: {stats['num_images']}")
        print(f"   • Workspace: {stats['workspace']}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 70)
    print("✅ TEST TERMINÉ")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    # Vérifier si dossier spécifié
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        print(f"📁 Dossier spécifié: {folder}")
    
    sys.exit(main())
