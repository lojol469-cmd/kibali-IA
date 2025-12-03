#!/usr/bin/env python3.13
"""
Demo script pour l'outil d'optimisation de photogrammétrie
Crée un dataset de test et lance l'optimisation avec visualisation
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

def create_demo_dataset(output_dir: str, num_images: int = 50):
    """
    Crée un dataset de démonstration avec des images synthétiques
    simulant une séquence de photogrammétrie
    """
    print(f"📸 Création de {num_images} images de démonstration...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Créer des images avec variation progressive (simulation d'une trajectoire)
    for i in range(num_images):
        # Position sur une trajectoire circulaire
        angle = (i / num_images) * 2 * np.pi
        x_offset = int(256 + 200 * np.cos(angle))
        y_offset = int(256 + 200 * np.sin(angle))
        
        # Créer une image avec des patterns variables
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Cercle central (objet à reconstruire)
        y, x = np.ogrid[:512, :512]
        mask = (x - x_offset)**2 + (y - y_offset)**2 <= 100**2
        
        # Couleurs variables selon l'angle
        r = int(127 + 127 * np.sin(angle))
        g = int(127 + 127 * np.cos(angle))
        b = int(127 + 127 * np.sin(angle + np.pi/2))
        
        img_array[mask] = [r, g, b]
        
        # Ajouter du bruit pour différencier les images
        noise = np.random.randint(0, 50, (512, 512, 3), dtype=np.uint8)
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Sauvegarder
        img = Image.fromarray(img_array)
        img.save(output_path / f"photo_{i:04d}.jpg")
        
        if (i + 1) % 10 == 0:
            print(f"   Créé: {i + 1}/{num_images} images")
    
    print(f"✅ Dataset créé dans: {output_path}\n")
    return str(output_path)


def run_optimization(input_folder: str):
    """
    Lance l'optimisation du dataset avec l'outil
    """
    print("🚀 Lancement de l'optimisation...\n")
    
    from outils.photogrammetry_optimizer_tool import PhotogrammetryOptimizerTool
    
    tool = PhotogrammetryOptimizerTool()
    
    context = {
        'input_folder': input_folder,
        'target_count': None,  # Automatique
        'coverage_threshold': 0.90,  # 90% pour la démo
        'similarity_threshold': 0.85
    }
    
    result = tool.execute("", context=context)
    
    print("\n" + "=" * 80)
    print(result)
    print("=" * 80)
    
    # Afficher les fichiers générés
    output_folder = Path(input_folder + "_optimized")
    if output_folder.exists():
        print("\n📁 Fichiers générés:")
        for file in sorted(output_folder.iterdir()):
            size = file.stat().st_size / 1024  # KB
            print(f"   - {file.name} ({size:.1f} KB)")


def main():
    """
    Script principal de démonstration
    """
    print("""
╔════════════════════════════════════════════════════════════════╗
║  📷 DÉMO - Optimisation de Photogrammétrie pour Dust3R        ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    # Configuration
    demo_dir = "/tmp/photogrammetry_demo"
    num_images = 50
    
    print(f"⚙️  Configuration:")
    print(f"   - Dossier: {demo_dir}")
    print(f"   - Images: {num_images}")
    print(f"   - Réduction cible: Automatique (~5%)")
    print(f"   - Couverture min: 90%\n")
    
    # Créer le dataset
    input_folder = create_demo_dataset(demo_dir, num_images)
    
    # Lancer l'optimisation
    run_optimization(input_folder)
    
    print(f"""
✅ Démonstration terminée!

📊 Résultats disponibles dans:
   {demo_dir}_optimized/

📝 Fichiers importants:
   - 0001_*.jpg, 0002_*.jpg, ... : Images ordonnées séquentiellement
   - image_order.txt : Ordre optimal pour Dust3R
   - image_positions.ply : Visualisation 3D
   - sequence_visualization.ply : Parcours complet
   - launch_viewer.py : Script visionneuse Open3D
   - optimization_report.txt : Rapport détaillé

🎨 Pour visualiser en 3D:
   cd {demo_dir}_optimized
   python launch_viewer.py

🚀 Pour utiliser avec Dust3R:
   cd Dust3R
   python demo.py --image_dir {demo_dir}_optimized

💡 Pour tester avec vos propres photos:
   python demo_photogrammetry.py /chemin/vers/vos/photos
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Utiliser le dossier fourni
        input_folder = sys.argv[1]
        if not Path(input_folder).exists():
            print(f"❌ Dossier introuvable: {input_folder}")
            sys.exit(1)
        
        print(f"📂 Utilisation du dossier: {input_folder}\n")
        run_optimization(input_folder)
    else:
        # Mode démo avec dataset synthétique
        main()
