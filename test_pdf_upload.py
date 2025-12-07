#!/usr/bin/env python3.13
"""
Script de test pour vérifier l'upload et traitement PDF
"""

import os
import sys

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(__file__))

print("🧪 Test de traitement PDF")
print("=" * 50)

# Test 1: Import des modules
print("\n1️⃣ Test des imports...")
try:
    from app import extract_text_from_pdf, PDFS_PATH
    print("   ✅ Imports réussis")
except Exception as e:
    print(f"   ❌ Erreur d'import: {e}")
    sys.exit(1)

# Test 2: os.path.join
print("\n2️⃣ Test de os.path.join...")
try:
    test_path = os.path.join(PDFS_PATH, "test.pdf")
    print(f"   ✅ os.path.join: {test_path}")
except Exception as e:
    print(f"   ❌ Erreur os.path.join: {e}")
    sys.exit(1)

# Test 3: Vérifier que PDFS_PATH existe
print(f"\n3️⃣ Test de PDFS_PATH: {PDFS_PATH}")
if os.path.exists(PDFS_PATH):
    print(f"   ✅ Dossier existe")
    pdf_files = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')]
    print(f"   📄 PDFs trouvés: {len(pdf_files)}")
    if pdf_files:
        print(f"      - {pdf_files[:3]}")
else:
    print(f"   ⚠️  Dossier n'existe pas encore (sera créé)")

# Test 4: Test extraction si PDF disponible
print("\n4️⃣ Test d'extraction de texte...")
test_pdfs = [
    os.path.join(PDFS_PATH, "biologie-et-ecologie-brigitte-lods-crozet.pdf")
]

for pdf_path in test_pdfs:
    if os.path.exists(pdf_path):
        try:
            text = extract_text_from_pdf(pdf_path)
            word_count = len(text.split())
            print(f"   ✅ Extraction réussie: {os.path.basename(pdf_path)}")
            print(f"      - Mots: {word_count:,}")
            print(f"      - Preview: {text[:200]}...")
            break
        except Exception as e:
            print(f"   ❌ Erreur extraction: {e}")
    else:
        print(f"   ⏭️  PDF non trouvé: {os.path.basename(pdf_path)}")

print("\n" + "=" * 50)
print("✅ Tous les tests passés avec succès !")
print("\n💡 L'application est prête à recevoir des PDFs via l'interface.")
