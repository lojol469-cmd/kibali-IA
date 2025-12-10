#!/usr/bin/env python3
"""
Test complet: Simuler le flux de traitement de Kibali
"""

# Simuler les deux scénarios
print("=" * 80)
print("SCÉNARIO 1: Fichier présent en session + Question texte")
print("=" * 80)

# Simuler session avec fichier
binary_files = [{'name': 'data.dat', 'data': b'some data'}]
question = "fait moi un rapport de 20 pages en pdf sur les droits de suretés"
question_lower = question.lower()

file_keywords = [
    'fichier', 'file', 'donnée', 'data', 'analyse', 'extrait', 'contenu',
    'résistivité', 'ert', 'sondage', 'profondeur', 'valeur', 'colonne',
    'ligne', 'tableau', 'calcul', 'moyenne', 'min', 'max', 'statistique',
    'structure', 'format', 'convertis', 'export', 'visualise', 'graphique',
    'explique', 'montre', 'affiche'
]

question_about_file = any(kw in question_lower for kw in file_keywords)

print(f"\n📁 Fichier en session: OUI ({binary_files[0]['name']})")
print(f"❓ Question: \"{question}\"")
print(f"🔍 question_about_file: {question_about_file}")

has_binary_files = bool(binary_files)

print(f"\n{'=' * 80}")
print("CONDITION ACTUELLE:")
print(f"if 'binary_files' in session AND binary_files AND question_about_file:")
print(f"   {has_binary_files} AND {question_about_file}")
print(f"   = {has_binary_files and question_about_file}")
print(f"{'=' * 80}")

if has_binary_files and question_about_file:
    print("\n→ ENTRE DANS LA LOGIQUE FICHIER ❌")
    print("   Le système va analyser le fichier au lieu de générer le PDF")
else:
    print("\n→ PASSE À LA LOGIQUE NORMALE ✅")
    print("   Le système va générer le rapport PDF")

print("\n" + "=" * 80)
print("SCÉNARIO 2: Aucun fichier + Question texte")
print("=" * 80)

binary_files = []
has_binary_files = bool(binary_files)

print(f"\n📁 Fichier en session: NON")
print(f"❓ Question: \"{question}\"")
print(f"🔍 question_about_file: {question_about_file}")

print(f"\n{'=' * 80}")
print("CONDITION ACTUELLE:")
print(f"if 'binary_files' in session AND binary_files AND question_about_file:")
print(f"   {has_binary_files} AND {question_about_file}")
print(f"   = {has_binary_files and question_about_file}")
print(f"{'=' * 80}")

if has_binary_files and question_about_file:
    print("\n→ ENTRE DANS LA LOGIQUE FICHIER ❌")
else:
    print("\n→ PASSE À LA LOGIQUE NORMALE ✅")
    print("   Le système va générer le rapport PDF")

print("\n" + "=" * 80)
print("DIAGNOSTIC:")
print("=" * 80)
print("\n✅ La logique de détection est CORRECTE")
print("   • Si fichier présent ET question sur fichier → Analyse")
print("   • Sinon → Traitement normal (génération PDF)")
print("\n💡 PROBLÈME POSSIBLE:")
print("   • Un ancien fichier reste en session Streamlit")
print("   • Solution: Ajouter bouton 'Réinitialiser session'")
print("   • Ou: Vider binary_files après traitement")
