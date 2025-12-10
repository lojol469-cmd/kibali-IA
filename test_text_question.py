#!/usr/bin/env python3
"""
Test pour vérifier si Kibali traite les questions texte sans attendre de fichier
"""

# Simuler la logique de détection
question = "fait moi un rapport de 20 pages en pdf sur les droits de suretés"
question_lower = question.lower()

# Mots-clés pour détecter question sur fichier
file_keywords = [
    'fichier', 'file', 'donnée', 'data', 'analyse', 'extrait', 'contenu',
    'résistivité', 'ert', 'sondage', 'profondeur', 'valeur', 'colonne',
    'ligne', 'tableau', 'calcul', 'moyenne', 'min', 'max', 'statistique',
    'structure', 'format', 'convertis', 'export', 'visualise', 'graphique',
    'explique', 'montre', 'affiche'
]

question_about_file = any(kw in question_lower for kw in file_keywords)

print("=" * 80)
print("TEST: Détection de question sur fichier")
print("=" * 80)
print(f"\nQuestion: \"{question}\"")
print(f"\nMots-clés détectés:")
for kw in file_keywords:
    if kw in question_lower:
        print(f"  ✓ '{kw}' trouvé dans la question")

print(f"\n{'=' * 80}")
print(f"Résultat: question_about_file = {question_about_file}")
print(f"{'=' * 80}")

if question_about_file:
    print("\n❌ PROBLÈME: La question est considérée comme concernant un fichier")
    print("   → Kibali va attendre un fichier uploadé au lieu de générer le PDF")
    print("\n💡 MOTS-CLÉS PROBLÉMATIQUES DÉTECTÉS:")
    for kw in file_keywords:
        if kw in question_lower:
            print(f"   • '{kw}'")
else:
    print("\n✅ CORRECT: La question est traitée comme une demande normale")
    print("   → Kibali va générer le rapport PDF directement")

print("\n" + "=" * 80)
print("RECOMMANDATION:")
print("=" * 80)
if question_about_file:
    print("⚠️  Les mots-clés de détection sont TROP LARGES")
    print("    Il faut être plus spécifique pour différencier:")
    print("    • Question sur fichier UPLOADÉ (analyse)")
    print("    • Question demandant CRÉATION de document (génération)")
    print("\n💡 Solution: Exclure 'rapport', 'pdf', 'génère', 'crée', 'rédige'")
    print("    des mots-clés de détection de fichier")
