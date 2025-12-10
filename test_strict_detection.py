#!/usr/bin/env python3
"""
Test de la nouvelle détection de questions sur fichier
"""

def test_file_question_detection(prompt, has_binary_file=False):
    """Test si la question concerne le fichier"""
    question_lower = prompt.lower()
    
    # Nouvelle détection STRICTE
    question_about_file = (
        has_binary_file and
        any(kw in question_lower for kw in [
            'ce fichier', 'le fichier', 'du fichier', 'dans le fichier',
            'fichier uploadé', 'fichier téléchargé', 'données du fichier',
            'contenu du fichier', 'analyse du fichier', 'ce .dat', 'ce .csv',
            'résistivité du', 'profondeur dans', 'valeurs du fichier',
            'colonnes du', 'lignes du fichier', 'structure du fichier'
        ])
    )
    
    return question_about_file

# Tests
tests = [
    # Questions qui NE DOIVENT PAS déclencher l'analyse fichier
    ("c est quoi la topographie", False, False),
    ("explique moi l intelligence artificielle", False, False),
    ("fait moi un pdf sur l IA", False, False),
    ("comment gagner de l argent", False, False),
    ("calcule la moyenne", False, False),
    
    # Questions qui DOIVENT déclencher (avec fichier présent)
    ("analyse ce fichier", True, True),
    ("que contient le fichier", True, True),
    ("montre moi les données du fichier", True, True),
    ("quelle est la structure du fichier", True, True),
    ("extrais les valeurs du fichier", True, True),
    
    # Questions avec fichier présent mais ne concernant PAS le fichier
    ("c est quoi la topographie", True, False),
    ("explique moi l IA", True, False),
]

print("="*80)
print("TEST: Détection STRICTE de questions sur fichier")
print("="*80)

for prompt, has_file, expected in tests:
    result = test_file_question_detection(prompt, has_file)
    status = "✅" if result == expected else "❌"
    
    print(f"\n{status} Prompt: \"{prompt}\"")
    print(f"   Fichier présent: {has_file}")
    print(f"   Attendu: {expected}, Obtenu: {result}")
    
    if result != expected:
        print(f"   ⚠️  PROBLÈME DÉTECTÉ!")

print("\n" + "="*80)
print("✅ Tests terminés")
