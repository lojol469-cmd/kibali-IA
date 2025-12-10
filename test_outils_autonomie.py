#!/usr/bin/env python3
"""
Test du système d'outils autonome de Kibali
Vérifie que les outils sont appelés intelligemment selon le contexte
"""

import sys
import os

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(__file__))

# Test des importations
print("🔧 Test du système d'outils autonome\n")
print("="*60)

try:
    from outils import ToolManager
    print("✅ ToolManager importé")
except Exception as e:
    print(f"❌ Erreur import ToolManager: {e}")
    sys.exit(1)

# Initialiser le gestionnaire
try:
    tool_manager = ToolManager()
    print(f"✅ {len(tool_manager.tools)} outils chargés:")
    for name in tool_manager.tools.keys():
        print(f"   - {name}")
except Exception as e:
    print(f"❌ Erreur initialisation: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("🧪 Tests de détection des outils\n")

# Questions de test
test_cases = [
    {
        'query': 'comment gagner de l\'argent avec une IA?',
        'expected_tools': ['Recherche web avancée'],
        'context': {'has_local_info': False, 'web_enabled': True}
    },
    {
        'query': 'calcule la somme de 15 et 27',
        'expected_tools': ['Calculatrice mathématique'],
        'context': {}
    },
    {
        'query': 'traduis "hello" en français',
        'expected_tools': ['Outil de traduction'],
        'context': {}
    },
    {
        'query': 'écris une fonction Python pour trier une liste',
        'expected_tools': ['Générateur de code'],
        'context': {}
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"\n📝 Test {i}: {test['query']}")
    print(f"   Outils attendus: {', '.join(test['expected_tools'])}")
    
    # Récupérer les outils pertinents
    relevant_tools = tool_manager.get_relevant_tools(
        test['query'],
        test.get('context', {}),
        max_tools=2
    )
    
    if relevant_tools:
        detected_names = [tool.name for tool in relevant_tools]
        print(f"   ✅ Outils détectés: {', '.join(detected_names)}")
        
        # Vérifier si au moins un outil attendu est détecté
        match = any(expected in detected_names for expected in test['expected_tools'])
        if match:
            print(f"   ✅ SUCCÈS: Outil correct détecté")
        else:
            print(f"   ⚠️  ATTENTION: Outils détectés différents des attentes")
    else:
        print(f"   ❌ Aucun outil détecté")

print("\n" + "="*60)
print("✅ Tests terminés\n")
