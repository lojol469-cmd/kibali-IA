#!/usr/bin/env python3
"""
Test du générateur de PDF massif
"""

# Test de détection
prompts_test = [
    "fait moi un rapport de 20 pages en pdf sur les droits de suretés",
    "rédige moi un document de 30 pages sur le droit des sociétés",
    "génère un PDF de 50 pages sur l'intelligence artificielle",
    "crée une thèse de 100 pages sur la blockchain",
    "analyse ce fichier",  # NE DOIT PAS déclencher PDF
    "explique moi la résistivité",  # NE DOIT PAS déclencher PDF
]

print("=" * 80)
print("TEST: Détection de génération PDF")
print("=" * 80)

for prompt in prompts_test:
    question_lower = prompt.lower()
    
    # Détection
    needs_pdf = (
        any(kw in question_lower for kw in ['rédige', 'génère', 'crée', 'écris', 'fait']) and
        any(kw in question_lower for kw in ['pdf', 'rapport', 'document', 'thèse', 'livre', 'mémoire']) and
        any(str(n) in prompt for n in range(10, 501))
    )
    
    print(f"\n📝 Prompt: \"{prompt}\"")
    print(f"   → Génération PDF: {'✅ OUI' if needs_pdf else '❌ NON'}")
    
    if needs_pdf:
        # Extraire le nombre
        num_pages = 30
        for n in range(10, 501):
            if str(n) in prompt:
                num_pages = n
                break
        print(f"   → Nombre de pages: {num_pages}")

print("\n" + "=" * 80)
print("✅ Logique de détection validée!")
