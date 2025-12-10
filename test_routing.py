#!/usr/bin/env python3
"""
Diagnostic complet du flux de traitement des questions
"""

def test_question_routing(prompt):
    """Simule le routing d'une question dans l'application"""
    question_lower = prompt.lower()
    
    # Détections
    needs_calculation = any(kw in question_lower for kw in ['calcul', 'moyenne', 'somme', 'total', 'combien', 'valeur', 'profondeur', 'exact'])
    needs_web_research = any(kw in question_lower for kw in ['domaine', 'activité', 'industrie', 'application', 'utilisation', 'pourquoi', 'comment'])
    needs_data_extraction = any(kw in question_lower for kw in ['données', 'valeurs', 'liste', 'extrait', 'montre', 'affiche', 'trouve'])
    needs_conversion = any(kw in question_lower for kw in ['convertis', 'convert', 'numpy', 'tableau', 'dataframe', 'pandas', 'csv', 'json', 'extraction'])
    needs_ert_analysis = any(kw in question_lower for kw in ['ert', 'résistivité', 'resistivité', 'géophysique', 'forage', 'nappe', 'aquifère', 'eau souterraine'])
    needs_pdf_generation = any(kw in question_lower for kw in ['rédige', 'génère', 'crée', 'écris', 'fait']) and any(kw in question_lower for kw in ['pdf', 'rapport', 'document', 'thèse', 'livre', 'mémoire'])
    
    question_about_file = any(kw in question_lower for kw in [
        'fichier', 'file', 'donnée', 'data', 'analyse', 'extrait', 'contenu',
        'résistivité', 'ert', 'sondage', 'profondeur', 'valeur', 'colonne',
        'ligne', 'tableau', 'calcul', 'moyenne', 'min', 'max', 'statistique',
        'structure', 'format', 'convertis', 'export', 'visualise', 'graphique',
        'explique', 'montre', 'affiche'
    ])
    
    # Simuler presence de fichier
    has_binary_file = False  # Pas de fichier uploadé
    
    print(f"\n{'='*80}")
    print(f"QUESTION: \"{prompt}\"")
    print(f"{'='*80}")
    print(f"\n🔍 DÉTECTIONS:")
    print(f"   needs_calculation: {needs_calculation}")
    print(f"   needs_web_research: {needs_web_research}")
    print(f"   needs_data_extraction: {needs_data_extraction}")
    print(f"   needs_conversion: {needs_conversion}")
    print(f"   needs_ert_analysis: {needs_ert_analysis}")
    print(f"   needs_pdf_generation: {needs_pdf_generation}")
    print(f"   question_about_file: {question_about_file}")
    print(f"   has_binary_file: {has_binary_file}")
    
    print(f"\n🔀 ROUTING:")
    
    if needs_pdf_generation:
        print(f"   ✅ → GÉNÉRATION PDF")
        print(f"      (Génère un PDF avec ReportLab)")
        return "PDF_GENERATION"
    elif has_binary_file and question_about_file:
        print(f"   ✅ → ANALYSE FICHIER")
        print(f"      (Analyse le fichier uploadé)")
        return "FILE_ANALYSIS"
    else:
        print(f"   ✅ → CHAT NORMAL")
        print(f"      (Traitement IA standard avec outils si nécessaire)")
        return "NORMAL_CHAT"

# Tests
questions_test = [
    "fait moi un pdf sur comment gagner de l argent avec sa technologie IA",
    "rédige un rapport de 30 pages sur la blockchain",
    "c est quoi la topographie",
    "explique moi l intelligence artificielle",
    "analyse ce fichier",  # Sans fichier uploadé
    "comment gagner de l'argent",
]

print("="*80)
print("DIAGNOSTIC DU FLUX DE TRAITEMENT")
print("="*80)

for q in questions_test:
    result = test_question_routing(q)
    
print("\n" + "="*80)
print("✅ DIAGNOSTIC TERMINÉ")
print("="*80)
print("\n💡 SI LE CHAT NE RÉPOND PAS:")
print("   1. Vérifier que st.session_state.tool_manager est initialisé")
print("   2. Vérifier qu'il n'y a pas d'exception silencieuse")
print("   3. Regarder les logs Streamlit dans le terminal")
print("   4. Vérifier que le modèle Ollama répond (test avec curl)")
