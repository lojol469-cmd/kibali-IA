#!/usr/bin/env python3
"""
Test automatisé pour identifier où le chat se bloque
"""
import sys
sys.path.insert(0, '/home/belikan/kibali-IA')

# Simuler les détections
def test_chat_flow():
    prompt = "c est quoi la topographie"
    question_lower = prompt.lower()
    
    # Simulations des variables session_state
    has_binary_files = False
    tool_manager_available = True
    
    print("="*80)
    print("TEST: Flux de traitement pour question normale")
    print("="*80)
    print(f"\n📝 Question: '{prompt}'")
    print(f"   has_binary_files: {has_binary_files}")
    print(f"   tool_manager: {tool_manager_available}")
    
    # Détection PDF
    needs_pdf = (
        any(kw in question_lower for kw in ['rédige', 'génère', 'crée', 'écris', 'fait']) and
        any(kw in question_lower for kw in ['pdf', 'rapport', 'document', 'thèse', 'livre', 'mémoire'])
    )
    print(f"\n🔍 Détection PDF: {needs_pdf}")
    
    if needs_pdf:
        print("   → Entre dans bloc GÉNÉRATION PDF")
        print("   → FIN (return)")
        return "PDF_GENERATION"
    
    # Détection fichier
    question_about_file = (
        has_binary_files and
        any(kw in question_lower for kw in [
            'ce fichier', 'le fichier', 'du fichier', 'dans le fichier',
            'fichier uploadé', 'fichier téléchargé', 'données du fichier',
            'contenu du fichier', 'analyse du fichier', 'ce .dat', 'ce .csv',
            'résistivité du', 'profondeur dans', 'valeurs du fichier',
            'colonnes du', 'lignes du fichier', 'structure du fichier'
        ])
    )
    print(f"\n🔍 Détection fichier: {question_about_file}")
    
    if question_about_file:
        print("   → Entre dans bloc ANALYSE FICHIER")
        print("   → Traitement fichier...")
        print("   → FIN (continue vers chat)")
        # Ne fait PAS return, continue
    
    # Chat normal
    print(f"\n💬 Chat normal:")
    print(f"   → Ajout message historique")
    print(f"   → Pré-calculs (if binary_files): SKIP (pas de fichier)")
    print(f"   → Extraction (if binary_files): SKIP (pas de fichier)")
    
    # Ici est le PROBLÈME POTENTIEL
    print(f"\n⚠️  POINT CRITIQUE:")
    print(f"   if tool_manager and TOOLS_AVAILABLE:")
    
    if tool_manager_available:
        print(f"      ✅ Tool manager disponible")
        print(f"      → Entre dans bloc outils")
        print(f"      → Vérifie orchestration_enabled")
        print(f"      → Si pas orchestration: génère réponse normale")
        print(f"      → Affiche avec st.chat_message()")
        return "CHAT_WITH_TOOLS"
    else:
        print(f"      ❌ Tool manager NON disponible")
        print(f"      → Entre dans bloc else")
        print(f"      → Crée agent si besoin")
        print(f"      → Mode RAG ou web")
        return "CHAT_FALLBACK"

# Test
result = test_chat_flow()
print(f"\n{'='*80}")
print(f"✅ Résultat: {result}")
print(f"{'='*80}")

# Diagnostic
print(f"\n💡 DIAGNOSTIC:")
if result == "CHAT_WITH_TOOLS":
    print(f"   Le chat DEVRAIT fonctionner si:")
    print(f"   1. create_client() fonctionne")
    print(f"   2. Le modèle répond (Qwen local)")
    print(f"   3. Pas d'exception silencieuse")
    print(f"\n⚠️  PROBLÈME PROBABLE:")
    print(f"   - create_client() échoue silencieusement")
    print(f"   - OU le modèle ne répond pas")
    print(f"   - OU exception dans le try/except qui ne log rien")
elif result == "CHAT_FALLBACK":
    print(f"   Le chat utilise le fallback (agent)")
    print(f"   DEVRAIT fonctionner avec generate_answer_enhanced_stream()")
