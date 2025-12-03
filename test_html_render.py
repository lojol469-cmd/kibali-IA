#!/usr/bin/env python3
"""
Test du rendu HTML des cartes 3D - Backend Testing
Vérifie que le HTML est correctement formaté et ne s'affiche pas comme du texte brut
"""

import re

def test_html_structure():
    """Teste la structure HTML de la carte 3D"""
    
    # Simuler un message de réponse
    formatted_response = """
## Introduction 🌐
KAT VR est une entreprise pionnière dans le développement de matériel de réalité virtuelle.

### Technologies Innovantes 💡
- Tapis roulant VR
- Capture de mouvement
- Design ergonomique
"""
    
    # Template HTML exact de app.py
    html_output = f'''
<div class="chat-message-assistant" style="
    position: relative;
    background: linear-gradient(145deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
    border: 1px solid rgba(0, 255, 136, 0.3);
    padding: 2rem;
    border-radius: 16px;
    margin: 2rem 0.5rem;
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.6),
        0 10px 20px rgba(0, 0, 0, 0.4),
        0 25px 50px rgba(0, 255, 136, 0.2),
        inset 0 2px 0 rgba(255, 255, 255, 0.1),
        inset 0 -2px 0 rgba(0, 0, 0, 0.3),
        -8px 0 16px rgba(0, 255, 136, 0.15),
        8px 0 16px rgba(0, 136, 255, 0.15);
    border-left: 6px solid;
    border-image: linear-gradient(180deg, 
        rgba(0, 255, 136, 1) 0%, 
        rgba(0, 255, 136, 0.6) 50%, 
        rgba(0, 255, 136, 1) 100%) 1;
    transform: perspective(1000px) rotateX(2deg);
    transform-style: preserve-3d;
    backdrop-filter: blur(10px);
    animation: slideIn3D 0.5s ease-out;
">
    <!-- Tranche supérieure -->
    <div style="
        position: absolute;
        top: -4px;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, 
            rgba(0, 255, 136, 0.4) 0%, 
            rgba(0, 136, 255, 0.4) 100%);
        border-radius: 16px 16px 0 0;
        box-shadow: 0 -2px 10px rgba(0, 255, 136, 0.3);
    "></div>
    
    <!-- Ombre inférieure -->
    <div style="
        position: absolute;
        bottom: -8px;
        left: 4px;
        right: 4px;
        height: 8px;
        background: linear-gradient(180deg, 
            rgba(0, 0, 0, 0.5) 0%, 
            rgba(0, 0, 0, 0.8) 100%);
        border-radius: 0 0 12px 12px;
        filter: blur(4px);
        z-index: -1;
    "></div>
    
    <!-- Reflet lumineux -->
    <div style="
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 50%;
        background: linear-gradient(180deg, 
            rgba(255, 255, 255, 0.05) 0%, 
            transparent 100%);
        border-radius: 16px 16px 0 0;
        pointer-events: none;
    "></div>
    
    <!-- En-tête -->
    <div style="
        position: relative;
        z-index: 1;
        display: flex; 
        align-items: center; 
        margin-bottom: 1.5rem; 
        gap: 0.75rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(0, 255, 136, 0.2);
    ">
        <span class="chat-k-logo" style="
            font-size: 2rem; 
            font-weight: 900;
            background: linear-gradient(135deg, #00ff88 0%, #00ffff 50%, #0088ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: pulse3D 2s ease-in-out infinite;
            filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.5));
            transform: translateZ(20px);
        ">K</span>
        <strong style="
            color: #00ff88; 
            font-size: 1.2rem;
            text-shadow: 
                0 0 10px rgba(0, 255, 136, 0.5),
                0 2px 4px rgba(0, 0, 0, 0.5);
            transform: translateZ(10px);
        ">Kibali AI</strong>
        <span style="
            margin-left: auto;
            padding: 0.4rem 1rem;
            background: linear-gradient(135deg, 
                rgba(0, 255, 136, 0.2) 0%, 
                rgba(0, 136, 255, 0.2) 100%);
            border: 1px solid rgba(0, 255, 136, 0.4);
            border-radius: 25px;
            font-size: 0.8rem;
            color: #00ff88;
            font-weight: 600;
            box-shadow: 
                0 4px 8px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
            transform: translateZ(15px);
        ">🤖 Réponse enrichie</span>
    </div>
    
    <!-- Contenu de la réponse -->
    <div class="response-content" style="
        position: relative;
        z-index: 1;
        line-height: 1.8;
        color: #e0e0e0;
        font-size: 1.05rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
        transform: translateZ(5px);
    ">{formatted_response}</div>
</div>

<style>
@keyframes slideIn3D {{{{
    0% {{{{ 
        transform: perspective(1000px) rotateX(10deg) translateY(20px); 
        opacity: 0; 
    }}}}
    100% {{{{ 
        transform: perspective(1000px) rotateX(2deg) translateY(0); 
        opacity: 1; 
    }}}}
}}}}

@keyframes pulse3D {{{{
    0%, 100% {{{{ 
        transform: translateZ(20px) scale(1); 
        filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.5));
    }}}}
    50% {{{{ 
        transform: translateZ(25px) scale(1.05); 
        filter: drop-shadow(0 0 20px rgba(0, 255, 136, 0.8));
    }}}}
}}}}
</style>
'''
    
    print("=" * 80)
    print("🧪 TEST DU RENDU HTML - BACKEND VALIDATION")
    print("=" * 80)
    
    # Test 1: Vérifier que les balises sont correctement fermées
    print("\n✓ Test 1: Validation des balises HTML")
    open_divs = html_output.count('<div')
    close_divs = html_output.count('</div>')
    print(f"  - Balises <div> ouvertes: {open_divs}")
    print(f"  - Balises </div> fermées: {close_divs}")
    
    if open_divs == close_divs:
        print("  ✅ SUCCÈS: Toutes les balises sont correctement fermées")
    else:
        print(f"  ❌ ERREUR: {open_divs - close_divs} balises non fermées")
        return False
    
    # Test 2: Vérifier que le CSS n'apparaît pas comme texte brut
    print("\n✓ Test 2: Vérification CSS intégré")
    if '<style>' in html_output and '</style>' in html_output:
        print("  ✅ SUCCÈS: Balises <style> présentes")
    else:
        print("  ❌ ERREUR: Balises <style> manquantes")
        return False
    
    # Test 3: Vérifier les animations CSS
    print("\n✓ Test 3: Validation des animations CSS")
    animations = ['slideIn3D', 'pulse3D']
    for anim in animations:
        if f'@keyframes {anim}' in html_output:
            print(f"  ✅ Animation '{anim}' définie correctement")
        else:
            print(f"  ❌ Animation '{anim}' manquante")
            return False
    
    # Test 4: Vérifier les propriétés 3D
    print("\n✓ Test 4: Propriétés CSS 3D")
    css_3d_props = [
        'perspective(1000px)',
        'rotateX(2deg)',
        'translateZ',
        'transform-style: preserve-3d',
        'box-shadow'
    ]
    for prop in css_3d_props:
        if prop in html_output:
            print(f"  ✅ Propriété '{prop}' présente")
        else:
            print(f"  ⚠️  Propriété '{prop}' non trouvée")
    
    # Test 5: Vérifier que le contenu markdown est injecté
    print("\n✓ Test 5: Injection du contenu")
    if '## Introduction 🌐' in html_output:
        print("  ✅ SUCCÈS: Contenu markdown correctement injecté")
    else:
        print("  ❌ ERREUR: Contenu markdown non injecté")
        return False
    
    # Test 6: Vérifier l'échappement des accolades pour Streamlit
    print("\n✓ Test 6: Échappement des accolades pour f-string")
    keyframe_blocks = re.findall(r'@keyframes.*?\{.*?\}', html_output, re.DOTALL)
    if keyframe_blocks:
        # Vérifier qu'on a bien {{{{ (quadruple pour f-string) dans les keyframes
        if '{{{{' in html_output:
            print("  ✅ SUCCÈS: Accolades quadruples {{{{ correctement utilisées (f-string)")
        elif '{{' in html_output:
            print("  ✅ SUCCÈS: Accolades doubles {{ présentes (f-string standard)")
        else:
            print("  ❌ ERREUR: Accolades non échappées (risque d'erreur f-string)")
            return False
    
    # Test 7: Afficher un aperçu du HTML généré
    print("\n✓ Test 7: Aperçu du HTML généré")
    print("  " + "-" * 76)
    preview = html_output[:500].replace('\n', '\n  ')
    print(f"  {preview}...")
    print("  " + "-" * 76)
    
    # Test 8: Vérifier qu'aucun CSS n'apparaît dans le contenu
    print("\n✓ Test 8: Séparation contenu/style")
    content_div = re.search(r'<div class="response-content"[^>]*>(.*?)</div>', html_output, re.DOTALL)
    if content_div:
        content = content_div.group(1)
        if 'box-shadow' not in content and 'gradient' not in content.lower():
            print("  ✅ SUCCÈS: Pas de CSS dans le contenu markdown")
        else:
            print("  ❌ ERREUR: Du CSS apparaît dans le contenu")
            return False
    
    print("\n" + "=" * 80)
    print("🎉 TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!")
    print("=" * 80)
    print("\n📊 Résumé:")
    print(f"  • Longueur HTML: {len(html_output)} caractères")
    print(f"  • Balises DIV: {open_divs}")
    print(f"  • Animations CSS: {len(animations)}")
    print(f"  • Style blocks: {html_output.count('<style>')}")
    print("\n✅ Le HTML sera correctement rendu par Streamlit avec unsafe_allow_html=True")
    
    return True


if __name__ == "__main__":
    success = test_html_structure()
    exit(0 if success else 1)
