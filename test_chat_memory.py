#!/usr/bin/env python3
"""
Test du système de mémoire conversationnelle vectorielle
"""

from chat_memory import (
    load_chat_vectordb,
    add_to_chat_memory,
    search_chat_memory,
    get_conversation_context,
    get_memory_stats,
    clear_chat_memory
)

def test_memory_system():
    print("🧪 Test du système de mémoire conversationnelle\n")
    
    # 1. Charger/créer la base vectorielle
    print("1️⃣ Chargement de la base vectorielle...")
    vectordb, msg = load_chat_vectordb()
    print(f"   {msg}\n")
    
    # 2. Afficher les stats initiales
    print("2️⃣ Statistiques initiales:")
    stats = get_memory_stats(vectordb)
    print(f"   💬 Conversations: {stats['count']}")
    print(f"   💾 Taille: {stats['size']}")
    print(f"   📊 État: {stats['status']}\n")
    
    # 3. Ajouter quelques conversations de test
    print("3️⃣ Ajout de conversations de test...")
    test_conversations = [
        ("Quel temps fait-il à Paris ?", "À Paris, il fait actuellement 18°C avec un ciel partiellement nuageux."),
        ("Comment créer une fonction Python ?", "Voici comment créer une fonction Python:\n```python\ndef ma_fonction(param):\n    return param * 2\n```"),
        ("Explique-moi la photosynthèse", "La photosynthèse est le processus par lequel les plantes convertissent la lumière du soleil en énergie chimique."),
        ("Quelle est la capitale de la France ?", "La capitale de la France est Paris."),
        ("Comment faire une boucle en JavaScript ?", "En JavaScript, une boucle for s'écrit: for(let i=0; i<10; i++) { console.log(i); }")
    ]
    
    for user_msg, ai_msg in test_conversations:
        vectordb = add_to_chat_memory(user_msg, ai_msg, vectordb)
        print(f"   ✅ Ajouté: {user_msg[:50]}...")
    print()
    
    # 4. Afficher les stats après ajout
    print("4️⃣ Statistiques après ajout:")
    stats = get_memory_stats(vectordb)
    print(f"   💬 Conversations: {stats['count']}")
    print(f"   💾 Taille: {stats['size']}")
    print(f"   📊 État: {stats['status']}\n")
    
    # 5. Tester la recherche sémantique
    print("5️⃣ Test de recherche sémantique:")
    test_queries = [
        "météo",
        "programmation Python",
        "capitale française"
    ]
    
    for query in test_queries:
        print(f"\n   🔍 Recherche: '{query}'")
        results = search_chat_memory(query, vectordb, k=2)
        for i, doc in enumerate(results, 1):
            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"      {i}. {content}")
    
    # 6. Tester le contexte de conversation
    print("\n6️⃣ Test de contexte de conversation:")
    context = get_conversation_context("Comment programmer ?", vectordb)
    if context:
        print("   📚 Contexte récupéré:")
        print(f"   {context[:200]}...\n")
    else:
        print("   ℹ️ Aucun contexte pertinent trouvé\n")
    
    print("✅ Test terminé avec succès!")
    return True

if __name__ == "__main__":
    try:
        test_memory_system()
    except Exception as e:
        print(f"❌ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()
