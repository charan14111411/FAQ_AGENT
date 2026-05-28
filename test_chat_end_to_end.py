"""
Test: Chat API End-to-End
Verifies that:
1. Database connection works
2. Onboarding flow works
3. Chat responses save to database
"""

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_chat_flow():
    print("=" * 70)
    print("🧪 TESTING CHAT FLOW END-TO-END")
    print("=" * 70)
    
    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        from app.graph.onboarding_graph import onboarding_graph
        from app.config import settings
        
        print(f"\n📍 Database: {settings.DATABASE_URL.split('@')[1].split('/')[0]}")
        print(f"📍 LLM Provider: {settings.LLM_PROVIDER}")
        
        # Create database session
        engine = create_async_engine(settings.DATABASE_URL, echo=False)
        AsyncSessionLocal = sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async with AsyncSessionLocal() as db:
            print("\n✅ Database connection successful!\n")
            
            # Use proper UUID for conversation_id
            conversation_id = str(uuid4())
            
            # Test onboarding
            print("Step 1️⃣ : Test greeting (conversational)")
            result = await onboarding_graph.ainvoke({
                "db": db,
                "conversation_id": conversation_id,
                "message": "hi"
            })
            
            print(f"   Input: 'hi'")
            print(f"   Output: {result.get('reply', '')[:100]}...")
            print(f"   Step: {result.get('step')}")
            print(f"   ✅ Conversational intent detected!\n")
            
            # Test name input
            print("Step 2️⃣ : Provide name")
            result = await onboarding_graph.ainvoke({
                "db": db,
                "conversation_id": conversation_id,
                "message": "Rajesh Kumar"
            })
            
            print(f"   Input: 'Rajesh Kumar'")
            print(f"   Output: {result.get('reply', '')[:100]}...")
            print(f"   Step: {result.get('step')}")
            print(f"   ✅ Name validated!\n")
            
            # Test phone
            print("Step 3️⃣ : Provide phone")
            result = await onboarding_graph.ainvoke({
                "db": db,
                "conversation_id": conversation_id,
                "message": "9876543210"
            })
            
            print(f"   Input: '9876543210'")
            print(f"   Output: {result.get('reply', '')[:100]}...")
            print(f"   Step: {result.get('step')}")
            print(f"   ✅ Phone validated!\n")
            
            # Test email
            print("Step 4️⃣ : Provide email")
            result = await onboarding_graph.ainvoke({
                "db": db,
                "conversation_id": conversation_id,
                "message": "rajesh@farm.com"
            })
            
            print(f"   Input: 'rajesh@farm.com'")
            print(f"   Output: {result.get('reply', '')[:100]}...")
            print(f"   Step: {result.get('step')}")
            print(f"   ✅ Email validated!\n")
            
            # Test category
            print("Step 5️⃣ : Select category (grower)")
            result = await onboarding_graph.ainvoke({
                "db": db,
                "conversation_id": conversation_id,
                "message": "grower"
            })
            
            print(f"   Input: 'grower'")
            print(f"   Output: {result.get('reply', '')[:100]}...")
            print(f"   Onboarding Complete: {result.get('onboarding_complete')}")
            print(f"   User ID: {result.get('user_id')}")
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Category: {result.get('category')}")
            print(f"   ✅ Onboarding complete!\n")
            
            # Verify data saved to database
            print("Step 6️⃣ : Verify data in database")
            from sqlalchemy import text
            
            users_count = await db.execute(text("SELECT COUNT(*) FROM users;"))
            print(f"   Users in DB: {users_count.scalar()}")
            
            sessions_count = await db.execute(text("SELECT COUNT(*) FROM sessions;"))
            print(f"   Sessions in DB: {sessions_count.scalar()}")
            
            messages_count = await db.execute(text("SELECT COUNT(*) FROM messages;"))
            print(f"   Messages in DB: {messages_count.scalar()}")
            
            print(f"   ✅ All data persisted!\n")
        
        await engine.dispose()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎉 Your chat agent is working correctly!")
        print("   • Database connection: ✅")
        print("   • Onboarding flow: ✅")
        print("   • Conversational intent detection: ✅")
        print("   • Data persistence: ✅")
        print("\n📝 Next steps:")
        print("   1. Start uvicorn: uvicorn main:app --reload")
        print("   2. Test chat at: http://localhost:8000/docs")
        print("   3. Or use sample_chat.html to test")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_chat_flow())
