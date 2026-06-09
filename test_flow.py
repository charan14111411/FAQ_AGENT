import asyncio
from langgraph.checkpoint.memory import MemorySaver
from app.agents.graph import faq_graph_builder
from app.db import AsyncSessionLocal
from sqlalchemy import text

async def run_exit_skip_test():
    print("\n=============================================")
    print("RUNNING SKIP FLOW TEST (Preserving all users)")
    print("=============================================")
    graph = faq_graph_builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test_thread_skip_flow"}}

    # 1. Start: Classify category
    print("\n--- 1. START (Category selection) ---")
    res = await graph.ainvoke({"user_input": "i am a grower", "step": "start"}, config)
    print("Bot Reply:", res.get("reply"))
    print("Next Step:", res.get("step"))

    # 2. Enter Name
    print("\n--- 2. AWAIT_NAME (Enter Name) ---")
    res = await graph.ainvoke({"user_input": "Alice Green", "step": res.get("step")}, config)
    print("Bot Reply:", res.get("reply"))
    print("Next Step:", res.get("step"))
    temp_user_id = res.get("user_id")
    session_id = res.get("session_id")
    print(f"Temporary User ID created: {temp_user_id}")
    print(f"Session ID created: {session_id}")

    # 3. Chatting (FAQ query)
    print("\n--- 3. CHATTING (Ask FAQ) ---")
    res = await graph.ainvoke({"user_input": "Can I compare two blocks of my estate?", "step": res.get("step")}, config)
    print("Bot Reply:", res.get("reply"))
    print("Next Step:", res.get("step"))

    # 4. First Farewell (polite confirmation 1)
    print("\n--- 4. FAREWELL 1 (polite confirmation 1) ---")
    res = await graph.ainvoke({"user_input": "bye", "step": res.get("step")}, config)
    print("Bot Reply:", res.get("reply"))
    print("Next Step:", res.get("step"))

    # 5. Second Farewell (polite confirmation 2)
    print("\n--- 5. FAREWELL 2 (polite confirmation 2) ---")
    res = await graph.ainvoke({"user_input": "quit", "step": res.get("step")}, config)
    print("Bot Reply:", res.get("reply"))
    print("Next Step:", res.get("step"))

    # 6. Third Farewell (transitions to await_phone_on_exit)
    print("\n--- 6. FAREWELL 3 (starts exit onboarding) ---")
    res = await graph.ainvoke({"user_input": "that is all", "step": res.get("step")}, config)
    print("Bot Reply:", res.get("reply"))
    print("Next Step:", res.get("step"))

    # 7. Skip Phone Number on Exit (routes to email)
    print("\n--- 7. AWAIT_PHONE_ON_EXIT (Skip) ---")
    res = await graph.ainvoke({"user_input": "skip", "step": res.get("step")}, config)
    print("Bot Reply:", res.get("reply"))
    print("Next Step:", res.get("step"))

    # 8. Skip Email on Exit (professional goodbye and wrap up)
    print("\n--- 8. AWAIT_EMAIL_ON_EXIT (Skip) ---")
    res = await graph.ainvoke({"user_input": "no thanks", "step": res.get("step")}, config)
    print("Bot Reply:", res.get("reply"))
    print("Next Step:", res.get("step"))

    # Verify database: check that Alice Green is still preserved in the database
    print("\n--- Verifying database preservation ---")
    async with AsyncSessionLocal() as db:
        stmt = text("SELECT id, name, phone, email FROM users WHERE id = :user_id")
        user_exists = (await db.execute(stmt, {"user_id": temp_user_id})).fetchone()
        if user_exists:
            print(f"Success! Temporary User '{user_exists[1]}' (id={user_exists[0]}) is still preserved in the database.")
            print(f"Details: phone={user_exists[2]}, email={user_exists[3]}")
        else:
            print("Error: Temporary user was deleted!")

async def clean_database():
    print("\nCleaning test data from database...")
    async with AsyncSessionLocal() as db:
        await db.execute(text("DELETE FROM users WHERE name IN ('Alice Green')"))
        await db.commit()
    print("Cleaned up successfully.")

async def main():
    await clean_database()
    try:
        await run_exit_skip_test()
    finally:
        await clean_database()

if __name__ == "__main__":
    asyncio.run(main())
