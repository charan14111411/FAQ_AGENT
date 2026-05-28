import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.rag.retriever import retrieve

async def main():
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    AsyncSessionLocal = sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    print("Testing semantic search retrieval...")
    async with AsyncSessionLocal() as db:
        query = "Will the sensors get damaged during a heavy monsoon?"
        print(f"Query: '{query}'")
        result = await retrieve(db, query, top_k=2)
        print("\nRetrieval Results:")
        print(result)
        
    await engine.dispose()

if __name__ == '__main__':
    asyncio.run(main())
