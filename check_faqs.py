import asyncio
from app.config import settings
from qdrant_client import AsyncQdrantClient

def _get_client():
    if settings.QDRANT_URL.startswith("http://") or settings.QDRANT_URL.startswith("https://"):
        return AsyncQdrantClient(url=settings.QDRANT_URL)
    else:
        return AsyncQdrantClient(location=settings.QDRANT_URL)

async def run():
    client = _get_client()
    try:
        if not await client.collection_exists(settings.QDRANT_COLLECTION):
            print(f"Collection {settings.QDRANT_COLLECTION} does not exist in Qdrant.")
            return

        res = await client.count(collection_name=settings.QDRANT_COLLECTION)
        print(f"Total FAQs in Qdrant: {res.count}")
        
        # Scroll to get all points
        points, _ = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        print("\nAll FAQ questions in Qdrant:")
        questions = sorted([p.payload.get("question") for p in points if p.payload])
        for q in questions:
            print(f"  - {q}")
    finally:
        await client.close()

asyncio.run(run())

