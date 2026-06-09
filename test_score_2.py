import asyncio
from app.config import settings
from qdrant_client import AsyncQdrantClient
from app.rag.embedder import embed_text

async def main():
    q = "is there any other devices that u are having along with microclime"
    client = AsyncQdrantClient(url=settings.QDRANT_URL) if settings.QDRANT_URL.startswith("http") else AsyncQdrantClient(location=settings.QDRANT_URL)
    emb = await embed_text(q)
    res = await client.query_points(collection_name=settings.QDRANT_COLLECTION, query=emb, limit=5)
    for hit in res.points:
        print(f"Score: {hit.score:.4f} | Q: {hit.payload.get('question')}")
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
