import uuid
from app.data.faq import FAQ
from app.rag.embedder import embed_text
from app.config import settings
from app.logger import get_logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList

logger = get_logger()

def _get_client():
    if settings.QDRANT_URL.startswith("http://") or settings.QDRANT_URL.startswith("https://"):
        return AsyncQdrantClient(url=settings.QDRANT_URL)
    else:
        return AsyncQdrantClient(location=settings.QDRANT_URL)

async def seed():
    client = None
    try:
        client = _get_client()
        
        # Ensure collection exists
        if not await client.collection_exists(settings.QDRANT_COLLECTION):
            # Embed a sample to get dimension size dynamically
            sample_emb = await embed_text("sample")
            dim = len(sample_emb)
            await client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION} with dimension {dim}")

        # 1. DELETE outdated FAQs (Pruning step)
        # Determine current expected point IDs based on a deterministic hash of the question
        current_ids = {str(uuid.uuid5(uuid.NAMESPACE_DNS, item["question"])) for item in FAQ}
        
        # Retrieve all point IDs in Qdrant to find obsolete ones
        obsolete_ids = []
        offset = None
        while True:
            scroll_result = await client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=100,
                with_payload=False,
                with_vectors=False,
                offset=offset
            )
            points = scroll_result[0]
            offset = scroll_result[1]
            for p in points:
                if str(p.id) not in current_ids:
                    obsolete_ids.append(p.id)
            if offset is None:
                break

        if obsolete_ids:
            await client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=PointIdsList(points=obsolete_ids)
            )
            logger.info(f"Pruned outdated FAQ records: {len(obsolete_ids)}")

        # 2. INSERT or UPDATE (Upsert step)
        inserted_count = 0
        updated_count = 0
        
        for item in FAQ:
            category = item["category"]
            question = item["question"]
            answer = item["answer"]
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, question))
            
            try:
                existing_points = await client.retrieve(
                    collection_name=settings.QDRANT_COLLECTION,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=False
                )
            except Exception:
                existing_points = []
            
            existing = existing_points[0] if existing_points else None
            
            if not existing:
                # New Question -> Embed and Insert
                text_to_embed = f"{question} {answer}"
                embedding = await embed_text(text_to_embed)
                
                await client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "category": category,
                                "question": question,
                                "answer": answer
                            }
                        )
                    ]
                )
                inserted_count += 1
                
            elif existing.payload.get("answer") != answer or existing.payload.get("category") != category:
                # Answer or Category changed -> Re-embed and Update
                text_to_embed = f"{question} {answer}"
                embedding = await embed_text(text_to_embed)
                
                await client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "category": category,
                                "question": question,
                                "answer": answer
                            }
                        )
                    ]
                )
                updated_count += 1
                
        logger.info(f"FAQ Sync complete. Added: {inserted_count}, Updated: {updated_count}.")
        
    except Exception as e:
        logger.error(f"Failed to sync FAQ embeddings in Qdrant: {e}")
        raise e
    finally:
        if client:
            await client.close()

