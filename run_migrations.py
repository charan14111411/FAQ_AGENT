import asyncio
import os
from app.db import get_db_direct
from app.logger import get_logger

logger = get_logger()

async def run_migrations():
    conn = None
    try:
        print("Connecting to the database...")
        conn = await get_db_direct()
        print("Connected successfully!")
        
        migrations_dir = "migrations"
        migration_files = sorted([f for f in os.listdir(migrations_dir) if f.endswith(".sql")])
        
        for file_name in migration_files:
            file_path = os.path.join(migrations_dir, file_name)
            print(f"Applying migration: {file_name}...")
            
            with open(file_path, "r", encoding="utf-8") as f:
                sql = f.read()
                
            # Execute the SQL commands
            try:
                await conn.execute(sql)
                print(f"Successfully applied {file_name}")
            except Exception as e:
                if file_name == "005_faq_embeddings.sql":
                    print(f"Warning: Skipping {file_name} because pgvector is not available: {e}")
                    logger.warning(f"Skipping {file_name} due to missing pgvector: {e}")
                else:
                    raise e
            
        print("All migrations applied successfully!")
    except Exception as e:
        print(f"Migration failed: {e}")
        logger.error(f"Migration failed: {e}")
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    asyncio.run(run_migrations())
