"""
Database Connection Diagnostic
Helps troubleshoot PostgreSQL connection issues
"""

import asyncio
import asyncpg
import os
from pathlib import Path

# Load .env manually
env_path = Path(".env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

DATABASE_URL = os.environ.get("DATABASE_URL", "")
print(f"📍 Database URL: {DATABASE_URL}\n")

# Parse connection details
if DATABASE_URL:
    # Expected format: postgresql+asyncpg://user:password@host:port/db
    parts = DATABASE_URL.replace("postgresql+asyncpg://", "").split("@")
    if len(parts) == 2:
        creds, host_db = parts
        user, password = creds.split(":")
        host, db = host_db.split("/")
        if ":" in host:
            host, port = host.split(":")
            port = int(port)
        else:
            port = 5432
        
        print(f"🔍 Connection Details:")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   User: {user}")
        print(f"   Database: {db}")
        print(f"   Password: {'*' * len(password)}\n")
        
        async def test_connection():
            print("🧪 Testing connection...\n")
            
            # Test 1: Try localhost
            print("1️⃣  Trying localhost...")
            try:
                conn = await asyncpg.connect(
                    host="localhost",
                    port=port,
                    user=user,
                    password=password,
                    database=db,
                    timeout=5
                )
                version = await conn.fetchval("SELECT version();")
                await conn.close()
                print(f"   ✅ SUCCESS (localhost)")
                print(f"   PostgreSQL: {version.split(',')[0]}\n")
                return True
            except Exception as e:
                print(f"   ❌ FAILED: {str(e)[:100]}\n")
            
            # Test 2: Try 127.0.0.1
            print("2️⃣  Trying 127.0.0.1...")
            try:
                conn = await asyncpg.connect(
                    host="127.0.0.1",
                    port=port,
                    user=user,
                    password=password,
                    database=db,
                    timeout=5
                )
                version = await conn.fetchval("SELECT version();")
                await conn.close()
                print(f"   ✅ SUCCESS (127.0.0.1)")
                print(f"   PostgreSQL: {version.split(',')[0]}\n")
                return True
            except Exception as e:
                print(f"   ❌ FAILED: {str(e)[:100]}\n")
            
            # Test 3: Try ::1 (IPv6)
            print("3️⃣  Trying ::1 (IPv6)...")
            try:
                conn = await asyncpg.connect(
                    host="::1",
                    port=port,
                    user=user,
                    password=password,
                    database=db,
                    timeout=5
                )
                version = await conn.fetchval("SELECT version();")
                await conn.close()
                print(f"   ✅ SUCCESS (::1)")
                print(f"   PostgreSQL: {version.split(',')[0]}\n")
                return True
            except Exception as e:
                print(f"   ❌ FAILED: {str(e)[:100]}\n")
            
            return False
        
        result = asyncio.run(test_connection())
        
        if result:
            print("=" * 70)
            print("✅ DATABASE CONNECTION WORKING!")
            print("=" * 70)
            print("\n🔧 SOLUTIONS:")
            print("   1. If only IPv6 (::1) works:")
            print("      → Update .env to use IPv6:")
            print("      → DATABASE_URL=postgresql+asyncpg://user:pass@[::1]:5433/123")
            print("\n   2. If only 127.0.0.1 works:")
            print("      → Update .env to use IPv4:")
            print("      → DATABASE_URL=postgresql+asyncpg://user:pass@127.0.0.1:5433/123")
            print("\n   3. Configure PostgreSQL to listen on both IPv4 and IPv6:")
            print("      → Edit postgresql.conf: listen_addresses = '*'")
            print("      → Then restart PostgreSQL")
        else:
            print("=" * 70)
            print("❌ CANNOT CONNECT TO DATABASE")
            print("=" * 70)
            print("\n💡 Troubleshooting steps:")
            print("   1. Is PostgreSQL running?")
            print("      → Check Windows Services or run: pg_ctl status")
            print("\n   2. Is the port correct? (5433 not 5432)")
            print("      → Check: netstat -ano | findstr 5433")
            print("\n   3. Are credentials correct?")
            print("      → User: postgres")
            print("      → Password: admin123")
            print("      → Database: 123")
            print("\n   4. Is the database created?")
            print("      → Run: CREATE DATABASE \"123\";")
            print("\n   5. Check PostgreSQL logs:")
            print("      → Windows: C:\\Program Files\\PostgreSQL\\data\\pg_log\\")
else:
    print("❌ DATABASE_URL not found in .env")
