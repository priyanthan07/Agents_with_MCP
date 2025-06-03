import asyncio
import redis.asyncio as redis
import chromadb
import requests

async def test_services():
    print("🧪 Testing Docker services...")
    
    # Test Redis
    print("\n📡 Testing Redis connection...")
    try:
        redis_client = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Test ping
        pong = await redis_client.ping()
        print(f"✅ Redis ping: {pong}")
        
        # Test set/get
        await redis_client.set("test_key", "Hello Redis!")
        value = await redis_client.get("test_key")
        print(f"✅ Redis set/get: {value}")
        
        # Cleanup
        await redis_client.delete("test_key")
        await redis_client.close()
        
    except Exception as e:
        print(f"❌ Redis failed: {e}")
    
    # Test ChromaDB
    print("\n🔍 Testing ChromaDB connection...")
    try:
        # Test HTTP endpoint
        response = requests.get("http://localhost:8000/api/v1/heartbeat")
        print(f"✅ ChromaDB heartbeat: {response.status_code}")
        
        # Test client
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Test collection operations
        collection = chroma_client.get_or_create_collection("test_collection")
        print(f"✅ ChromaDB collection created: {collection.name}")
        
        # Test embedding storage
        collection.add(
            documents=["This is a test document"],
            ids=["test_1"]
        )
        
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )
        print(f"✅ ChromaDB query result: {len(results['documents'][0])} documents")
        
        # Cleanup
        chroma_client.delete_collection("test_collection")
        print("✅ Test collection cleaned up")
        
    except Exception as e:
        print(f"❌ ChromaDB failed: {e}")
    
    print("\n🎉 Service tests completed!")

if __name__ == "__main__":
    asyncio.run(test_services())