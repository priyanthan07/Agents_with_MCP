import asyncio
import json
from datetime import datetime
from src.memory_cache import MemoryCacheLayer

async def test_cache_system():
    """Simple test for ChromaDB and Redis functionality"""
    
    # Initialize cache
    cache = MemoryCacheLayer()
    
    # Test data
    test_query = "What are the benefits of artificial intelligence?"
    test_task_id = "test-task-123"
    
    test_data = {
        "web_result": {
            "summary": "AI has many benefits including automation and efficiency",
            "sources": ["source1.com", "source2.com"],
            "key_findings": ["AI improves productivity", "AI reduces errors"]
        },
        "arxiv_result": {
            "papers_found": 15,
            "key_insights": ["ML algorithms are advancing", "Deep learning shows promise"]
        },
        "multimodal_result": {
            "files_processed": 3,
            "insights": ["Video analysis reveals trends", "Audio processing improves"]
        }
    }
    
    print("🧪 Testing Cache System...")
    print("-" * 50)
    
    # Test 1: Store data in Redis
    print("1️⃣ Testing Redis Storage...")
    await cache.store_task_data(test_task_id, test_data)
    print("✅ Data stored in Redis")
    
    # Test 2: Retrieve data from Redis
    print("\n2️⃣ Testing Redis Retrieval...")
    retrieved_data = await cache.retrieve_task_data(test_task_id)
    
    if retrieved_data:
        print("✅ Data retrieved from Redis:")
        print(f"   - Web sources: {len(retrieved_data.get('web_result', {}).get('sources', []))}")
        print(f"   - Papers found: {retrieved_data.get('arxiv_result', {}).get('papers_found', 0)}")
        print(f"   - Files processed: {retrieved_data.get('multimodal_result', {}).get('files_processed', 0)}")
    else:
        print("❌ Failed to retrieve data from Redis")
    
    # Test 3: Store query in ChromaDB
    print("\n3️⃣ Testing ChromaDB Storage...")
    await cache.store_query_with_task_id(test_query, test_task_id)
    print("✅ Query stored in ChromaDB")
    
    # Test 4: Find similar query in ChromaDB
    print("\n4️⃣ Testing ChromaDB Similarity Search...")
    
    # Test exact match
    similar_task_id = await cache.find_similar_query(test_query)
    if similar_task_id:
        print(f"✅ Found exact match: {similar_task_id}")
    else:
        print("❌ Exact match not found")
    
    # Test similar query
    similar_query = "What are AI benefits and advantages?"
    similar_task_id = await cache.find_similar_query(similar_query)
    if similar_task_id:
        print(f"✅ Found similar query match: {similar_task_id}")
    else:
        print("ℹ️ No similar query found above threshold")
    
    # Test different query
    different_query = "How to cook pasta?"
    different_task_id = await cache.find_similar_query(different_query)
    if different_task_id:
        print(f"⚠️ Unexpected match for different query: {different_task_id}")
    else:
        print("✅ No match for different query (as expected)")
    
    print("\n" + "=" * 50)
    print("🎉 Cache system test completed!")

async def test_multiple_queries():
    """Test with multiple queries to see similarity matching"""
    
    cache = MemoryCacheLayer()
    
    queries_and_data = [
        {
            "query": "Benefits of renewable energy",
            "task_id": "renewable-123",
            "data": {"result": "Solar and wind energy are clean"}
        },
        {
            "query": "Machine learning applications",
            "task_id": "ml-456", 
            "data": {"result": "ML is used in healthcare and finance"}
        },
        {
            "query": "Climate change effects",
            "task_id": "climate-789",
            "data": {"result": "Rising temperatures affect ecosystems"}
        }
    ]
    
    print("\n🔄 Testing Multiple Queries...")
    print("-" * 50)
    
    # Store all queries
    for item in queries_and_data:
        await cache.store_query_with_task_id(item["query"], item["task_id"])
        await cache.store_task_data(item["task_id"], item["data"])
        print(f"✅ Stored: {item['query'][:30]}...")
    
    # Test similarity searches
    test_queries = [
        "What are the advantages of renewable energy?",  # Should match renewable
        "Applications of machine learning",              # Should match ML
        "How does climate change impact environment?",   # Should match climate
        "Best pizza recipes"                            # Should not match any
    ]
    
    print("\n🔍 Testing Similarity Searches...")
    for test_query in test_queries:
        result = await cache.find_similar_query(test_query)
        if result:
            print(f"✅ '{test_query[:25]}...' → Found: {result}")
        else:
            print(f"❌ '{test_query[:25]}...' → No match")

async def cleanup_test_data():
    """Clean up all test data from Redis and ChromaDB"""
    
    cache = MemoryCacheLayer()
    
    print("\n🧹 Cleaning up test data...")
    print("-" * 50)
    
    # Clean Redis test data - Updated to use new Redis operation pattern
    test_task_ids = ["test-task-123", "renewable-123", "ml-456", "climate-789"]
    
    for task_id in test_task_ids:
        try:
            # Use the new Redis operation pattern
            async def _cleanup_operation(redis_client, task_id):
                metadata_key = f"task:{task_id}:metadata"
                metadata_json = await redis_client.get(metadata_key)
                
                if metadata_json:
                    metadata_dict = json.loads(metadata_json)
                    data_keys = metadata_dict.get("data_keys", [])
                    
                    # Delete all data keys
                    for key in data_keys:
                        redis_key = f"task:{task_id}:{key}"
                        await redis_client.delete(redis_key)
                    
                    # Delete metadata
                    await redis_client.delete(metadata_key)
                    return True
                return False
            
            success = await cache._execute_redis_operation(_cleanup_operation, task_id)
            if success:
                print(f"✅ Cleaned Redis data for task: {task_id}")
            else:
                print(f"ℹ️ No data found for task: {task_id}")
            
        except Exception as e:
            print(f"⚠️ Error cleaning Redis task {task_id}: {e}")
    
    # Clean ChromaDB test data
    try:
        # Get all items from collection
        all_items = cache.query_collection.get()
        
        if all_items['ids']:
            # Delete all items
            cache.query_collection.delete(ids=all_items['ids'])
            print(f"✅ Cleaned ChromaDB collection ({len(all_items['ids'])} items)")
        else:
            print("ℹ️ ChromaDB collection already empty")
            
    except Exception as e:
        print(f"⚠️ Error cleaning ChromaDB: {e}")
    
    print("🎉 Cleanup completed!")

async def test_event_loop_safety():
    """Test that new Redis client pattern works across different event loops"""
    
    print("\n🔄 Testing Event Loop Safety...")
    print("-" * 50)
    
    cache = MemoryCacheLayer()
    test_task_id = "event-loop-test"
    test_data = {"test": "event loop safety data"}
    
    # Store data
    await cache.store_task_data(test_task_id, test_data)
    print("✅ Data stored in first event loop")
    
    # Retrieve in same event loop
    result1 = await cache.retrieve_task_data(test_task_id)
    print(f"✅ Same loop retrieval: {bool(result1)}")
    
    # Test with different event loop (simulate Streamlit's pattern)
    def run_in_new_loop():
        async def retrieve_in_new_loop():
            cache2 = MemoryCacheLayer()
            result2 = await cache2.retrieve_task_data(test_task_id)
            return result2
        
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(retrieve_in_new_loop())
        finally:
            new_loop.close()
    
    # This should now work without "Future attached to different loop" errors
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_new_loop)
        result2 = future.result()
    
    print(f"✅ Different loop retrieval: {bool(result2)}")
    
    # Cleanup
    async def _cleanup_operation(redis_client, task_id):
        await redis_client.delete(f"task:{task_id}:metadata")
        await redis_client.delete(f"task:{task_id}:test")
        return True
    
    await cache._execute_redis_operation(_cleanup_operation, test_task_id)
    print("✅ Test data cleaned up")

if __name__ == "__main__":
    print("🚀 Starting Cache System Tests...")
    print("Make sure Redis and ChromaDB are running!")
    print()
    
    try:
        # Run basic test
        asyncio.run(test_cache_system())
        
        # Run multiple queries test
        asyncio.run(test_multiple_queries())
        
        # Test event loop safety
        asyncio.run(test_event_loop_safety())
        
    finally:
        # Always cleanup test data
        asyncio.run(cleanup_test_data())