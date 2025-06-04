import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import redis.asyncio as redis
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import chromadb.utils.embedding_functions as embedding_functions
from util.logger import get_logger
from config import REDIS_CONFIG, CHROMA_CONFIG, OPENAI_CONFIG

logger = get_logger(__name__)

@dataclass
class QueryMetadata:
    task_id: str
    created_at: str
    query_text: str
    
@dataclass
class TaskMetadata:
    task_id: str
    created_at: str
    data_keys: List[str]
    
class MemoryCacheLayer:
    """
        Advanced memory and caching system that handles query similarity matching,
        result storage, and semantic search across all research data.
    """
    def __init__(self):
        
        self.redis_config = {
            "host" : REDIS_CONFIG["redis_host"],
            "port" : REDIS_CONFIG["redis_port"],
            "db" : REDIS_CONFIG["redis_db"],
            "decode_responses" : True            
        }
        
        self.chroma_client = chromadb.HttpClient(
            host=CHROMA_CONFIG["chroma_host"], 
            port=CHROMA_CONFIG["chroma_port"]
        )
        
        # openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        #         api_key=OPENAI_CONFIG["api_key"],
        #         model_name=OPENAI_CONFIG["embd_model"]
        #     )
        
        self.query_collection = self.chroma_client.get_or_create_collection(
            name="user_queries",
            # embedding_function= openai_ef
        )
        
        self.similarity_threshold = 0.7
        
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        
        logger.info("Memory and caching layer initialized successfully")
        
    # async def _generate_embedding(self, text: str) -> List[float]:
    #     try:
    #         response = self.client.embeddings.create(
    #             model="text-embedding-3-small",
    #             input=text
    #         )
    #         return response.data[0].embedding
            
    #     except Exception as e:
    #         logger.error(f"Error generating embedding: {e}")
    #         # Return a zero vector as fallback
    #         return [0.0] * 1536
    
    
    async def _get_redis_client(self) -> redis.Redis:
        """
        Create a fresh Redis client for the current event loop.
        """
        try:
            return redis.Redis(**self.redis_config)
        
        except Exception as e:
            logger.error(f"Failed to create Redis client: {e}")
            raise
        
    async def _execute_redis_operation(self, ope_function, *args, **kwargs):
        redis_client = None
        try:
            redis_client = await self._get_redis_client()
            result = await ope_function(redis_client, *args, **kwargs)
            return result
        
        except Exception as e:
            logger.error(f"Redis operation failed: {e}")
            raise
        
        finally:
            if redis_client:
                try:
                    await redis_client.aclose()
                except Exception as cleanup_error:
                    logger.warning(f"Redis client cleanup failed: {cleanup_error}")
        
    async def find_similar_query(self, query: str) -> Optional[str]:
        """
            Find similar query with similarity > 0.95.
            Returns task_id if found, None otherwise.
        """
        try:
            # query_embedding = await self._generate_embedding(query)
            
            results = self.query_collection.query(
                query_texts=[query],
                n_results=1,
                include=['metadatas', 'distances']
            )
            
            if results['ids'][0]:
                distance = results['distances'][0][0]
                similarity = 1 - distance
                logger.info(f"with similarity {similarity:.3f} / {self.similarity_threshold}")
                
                if similarity >= self.similarity_threshold:
                    task_id = results['metadatas'][0][0]['task_id']
                    logger.info(f"Found similar query with similarity {similarity:.3f}, task_id: {task_id}")
                    
                    return task_id
                
            logger.info("No similar query found above threshold")
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar query: {e}")
            return None
        
    async def store_query_with_task_id(self, query: str, task_id: str):
        """
            Store query embedding in ChromaDB with task_id metadata.
        """
        try:
            # query_embedding = await self._generate_embedding(query)
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            metadata = QueryMetadata(
                task_id=task_id,
                created_at=datetime.now().isoformat(),
                query_text=query
            )
            
            self.query_collection.add(
                ids=[query_hash],
                documents=[query],
                metadatas=[asdict(metadata)]
            )
            
            logger.info(f"Stored query embedding for task_id: {task_id}")
            
        except Exception as e:
            logger.error(f"Error storing query: {e}")
            
    async def store_task_data(self, task_id: str, data: Dict[str, Any]):
        """
            Store all research data under task_id in Redis.
        """
        async def _store_operation(redis_client, task_id, data):
            for key, value in data.items():
                redis_key = f"task:{task_id}:{key}"
                await redis_client.set(redis_key, json.dumps(value, default=str))
                
            metadata = TaskMetadata(
                task_id=task_id,
                created_at=datetime.now().isoformat(),
                data_keys=list(data.keys())
            )
            
            await redis_client.set(f"task:{task_id}:metadata", json.dumps(asdict(metadata)))
            logger.info(f"Stored task data for task_id: {task_id}")
                    
        try:
            await self._execute_redis_operation(_store_operation, task_id, data)
        except Exception as e:
            logger.error(f"Error storing task data: {e}")
            
    async def retrieve_task_data(self, task_id: str) -> Dict[str, Any]:
        """
            Retrieve all data for a task_id from Redis.
        """
        async def _retrieve_operation(redis_client, task_id):
            metadata_key = f"task:{task_id}:metadata"
            metadata_json = await redis_client.get(metadata_key)
            
            if not metadata_json:
                logger.warning(f"No metadata found for task_id: {task_id}")
                return {}
            
            metadata_dict = json.loads(metadata_json)
            metadata = TaskMetadata(**metadata_dict)
            data = {}
            
            for key in metadata.data_keys:
                redis_key = f"task:{task_id}:{key}"
                value_json = await redis_client.get(redis_key)
                if value_json:
                    data[key] = json.loads(value_json)
                    
            logger.info(f"Retrieved task data for task_id: {task_id}")
            return data  
        
        try:     
            return await self._execute_redis_operation(_retrieve_operation, task_id)
        except Exception as e:
            logger.error(f"Error retrieving task data: {e}")
            return {}