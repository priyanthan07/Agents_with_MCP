import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import redis.asyncio as redis
import chromadb
from chromadb.config import Settings
import numpy as np
# from sentence_transformers import SentenceTransformer

from utils.logger import get_logger
from config import REDIS_CONFIG, CHROMA_CONFIG

logger = get_logger(__name__)

@dataclass
class CachedQuery:
    query_hash: str
    original_query: str
    query_embedding: List[float]
    plan_id: str
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = datetime.now()
    result_summary: Optional[Dict] =None
    
@dataclass
class AgentResult:
    result_id: str
    agent_type: str
    query: str
    result_data: Dict[str, Any]
    confidence_score: float
    plan_id: str
    created_id: datetime
    source_metadata: Dict[str, Any]
    
@dataclass
class ContradictionRecord:
    contradiction_id: str
    conflicting_results: List[str]
    contradiction_type: str
    resolution_strategy: Optional[str] = None
    resolution_data: Optional[Dict] = None
    is_resolved: bool = False
    created_at: datetime = datetime.now()
    
class MemoryCacheLayer:
    """
        Advanced memory and caching system that handles query similarity matching,
        result storage, and semantic search across all research data.
    """
    def __init__(self):
        self.redis_client = redis.Redis(
            host = REDIS_CONFIG["redis_host"],
            port = REDIS_CONFIG["redis_port"],
            db = REDIS_CONFIG["redis_db"],
            decode_responses=True
        )
        
        self.chroma_client = chromadb.PersistentClient(
            path = CHROMA_CONFIG["chroma_persist_directory"], 
            settings=Settings(anonymized_telemetry=False)
        ) 
        
        self.query_collection = self.chroma_client.get_or_create_collection(
            name="research_queries",
            metadata= {"description": "Stored research queries with embeddings"}
        )
        
        self.results_collection = self.chroma_client.get_or_create_collection(
            name="agent_results", 
            metadata={"description": "Results from individual research agents"}
        )
        
        self.contradictions_collection = self.chroma_client.get_or_create_collection(
            name="contradictions", 
            metadata={"description": "Contradiction records and resolutions"}
        )

        self.default_ttl = 3600 * 24 * 7  # 1 week default TTL
        self.similarity_threshold = 0.8
        
        logger.info("Memory and caching layer initialized successfully")
        
        
    async def check_similar_queries(self, query: str, similarity_threshold: float = None) -> List[Dict[str, Any]]:
        
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        logger.info(f"Checking for similar queries to: {query}")
        
        try:
            query_embedding  = "" # need to write the code using openai
            
            similar_results = self.query_collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=['metadatas', 'documents', 'distances']
            )
            
            similar_queries = []
            if similar_results['ids'][0]:
                for idx, query_id in enumerate(similar_results['ids'][0]):
                    distance = similar_results["distances"][0][idx]
                    similarity = 1 - distance
                    
                    if similarity >= similarity_threshold:
                        metadata = similar_results['metadatas'][0][idx]
                        cached_query_data = {
                            "query_id": query_id,
                            "original_query": similar_results['documents'][0][idx],
                            "similarity_score": similarity,
                            "plan_id": metadata.get("plan_id"),
                            "created_at": metadata.get("created_at"),
                            "access_count": metadata.get("access_count", 0)
                        } 

                        cached_data = await self._get_cached_results(metadata["plan_id"])
                        cached_query_data["cached_results"] = cached_data
                        
                        similar_queries.append(cached_query_data)
                        await self._update_query_access(query_id)
            logger.info(f"Found {len(similar_queries)} similar queries above threshold {similarity_threshold}")
            return similar_queries
            
        except Exception as e:
            logger.error(f"Error checking similar queries: {e}")
            return []
        
        
    