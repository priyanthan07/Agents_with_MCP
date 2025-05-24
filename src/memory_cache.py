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
import openai
from openai import OpenAI

from utils.logger import get_logger
from config import REDIS_CONFIG, CHROMA_CONFIG, OPENAI_CONFIG

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
    created_at: datetime
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
        
        self.report_collection = self.chroma_client.get_or_create_collection(
            name="research_reports",
            metadata={"description": "Final research reports"}
        )

        self.default_ttl = 3600 * 24 * 7  # 1 week default Time to Live
        self.similarity_threshold = 0.8
        
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        
        logger.info("Memory and caching layer initialized successfully")
        
    async def _generate_embedding(self, text: str) -> List[float]:
        
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536
        
    async def check_similar_queries(self, query: str, similarity_threshold: float = None) -> List[Dict[str, Any]]:
        
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        logger.info(f"Checking for similar queries to: {query}")
        
        try:
            query_embedding  = await self._generate_embedding(query)
            
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
        
    async def store_query(self, query: str, plan_id:str) -> str:
        """
            Store a new research query with its embedding for future similarity matching.
        """
        try:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            query_embeddings = await self._generate_embedding(query)
            
            # store data in chromaDB for similarity search
            self.query_collection.add(
                ids=[query_hash],
                documents=[query],
                embeddings=[query_embeddings],
                metadatas=[{
                    "plan_id" : plan_id,
                    "created_at" : datetime.now().isoformat(),
                    "access_counta" : 1
                }]
            )
            
            # store detailed query info in redis
            query_data = {
                "query_hash" : query_hash,
                "original_query" : query,
                "plan_id": plan_id,
                "created_at": datetime.now().isoformat(),
                "access_count": 1
            }
            
            await self.redis_client.setex(
                f"query : {query_hash}",
                self.default_ttl,
                json.dumps(query_data)
            )
            
            logger.info(f"stored query with hash : {query_hash}")
            return query_hash
            
        except Exception as e:
            logger.error(f"Error storing query: {e}")
            return ""
        
    async def store_agent_results(self, 
        agent_type: str, query: str, result_data: Dict[str, Any], plan_id: str, confidence_score: float = 0.7
    ) -> str:
        
        """
            Store results from individual agents with metadata for later retrieval and analysis.
        """
        try:
            result_id = f"{agent_type}_{plan_id}_{datetime.now().timestamp()}"
            
            agent_result = AgentResult(
                result_id = result_id,
                agent_type = agent_type,
                query = query,
                result_data = result_data,
                confidence_score = confidence_score,
                plan_id = plan_id,
                created_at = datetime.now(),
                source_metadata = result_data.get("metadata", {})
            )
        
            result_text = json.dumps(result_data, default=str)
            result_embedding = await self._generate_embedding(result_text)
            
            # store in chromDB for semantic search
            self.results_collection.add(
                ids=[result_id],
                documents=[result_text],
                embeddings=[result_embedding],
                metadatas=[{
                    "agent_type" : agent_type,
                    "plan_id" : plan_id,
                    "confidence_score" : confidence_score,
                    "created_at" : datetime.now().isoformat()
                }]
            )

            # store detailed result in redis
            await self.redis_client.setex(
                f"result : {result_id}",
                self.default_ttl,
                json.dumps(asdict(agent_result), default=str)
            )
            
            # add to plan's result list
            await self.redis_client.lpush(f"plan_results : {plan_id}", result_id)
            
            logger.info(f"Stored result from {agent_type}: {result_id}")
            return result_id
        
        except Exception as e:
            logger.error(f"Error storing agent result: {e}")
            return "" 
    
    async def get_relevant_context(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = await self._generate_embedding(query)
            
            # Search for relevant results
            relevant_results = self.results_collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['metadatas', 'documents', 'distances']
            )
            
            context_data = []
            if relevant_results['ids'][0]:
                for idx, result_id in enumerate(relevant_results['ids'][0]):

                    result_data = await self.redis_client.get(f"result:{result_id}")
                    if result_data:
                        result_obj = json.loads(result_data)
                        context_data.append({
                            "result_id": result_id,
                            "agent_type": result_obj["agent_type"],
                            "confidence_score": result_obj["confidence_score"],
                            "relevance_score": 1 - relevant_results["distances"][0][idx],
                            "summary": result_obj["result_data"].get("summary", "")
                        })
            
            return context_data
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return []
    
    async def store_contradiction(self, contradiction_record: ContradictionRecord) -> str:
        try:
            # Store in ChromaDB for similarity search
            contradiction_text = json.dumps({
                "conflicting_results": contradiction_record.conflicting_results,
                "contradiction_type": contradiction_record.contradiction_type
            })
            contradiction_embedding = await self._generate_embedding(contradiction_text)
            
            self.contradictions_collection.add(
                ids=[contradiction_record.contradiction_id],
                documents=[contradiction_text],
                embeddings=[contradiction_embedding],
                metadatas=[{
                    "contradiction_type": contradiction_record.contradiction_type,
                    "is_resolved": contradiction_record.is_resolved,
                    "created_at": contradiction_record.created_at.isoformat()
                }]
            )
            
            # Store detailed record in Redis
            await self.redis_client.setex(
                f"contradiction:{contradiction_record.contradiction_id}",
                self.default_ttl,
                json.dumps(asdict(contradiction_record), default=str)
            )
            
            logger.info(f"Stored contradiction: {contradiction_record.contradiction_id}")
            return contradiction_record.contradiction_id
            
        except Exception as e:
            logger.error(f"Error storing contradiction: {e}")
            return ""
        
    async def store_final_report(self, plan_id: str, report_content: str, original_query: str) -> str:
        try:
            report_id = f"report_{plan_id}_{datetime.now().timestamp()}"
            report_embedding = await self._generate_embedding(report_content)
            
            self.report_collection.add(
                ids=[report_id],
                documents=[report_content],
                embeddings=[report_embedding],
                metadatas=[{
                    "plan_id": plan_id,
                    "original_query": original_query,
                    "created_at": datetime.now().isoformat()
                }]
            )
            
            # Store detailed report in Redis
            report_data = {
                "report_id": report_id,
                "plan_id": plan_id,
                "original_query": original_query,
                "content": report_content,
                "created_at": datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                f"report:{report_id}",
                self.default_ttl * 4,  # Keep reports longer
                json.dumps(report_data)
            )
            
            await self.redis_client.set(f"plan_report:{plan_id}", report_id)
            
            logger.info(f"Stored final report: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error storing final report: {e}")
            return "" 
        
    async def _get_cached_results(self, plan_id: str) -> List[Dict[str, Any]]:
        try:
            result_ids = await self.redis_client.lrange(f"plan_results:{plan_id}", 0, -1)
            cached_results = []
            
            for result_id in result_ids:
                result_data = await self.redis_client.get(f"result:{result_id}")
                if result_data:
                    cached_results.append(json.loads(result_data))
            
            return cached_results
            
        except Exception as e:
            logger.error(f"Error getting cached results: {e}")
            return []
        
    async def _update_query_access(self, query_id: str):
        """
            Update access count and timestamp for cached queries.
        """
        try:
            access_key = f"query_access:{query_id}"
            await self.redis_client.incr(access_key)
            await self.redis_client.set(f"query_last_access:{query_id}", datetime.now().isoformat())
                    
        except Exception as e:
            logger.error(f"Error updating query access: {e}")
            
    async def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """
            Get comprehensive status information for a research plan.
        """
        try:

            results = await self._get_cached_results(plan_id)
            report_id = await self.redis_client.get(f"plan_report:{plan_id}")
            
            status = {
                "plan_id": plan_id,
                "total_results": len(results),
                "agents_completed": list(set([r["agent_type"] for r in results])),
                "average_confidence": sum([r["confidence_score"] for r in results]) / len(results) if results else 0,
                "has_final_report": report_id is not None,
                "report_id": report_id
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting plan status: {e}")
            return {"error": str(e)}