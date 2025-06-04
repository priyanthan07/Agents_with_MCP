import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel
import uuid
from util.logger import get_logger
from config import OPENAI_CONFIG

from agents.web_agent import WebResearchAgent, WebResearchResult
from agents.arxiv_agent import ArxivResearchAgent, GlobalResearchResult
from agents.multimodal_agent import MultiModalResearchAgent, MultiModalResearchResult

from src.memory_cache import MemoryCacheLayer
from src.validator import ResearchValidator

logger = get_logger(__name__)

@dataclass
class Contradiction:
    id: str
    source1: str
    source2: str
    claim1: str
    claim2: str
    topic: str
    severity: str
    
@dataclass
class Resolution:
    contradiction_id: str
    resolution_query: str
    evidence: str
    conclusion: str
    confidence: float
    
@dataclass
class ValidationResult:
    contradictions_found: List[Contradiction]
    resolutions_needed: List[str]
    validation_summary: str

@dataclass
class AgentExecutionResult:
    web_result: Optional[Any] = None
    arxiv_result: Optional[Any] = None
    multimodal_result: Optional[Any] = None
    execution_errors: List[str] = None

@dataclass
class CachedData:
    web_result: Optional[Dict[str, Any]]
    arxiv_result: Optional[Dict[str, Any]]
    multimodal_result: Optional[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    resolutions: List[Dict[str, Any]]
    executive_summary: str
    detailed_analysis: str

@dataclass
class ResearchReport:
    task_id: str
    query: str
    methodology: str
    web_insights: List[str]
    academic_insights: List[str]
    media_insights: List[str]
    contradictions_found: List[Contradiction]
    resolutions: List[Resolution]
    executive_summary: str
    detailed_analysis: str
    sources_analyzed: int
    timestamp: datetime
    used_cache: bool = False

class SynthesizeTask(BaseModel):
    EXECUTIVE_SUMMARY: str
    DETAILED_ANALYSIS : str
    
class OrchestratorAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        self.memory_cache = MemoryCacheLayer()
        self.validator = ResearchValidator()
        self.is_initialized = False
        
        logger.info("Orchestrator Agent initialized successfully")
        
    async def initialize(self):
        try:
            self.web_agent = await WebResearchAgent.create()
            self.arxiv_agent = await ArxivResearchAgent.create()
            self.multimodal_agent = await MultiModalResearchAgent.create()
            
            self.is_initialized = True
            logger.info("Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise RuntimeError(f"Initialization failed: {e}")
        
    async def research(self, query: str) -> ResearchReport:
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"Starting research for: {query}")
        
        try:
            similar_task_id = await self.memory_cache.find_similar_query(query)
            
            if similar_task_id:
                logger.info(f"Using cached data from task_id: {similar_task_id}")
                return await self._generate_report_from_cache(query, similar_task_id)
            
            else:
                logger.info("No similar query found - executing full research")
                return await self._execute_full_research(query)
            
            
            
        except Exception as e:
            logger.error(f"Error in research: {e}")
            return self._create_error_report(query, str(e))
        
    async def _generate_report_from_cache(self, query: str, task_id: str) -> ResearchReport:
        try:
            cached_data_dict = await self.memory_cache.retrieve_task_data(task_id)
            if not cached_data_dict:
                logger.warning("No cached data found, falling back to full research")
                return await self._execute_full_research(query)
            
            cached_data = CachedData(
                web_result=cached_data_dict["web_result"],
                arxiv_result=cached_data_dict["arxiv_result"],
                multimodal_result=cached_data_dict["multimodal_result"],
                contradictions=cached_data_dict.get("contradictions", []),
                resolutions=cached_data_dict.get("resolutions", []),
                executive_summary=cached_data_dict.get("executive_summary", ""),
                detailed_analysis=cached_data_dict.get("detailed_analysis", "")
            )
            
            if not cached_data.executive_summary:
                executive_summary, detailed_analysis = await self._synthesize_report(
                    query, cached_data.web_result, cached_data.arxiv_result, cached_data.multimodal_result, [Contradiction(**c) for c in cached_data.contradictions], [Resolution(**r) for r in cached_data.resolutions]
                )
            else:
                executive_summary = cached_data.executive_summary
                detailed_analysis = cached_data.detailed_analysis

            return ResearchReport(
                task_id=task_id,
                query=query,
                methodology="Cached Results (Similarity Match)",
                web_insights=self._extract_insights(cached_data.web_result, "web"),
                academic_insights=self._extract_insights(cached_data.arxiv_result, "arxiv"),
                media_insights=self._extract_insights(cached_data.multimodal_result, "media"),
                contradictions_found=[Contradiction(**c) for c in cached_data.contradictions],
                resolutions=[Resolution(**r) for r in cached_data.resolutions],
                executive_summary=executive_summary,
                detailed_analysis=detailed_analysis,
                sources_analyzed=self._count_sources_from_dict(cached_data.web_result, cached_data.arxiv_result, cached_data.multimodal_result),
                timestamp=datetime.now(),
                used_cache=True
            )
            
        except Exception as e:
            logger.error(f"Error generating cached report: {e}")
            return await self._execute_full_research(query)
        
    async def _execute_full_research(self, query: str) -> ResearchReport:
        task_id = str(uuid.uuid4())
        try:
            # Step 1: Execute all agents in parallel
            logger.info("Executing all agents in parallel")
            execution_result = await self._execute_agents(query)
            
            # Step 2: Detect contradictions
            logger.info("Detecting contradictions")
            contradictions = await self.validator.detect_contradictions({
                "web_result": execution_result.web_result,
                "arxiv_result": execution_result.arxiv_result,
                "multimodal_result": execution_result.multimodal_result
            })
            
            # Step 3: Resolve contradictions using web search
            resolutions = []
            if contradictions:
                logger.info(f"Resolving {len(contradictions)} contradictions")
                resolutions = await self._resolve_contradictions(contradictions)
            
            # Step 4: Generate final report
            executive_summary, detailed_analysis = await self._synthesize_report(
                query, execution_result.web_result, execution_result.arxiv_result, execution_result.multimodal_result, contradictions, resolutions
            )
            
            # Step 5: Store in cache
            await self._store_research_in_cache(query, task_id, CachedData(
                web_result = execution_result.web_result.__dict__ if execution_result.web_result else None,
                arxiv_result=execution_result.arxiv_result.__dict__ if execution_result.arxiv_result else None,
                multimodal_result=execution_result.multimodal_result.__dict__ if execution_result.multimodal_result else None,
                contradictions=[c.__dict__ for c in contradictions],
                resolutions=[r.__dict__ for r in resolutions],
                executive_summary=executive_summary,
                detailed_analysis=detailed_analysis
            ))
            
            return ResearchReport(
                task_id=task_id,
                query=query,
                methodology="Full Research Pipeline with Validation",
                web_insights=self._extract_insights(execution_result.web_result, "web"),
                academic_insights=self._extract_insights(execution_result.arxiv_result, "arxiv"),
                media_insights=self._extract_insights(execution_result.multimodal_result, "media"),
                contradictions_found=contradictions,
                resolutions=resolutions,
                executive_summary=executive_summary,
                detailed_analysis=detailed_analysis,
                sources_analyzed=self._count_sources(execution_result.web_result, execution_result.arxiv_result, execution_result.multimodal_result),
                timestamp=datetime.now(),
                used_cache=False
            )
            
        except Exception as e:
            logger.error(f"Error in full research: {e}")
            return self._create_error_report(query, str(e))
        
    async def _execute_agents(self, query: str) -> AgentExecutionResult:
        errors = []
        web_result = None
        arxiv_result = None
        media_result = None
        
        logger.info(" Starting Web Research Agent...")
        try:
            web_result = await self._safe_agent_execution("web", self.web_agent.research(query))
            logger.info(" Web agent completed successfully")
        except Exception as e:
            logger.error(f" Web agent failed: {e}")
            errors.append(f"web: {str(e)}")
        
        # Step 2: Execute ArXiv Agent
        logger.info(" Starting ArXiv Research Agent...")
        try:
            arxiv_result = await self._safe_agent_execution("arxiv", self.arxiv_agent.research(query))
            logger.info(" ArXiv agent completed successfully")
        except Exception as e:
            logger.error(f" ArXiv agent failed: {e}")
            errors.append(f"arxiv: {str(e)}")
        
        # Step 3: Execute Multimodal Agent
        logger.info("🎬 Starting Multimodal Research Agent...")
        try:
            media_result = await self._safe_agent_execution("multimodal", self.multimodal_agent.research(query))
            logger.info(" Multimodal agent completed successfully")
        except Exception as e:
            logger.error(f" Multimodal agent failed: {e}")
            errors.append(f"multimodal: {str(e)}")
            
        successful_agents = []
        if web_result: successful_agents.append("web")
        if arxiv_result: successful_agents.append("arxiv")  
        if media_result: successful_agents.append("multimodal")
        
        logger.info(f"Sequential execution completed: {len(successful_agents)}/3 agents successful")
        if successful_agents:
            logger.info(f"Successful agents: {', '.join(successful_agents)}")
        if errors:
            logger.warning(f"Failed agents: {len(errors)} errors")
            
        return AgentExecutionResult(
            web_result=web_result,
            arxiv_result=arxiv_result,
            multimodal_result=media_result,
            execution_errors=errors
        )
    
    async def _safe_agent_execution(self, agent_name: str, agent_task):
        try:
            result = await agent_task
            logger.info(f"{agent_name} agent completed successfully")
            return result
        except Exception as e:
            logger.error(f"{agent_name} agent failed: {e}")
            raise e
        
    async def _resolve_contradictions(self, contradictions: List[Contradiction]) -> List[Resolution]:
        resolutions = []
        
        for contradiction in contradictions:
            try:
                resolution_query = await self.validator.generate_resolution_query(contradiction)

                # Use web agent to search for resolution
                resolution_result = await self.web_agent.research(resolution_query)
                resolution = await self.validator.analyze_resolution(contradiction, resolution_result.summary)
                
                resolution.resolution_query = resolution_query
                resolutions.append(resolution)
                logger.info(f"Resolved contradiction: {contradiction.topic}")                
                
            except Exception as e:
                logger.error(f"Error resolving contradiction {contradiction.id}: {e}")
        
        return resolutions
    
    async def _synthesize_report(self, query: str, web_result, arxiv_result, media_result, contradictions: List[Contradiction], resolutions: List[Resolution]) -> Tuple[str, str]:
        try:
            prompt = f"""
                Create a research report for: "{query}"
                
                WEB FINDINGS: {self._extract_insights(web_result, "web")}
                ACADEMIC FINDINGS: {self._extract_insights(arxiv_result, "arxiv")}
                MEDIA FINDINGS: {self._extract_insights(media_result, "media")}
                
                CONTRADICTIONS: {len(contradictions)} found
                RESOLUTIONS: {len(resolutions)} resolved
                
                Generate:
                1. EXECUTIVE_SUMMARY: 2-3 sentences directly answering the query
                2. DETAILED_ANALYSIS: 400-500 words comprehensive analysis
                
                Format:
                EXECUTIVE_SUMMARY:
                [summary]
                
                DETAILED_ANALYSIS:
                [analysis]
            """
            response = self.client.responses.parse(
                model=OPENAI_CONFIG["default_model"],
                input=[
                    {"role": "system", "content": "Generate comprehensive research reports."},
                    {"role": "user", "content": prompt}
                ],
                text_format=SynthesizeTask,
                temperature=0.3
            )
            
            content = json.loads(response.output[0].content[0].text)
            
            executive_summary = content["EXECUTIVE_SUMMARY"]
            detailed_analysis = content["DETAILED_ANALYSIS"]
            return executive_summary, detailed_analysis
            
        except Exception as e:
            logger.error(f"Error synthesizing report: {e}")
            return "Research completed.", f"Analysis of '{query}' completed."
        
    async def _store_research_in_cache(self, query: str, task_id: str, cached_data: CachedData):
        try:
            # Store query embedding with task_id in chroma
            await self.memory_cache.store_query_with_task_id(query, task_id)
            
            # Store all research data under task_id in redis
            await self.memory_cache.store_task_data(task_id, cached_data.__dict__)
            
            logger.info(f"Stored research in cache with task_id: {task_id}")
        
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            
    def _extract_insights(self, result, source_type: str) -> List[str]:
        if not result:
            return []
        
        if source_type == "web":
            if hasattr(result, 'key_findings'):
                return result.key_findings or []
            
            elif isinstance(result, dict):
                return result.get('key_findings', [])
        
        elif source_type == "arxiv":
            insights = []
            
            if hasattr(result, 'topic_results'):
                for topic_result in result.topic_results:
                    if hasattr(topic_result, 'key_insights'):
                        insights.extend(topic_result.key_insights)

            elif isinstance(result, dict):
                topic_results = result.get('topic_results', [])
                for topic_result in topic_results:
                    if isinstance(topic_result, dict):
                        insights.extend(topic_result.get('key_insights', []))
                    elif hasattr(topic_result, 'key_insights'):
                        insights.extend(topic_result.key_insights or [])
            
            return insights   
            
        elif source_type == "media":
            if hasattr(result, 'key_insights'):
                return result.key_insights or []
            elif isinstance(result, dict):
                return result.get('key_insights', [])
            
        return []
    
    def _count_sources(self, web_result, arxiv_result, media_result) -> int:
        """
            Count total sources analyzed.
        """
        count = 0
        
        if web_result and hasattr(web_result, 'sources_analyzed'):
            count += web_result.sources_analyzed
        if arxiv_result and hasattr(arxiv_result, 'total_papers_analyzed'):
            count += arxiv_result.total_papers_analyzed
        if media_result and hasattr(media_result, 'files_processed'):
            count += media_result.files_processed
        
        return count
    
    def _count_sources_from_dict(self, web_dict, arxiv_dict, media_dict) -> int:
        """
            Count total sources from cached dictionary data.
        """
        count = 0
        
        if web_dict:
            count += web_dict.get('sources_analyzed', 0)
        if arxiv_dict:
            count += arxiv_dict.get('total_papers_analyzed', 0)
        if media_dict:
            count += media_dict.get('files_processed', 0)
        
        return count  
    
    def _create_error_report(self, query: str, error: str) -> ResearchReport:
        """
            Create error report.
        """
        
        return ResearchReport(
            task_id=str(uuid.uuid4()),
            query=query,
            methodology="Failed Research",
            web_insights=[],
            academic_insights=[],
            media_insights=[],
            contradictions_found=[],
            resolutions=[],
            executive_summary="Research failed due to technical error.",
            detailed_analysis=f"Error: {error}",
            sources_analyzed=0,
            timestamp=datetime.now(),
            used_cache=False
        )
        
async def create_simple_orchestrator() -> OrchestratorAgent:
        
    orchestrator = OrchestratorAgent()
    await orchestrator.initialize()
    return orchestrator