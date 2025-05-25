import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import openai
from openai import OpenAI
from pydantic import BaseModel

from memory_cache import MemoryCacheLayer
from util.logger import get_logger
from config import OPENAI_CONFIG, MCP_CONFIG

logger = get_logger(__name__)

class decomposedTask(BaseModel):
    agent: str
    specific_query : str
    priority : int
    
class tasksOutputFormat(BaseModel):
    tasks : List[decomposedTask]
    validation_requirements : List[str]
    
class validationTask(BaseModel):
    agent: str
    query : str
    focus : int

@dataclass
class ResearchTask:
    task_id: str
    agent_type: str
    query: str
    priority: int
    status: str = "PENDING"     # PENDING, IN_PROGRESS, COMPLETED, FAILED
    created_at: datetime = datetime.now()
    result: Optional[Dict] = None
    confidence_score: float = 0.0
    
@dataclass
class ResearchPlan:
    plan_id: str
    original_query: str
    decomposed_tasks: List[ResearchTask]
    validation_requirements: List[str]
    estimated_time: int
    created_at: datetime = datetime.now()
    
class OrchestratorAgent:
    def __init__(self, memory_cache: MemoryCacheLayer, validation_engine):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        self.memory_cache = memory_cache
        self.validation_engine = validation_engine
        self.mcp_server_url = MCP_CONFIG["mcp_server_url"]
        
        self.active_plans = Dict[str, ResearchPlan] = {}
        self.agent_status : Dict[str, str] = {
            "arxiv_agent": "IDLE",
            "web_agent": "IDLE", 
            "document_agent": "IDLE",
            "media_agent": "IDLE"
        }
        
        self.agent_capabilities = {
            "arxiv_agent": ["Searches academic papers", "analyzes citations", "finds research methodology"],
            "web_agent": ["Searches current web content", "news", "industry reports", "real-time information"],
            "document_agent": ["Analyzes local documents", "PDFs", "internal reports"],
            "media_agent": ["Processes videos", "images", "presentations for visual information"]
        }
        
    async def process_research_query(self, query: str, user_context: Dict[str, Any] = None) -> str:
        logger.info(f"Processing new research query: {query}")
        
        cached_results = await self.memory_cache.check_similar_queries(query)
        research_plan = await self._decompose_query(query, cached_results, user_context)
        
        self.active_plans[research_plan.plan_id] = research_plan
        
        # parallel agent execution
        await self._execute_research_plan(research_plan.plan_id)
        
        return research_plan.plan_id
    
    async def _decompose_query(self, query: str, cached_results: List[Dict], user_context: Dict[str, Any] = None) -> ResearchPlan:
        """
            Use LLM to decompose the query into specific tasks
        """
        
        prompt = f"""
            You are a research coordinator tasked with breaking down a complex research query into specific tasks for different specialized agents.
            Research Query: {query}
            
            Availabe agents and it's capabilities : {self.agent_capabilities}
            
            Cached Information Available: {json.dumps(cached_results, indent=2) if cached_results else None}
            
            Please decompose this query into specific, actionable tasks for each relevant agent. For each task, specify:
            1. Which agent should handle it
            2. Specific search terms or focus areas
            3. Priority level (1-5, where 5 is highest)
            
            Also mention the What type of validation might be needed for the retrived data from different sources. For example ['cross_source_verification', 'temporal_consistency']
            
            Return your response in given JSON structure with tasks and validation requirements.
        """
        
        try:
            response = await self.client.responses.parse(
                model = OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert research coordinator.Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                text_format=tasksOutputFormat,
                temperature=0.3
            )
            decomposition_result = json.loads(response.output_parsed)
            
        except Exception as e:
            logger.error(f"Error in query decomposition: {e}")
            decomposition_result = self._fallback_decomposition(query)
         
        # convert LLM responses into ResearchTask objects 
        tasks = []
        for idx, task_data in enumerate(decomposition_result["tasks"]):
            task = ResearchTask(
                task_id= f"task_{idx}_{datetime.now().timestamp()}",
                agent_type= task_data["agent"],
                query= task_data["specific_query"],
                priority=task_data["priority"]
            )
            tasks.append(task)
            
        # create the research plan
        plan_id = f"plan_{datetime.now().timestamp()}"
        research_plan = ResearchPlan(
            plan_id=plan_id,
            original_query=query,
            decomposed_tasks=tasks,
            validation_requirements=decomposition_result["validation_requirements"],
            estimated_time=len(tasks) * 30 # seconds
        )
        
        logger.info(f"Created research plan {plan_id} with {len(tasks)} tasks")
        return research_plan
    
    def _fallback_decomposition(self, query: str) -> Dict[str, Any]:
        return {
            "tasks": [
                {"agent": "arxiv_agent", "specific_query": f"academic research on {query}", "priority": 4},
                {"agent": "web_agent", "specific_query": f"current information about {query}", "priority": 3},
                {"agent": "document_agent", "specific_query": f"local documents related to {query}", "priority": 2},
                {"agent": "media_agent", "specific_query": f"visual content about {query}", "priority": 1}
            ],
            "validation_requirements": ["cross_source_verification", "temporal_consistency"]
        }
    
    async def _execute_research_plan(self, plan_id: str):
        """
            Execute the research plan by coordinating multiple agents in parallel
        """
        
        plan = self.active_plans[plan_id]
        
        # group tasks by agents for parallel execution
        agent_tasks = {}
        for task in plan.decomposed_tasks:
            if task.agent_type not in agent_tasks:
                agent_tasks[task.agent_type] = []
            agent_tasks[task.agent_type].append(task)
            
        # start parallel execution for each agent type
        agent_coroutines = []
        for agent_type, tasks in agent_tasks.items():
            coroutine = self._execute_agent_tasks(agent_type, tasks, plan_id)
            agent_coroutines.append(coroutine)
            
        # Wait for all agents to complete their initial tasks
        await asyncio.gather(*agent_coroutines)
        
        # Check for contradictions and trigger validation if needed
        await self._check_and_validate_results(plan_id)
        
        # generate final report
        await self._generate_final_report(plan_id)
        
    async def _execute_agent_tasks(self, agent_type: str, tasks: List[ResearchTask], plan_id: str):
        """
            Execute tasks for a specific agent type
            This method communicates with individual agents through the MCP server
            and manages their ReAct loops.
        """
        logger.info(f"Starting {agent_type} with {len(tasks)} tasks")
        self.agent_status[agent_type] = "ACTIVE"
        
        for task in tasks:
            try:
                task.status = "IN_PROGRESS"
                
                # Send task to agent via MCP server
                agent_result = await self._send_task_to_agent(agent_type, task)
                
                # store result in memory cache
                await self.memory_cache.store_agent_result(
                    agent_type, task.query, agent_result, plan_id
                )
                
                task.result = agent_result
                task.status = "COMPLETED"
                task.confidence_score = agent_result.get("confidence", 0.7)
                logger.info(f"Completed task {task.task_id} for {agent_type}")
            
            except Exception as e:
                logger.error(f"Error executing task {task.task_id}: {e}")
                task.status = "FAILED"
                task.result = {"error": str(e)}
                
        self.agent_status[agent_type] = "IDLE"
        logger.info(f"Finished all tasks for {agent_type}")
        
    async def _send_task_to_agent(self, agent_type: str, task: ResearchTask) -> Dict[str, Any]:
        task_payload = {
            "task_id" : task.task_id,
            "query" : task.query,
            "priority" : task.priority,
            "agent_type" : agent_type,
            "context" : await self.memory_cache.get_relevant_context(task.query)
        }
        
        # call to MCP server
        
        
    async def _check_and_validate_results(self, plan_id: str):
        """
            This method uses the validation engine to detect conflicts and decides whether additional research iterations are needed.
        """
        plan = self.active_plans[plan_id]
        
        completed_results = []
        for task in plan.decomposed_tasks:
            if task.status == "COMPLETED" and task.result:
                completed_results.append({
                    "agent_type": task.agent_type,
                    "task_id": task.task_id,
                    "query": task.query,
                    "result": task.result,
                    "confidence": task.confidence_score
                })
        
        contradictions = await self.validation_engine.detect_contradictions(completed_results)
        
        if contradictions:
            logger.info(f"Found {len(contradictions)} contradictions, triggering validation")
            
            # create validation tasks for each contradictions
            validation_tasks = []
            for contradiction in contradictions:
                validation_task = await self._create_validation_task(contradiction, plan_id)
                validation_tasks.append(validation_task)
                
            # Execute validation tasks
            for validation_task in validation_tasks:
                await self._execute_validation_task(validation_task, plan_id)
                
    async def _create_validation_task(self, contradiction: Dict, plan_id: str) -> ResearchTask:
        """
           This method analyzes the contradiction and determines what additional research is needed to resolve it.
        """
        
        # use llm to determine the best validation approach
        validation_prompt = f"""
            A contradiction has been detected in research results:
            
            Contradiction Details: {json.dumps(contradiction, indent=2)}
            
            Availabe agents and it's capabilities : {self.agent_capabilities}
            
            Please determine:
            1. Which agent should handle the validation research
            2. What specific query should be used for validation
            3. What type of sources would be most authoritative for resolving this contradiction

            Return a JSON response with the validation strategy.
        """
        
        try:
            response = await self.client.responses.parse(
                model = OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert at resolving research contradictions. Always respond with valid JSON."},
                    {"role": "user", "content": validation_prompt}
                ],
                text_format=validationTask,
                temperature=0.1
            )
            validation_strategy = json.loads(response.output_parsed)
            
        except Exception as e:
            logger.error(f"Error creating validation strategy : {e}")
            validation_strategy = {
                "agent": "web_agent",  # Default to web search for validation
                "query": f"validation research for: {contradiction.get('topic', 'unknown')}",
                "focus": "authoritative sources"
            }
            
        validation_task = ResearchTask(
            task_id=f"validation_{datetime.now().timestamp()}",
            agent_type = validation_strategy["agent"],
            query = validation_strategy["query"],
            priority=5
        )
        
        return validation_task
    
    async def _execute_validation_task(self, validation_task: ResearchTask, plan_id: str):
        logger.info(f"Executing validation task: {validation_task.task_id}")
        
        await self._execute_agent_tasks(validation_task.agent_type, [validation_task], plan_id)
        
        # update the validation engine with new evidence
        await self.validation_engine.update_with_validation_result(validation_task.result, validation_task.task_id)
        
        # add validation task to the plan
        plan = self.active_plans[plan_id]
        plan.decomposed_tasks.append(validation_task)
        
    async def _generate_final_report(self, plan_id: str):
        
        plan = self.active_plans[plan_id]
        
        all_results = []
        for task in plan.decomposed_tasks:
            if task.status == "COMPLETED" and task.result:
                all_results.append(task.result)
                
        resolved_contradictions = await self.validation_engine.get_resolved_contradictions()
        
        report_prompt = f"""
            Generate a comprehensive research report for the query: "{plan.original_query}"
            
            Research Results: {json.dumps(all_results, indent=2)}
            Resolved Contradictions: {json.dumps(resolved_contradictions, indent=2)}
            
            Please create a well-structured report that:
            1. Provides a clear answer to the original research question
            2. Explains any contradictions found and how they were resolved
            3. Cites specific sources and evidence
            4. Acknowledges limitations and areas for further research
            5. Uses appropriate academic formatting
            
            The report should be comprehensive but accessible.
            
        """
        
        try:
            response = await self.client.responses.parse(
                model = OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert research writer who creates comprehensive, well-structured reports."},
                    {"role": "user", "content": report_prompt}
                ],
                text_format=validationTask,
                temperature=0.5
            )
            final_report  = response.output_text
            
        except Exception as e:
            logger.error(f"Error creating validation strategy : {e}")
            final_report = "Error generating report. Please check logs for details."
            
        await self.memory_cache.store_final_report(plan_id, final_report, plan.original_query)
        
        logger.info(f"Generated final report for plan {plan_id}")
        return final_report
                
    async def get_research_status(self, plan_id: str) -> Dict[str, Any]:
        if plan_id not in self.active_plans:
            return {"error": "plan not found"}
        
        plan = self.active_plans[plan_id]
        
        status_summary = {
            "plan_id": plan_id,
            "original_query": plan.original_query,
            "total_tasks": len(plan.decomposed_tasks),
            "completed_tasks": len([task for task in plan.decomposed_tasks if task.status == "completed"]),
            "failed_tasks": len([task for task in plan.decomposed_tasks if task.status == "failed"]),
            "agent_status": self.agent_status.copy(),
            "estimated_completion": plan.estimated_completion_time,
            "created_at": plan.created_at.isoformat()
        }
        return status_summary
    