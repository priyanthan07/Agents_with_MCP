import asyncio
import json
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import openai
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from urllib.parse import quote_plus

from utils.logger import get_logger
from config import OPENAI_CONFIG, MCP_CONFIG

logger = get_logger(__name__)

class actionTypes(str, Enum):
    search = "SEARCH"
    analyze = "ANALYZE"
    refine = "REFINE"
    conclude = "CONCLUDE"

class thoughtOutputFormat(BaseModel):
    action : actionTypes
    thought: str
    
class decisionTypes(str, Enum):
    Conclude = "CONCLUDE"
    Continue = "CONTINUE"

class decisionOutputFormat(BaseModel):
    decision : decisionTypes
    reason: str

@dataclass
class SearchResult:
    url : str
    title: str
    snippet: str
    content: str = ""
    source_type: str = "web"
    extracted_at: datetime = datetime.now()
    
@dataclass
class WebResearchResult:
    query: str
    search_results: List[SearchResult]
    summary: str
    key_findings: List[str]
    sources_analyzed: int
    research_depth: str  # "SURFACE", "MODERATE", "DEEP"
    metadata: Dict[str, Any]
    
class WebResearchAgent:
    """
        Web Research Agent that performs ReAct (Reasoning + Acting) loops to search, analyze, and synthesize web content for research queries.
    """
    def __init__(self, mcp_client):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        self.mcp_client = mcp_client  # MCP server client for tool access
        self.max_iterations = 3       # Maximum ReAct loop iterations
        self.max_sources = 10         # Maximum sources to analyze per query
        
        logger.info("Web Research Agent initialized successfully")
    
    async def research(self, task_query: str, context: List[Dict] = None) -> WebResearchResult:
        logger.info(f"Starting web research for query: {task_query}")
        
        research_state = {
            "original_query": task_query,
            "context": context or [],
            "search_results": [],
            "analyzed_sources": [],
            "key_findings": [],
            "iteration": 0,
            "research_complete": False
        }
        
        while (research_state["iteration"] < self.max_iterations) and not research_state["research_complete"]:
            
            research_state["iteration"] += 1
            logger.info(f"ReAct iteration {research_state['iteration']}")
            
            # THOUGHT: Analyze current state and plan next action
            thought_result = await self._generate_thought(research_state)
            logger.info(f"Thought completed: {thought_result}")
            
            # ACTION: Execute the planned action
            action_result = await self._execute_action(thought_result, research_state)
            logger.info(f"Action completed: {action_result['action_type']}")
            
            # OBSERVATION: Process and understand the action results
            observation = await self._process_observation(action_result, research_state)
            logger.info(f"Observation: {observation}")
            
            # REFLECTION: Decide if more research is needed
            research_state["research_complete"] = await self._should_conclude_research(research_state)
        
        # Synthesize final results
        final_result = await self._synthesize_results(research_state)
        logger.info(f"Web research completed with {len(final_result.search_results)} sources")
        
        return final_result
    
    async def _generate_thought(self, research_state: Dict) -> str:
        
        thought_prompt = f"""
            You are conducting web research. Analyze the current state and decide what to do next.
            
            Original Query: {research_state['original_query']}
            Current Iteration: {research_state['iteration']}/{self.max_iterations}
            Sources Found: {len(research_state['search_results'])}
            Sources Analyzed: {len(research_state['analyzed_sources'])}
            Key Findings So Far: {research_state['key_findings']}
            
            Available Actions:
            1. SEARCH - Find new web sources relevant to the query
            2. ANALYZE - Deep dive into a specific source for detailed information
            3. REFINE - Refine search terms based on current findings
            4. CONCLUDE - Synthesize findings and conclude research
            
            What should be the next logical step? Explain your reasoning and specify the action.
            Respond with your thought process and the action you want to take.
        """
        try:
            
            response = await self.client.responses.parse(
                model = OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert web researcher using ReAct methodology.Always respond with valid JSON."},
                    {"role": "user", "content": thought_prompt}
                ],
                text_format=thoughtOutputFormat,
                temperature=0.1
            )
            return json.loads(response.output_parsed)
        
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return {
                    "action": "SEARCH", 
                    "thought": "Need to find more sources about the query"
                }
            
    async def _execute_action(self, thought_result: str, research_state: Dict) -> Dict[str, Any]:
        action_type  = thought_result["action"]
        if action_type == "SEARCH":
            return await self._perform_web_search(research_state)
        
        elif action_type == "ANALYZE":
            return await self._analyze_source(research_state)
        
        elif action_type == "REFINE":
            return await self._refine_search(research_state)
        
        elif action_type == "CONCLUDE":
            return {"action_type": "CONCLUDE", "ready_to_conclude": True}
        
        else:
            return await self._perform_web_search(research_state)
        
    async def _perform_web_search(self, research_state: Dict) -> Dict[str, Any]:
        try:
            search_params = {
                "query": research_state["original_query"],
                "num_results": min(10, self.max_sources - len(research_state["search_results"])),
                "safe_search": "moderate"
            }
            
            search_response = await self.mcp_client.call_tool(
                "web_search",
                search_params
            )
            
            new_results = []
            if search_response.get("success") and search_response.get("results"):
                for result_data in search_response["results"]:
                    search_result = SearchResult(
                        url=result_data.get("url", ""),
                        title=result_data.get("title", ""),
                        snippet=result_data.get("snippet", ""),
                        source_type="web_search"
                    )
                    new_results.append(search_result)
                    
            research_state["search_results"].extend(new_results)
            return {
                "action_type": "SEARCH",
                "new_results_count": len(new_results),
                "total_results": len(research_state["search_results"]),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return {
                "action_type": "SEARCH",
                "error": str(e),
                "success": False
            }
            
    async def _analyze_source(self, research_state: Dict) -> Dict[str, Any]:
        """
            Analyze a specific web source in detail using MCP server tools
        """
        try:
            # Find an unanalyzed source
            analyzed_urls = [analyzed["url"] for analyzed in research_state["analyzed_sources"]]
            unanalyzed_sources = []
            for result in research_state["search_results"]:
                if result.url not in analyzed_urls:
                    unanalyzed_sources.append(result.url)
                    
            if not unanalyzed_sources:
                return {"action_type": "ANALYZE", "message": "No unanalyzed sources available"}
            
            target_source = unanalyzed_sources[0]
            
            analysis_params = {
                "url": target_source.url,
                "extract_text": True,
                "summarize": True
            }
            
            analysis_response = await self.mcp_client.call_tool(
                "webpage_analyzer",
                analysis_params
            )
            
            if analysis_response["success"]:
                content = analysis_response.get("content", "")
                summary = analysis_response.get("summary", "")
                
                target_source.content = content
            
                analysis_result = {
                        "url": target_source.url,
                        "title": target_source.title,
                        "content": content,
                        "summary": summary
                    }
                research_state["analyzed_sources"].append(analysis_result)
                key_findings = await self._extract_key_findings(
                        content, research_state["original_query"]
                    )
                research_state["key_findings"].extend(key_findings)
                
                return {
                        "action_type": "ANALYZE",
                        "analyzed_url": target_source.url,
                        "content_length": len(content),
                        "new_findings": len(key_findings),
                        "success": True
                    }   
            else:
                return {
                    "action_type": "ANALYZE",
                    "error": analysis_response.get("error", "Analysis failed"),
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Error analyzing source: {e}")
            return {
                "action_type": "ANALYZE",
                "error": str(e),
                "success": False
            }
            
    async def _refine_search(self, research_state: Dict) -> Dict[str, Any]:
        """
            Refine search terms based on current findings
        """
        try:
            refine_prompt = f"""
                Based on the current research findings, suggest refined search terms that would help find more specific and relevant information.
                
                Original Query: {research_state['original_query']}
                Current Findings: {research_state['key_findings']} 
                
                Suggest 2-3 more specific search terms that would complement the existing research.
                Return only the search terms, one per line.
            """
            
            response = await self.client.client.responses.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert at refining search queries for research."},
                    {"role": "user", "content": refine_prompt}
                ],
                temperature=0.3
            )

            refined_terms = response.output_text.strip().split('\n')

            total_new_results = 0
            for term in refined_terms[:2]: 
                if term.strip():
                    refined_state = research_state.copy()
                    refined_state["original_query"] = term.strip()
                    
                    search_result = await self._perform_web_search(refined_state)
                    total_new_results += search_result.get("new_results_count", 0)
                    
            return {
                "action_type": "REFINE",
                "refined_terms": refined_terms,
                "new_results": total_new_results,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error refining search: {e}")
            return {
                "action_type": "REFINE",
                "error": str(e),
                "success": False
            }
            
    async def _process_observation(self, action_result: Dict, research_state: Dict) -> str:
        """
            Process and understand the results of the action
            This is the "Observation" phase of ReAct
        """
        action_type = action_result.get("action_type", "unknown")
        
        if action_type == "SEARCH":
            if action_result["success"]:
                return f"Found {action_result.get('new_results_count', 0)} new sources. Total sources: {action_result.get('total_results', 0)}"
            else:
                return f"Search failed: {action_result.get('error', 'Unknown error')}"
            
        elif action_type == "ANALYZE":
            if action_result["success"]:
                return f"Analyzed {action_result.get('analyzed_url', 'unknown URL')}. Found {action_result.get('new_findings', 0)} key findings."
            else:
                return f"Analysis failed: {action_result.get('error', 'Unknown error')}"
            
        elif action_type == "REFINE":
            if action_result["success"]:
                return f"Refined search with {len(action_result.get('refined_terms', []))} new terms, found {action_result.get('new_results', 0)} additional sources."
            else:
                return f"Search refinement failed: {action_result.get('error', 'Unknown error')}"
            
        elif action_type == "CONCLUDE":
            return "Ready to conclude research and synthesize findings."
        
        else:
            return f"Completed {action_type} action"
        
    async def _should_conclude_research(self, research_state: Dict) -> bool:
        
        has_enough_sources = len(research_state["analyzed_sources"]) >= 3
        has_findings = len(research_state["key_findings"]) >= 5
        max_iterations_reached = research_state["iteration"] >= self.max_iterations
        
        reflection_prompt = f"""
            Evaluate whether the current research is sufficient to answer the original query.
            
            Original Query: {research_state['original_query']}
            Sources Analyzed: {len(research_state['analyzed_sources'])}
            Key Findings: {len(research_state['key_findings'])}
            Current Iteration: {research_state['iteration']}/{self.max_iterations}
            
            Sample Findings: {research_state['key_findings'][:3]}
            
            Should we conclude the research now? Consider:
            1. Quality and relevance of findings
            2. Coverage of the research question
            3. Confidence in the current evidence
            
            Respond with either "CONCLUDE" or "CONTINUE" and explain why.
        """
        try:
            
            response = await self.client.responses.parse(
                model = OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert research evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": reflection_prompt}
                ],
                text_format=decisionOutputFormat,
                temperature=0.1
            )
            decision_result = json.loads(response.output_parsed)
            should_conclude = (decision_result["decision"] == "CONCLUDE") or max_iterations_reached

            logger.info(f"Research conclusion decision: {decision_result["decision"]}")
            return should_conclude
            
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return has_enough_sources and has_findings
        
    async def _extract_key_findings(self, content: str, query: str) -> List[str]:
        try:
            content_sample = content[:2000] if len(content) > 2000 else content
            
            extraction_prompt = f"""
                Extract the most important findings from this content that relate to the research query.
                
                Research Query: {query}
                Content: {content_sample}
                
                Extract 3-5 key findings that directly answer or provide evidence for the research query.
                Each finding should be:
                1. Specific and factual
                2. Directly relevant to the query
                3. Supported by the content
                
                Format each finding as a complete sentence.
                Return only the findings, one per line.
            """
            
            response = await self.client.client.responses.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert at extracting key research findings."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.3
            )
            
            findings_text = response.output_text.strip()
            findings = [f.strip() for f in findings_text.split('\n') if f.strip()]
            
            return findings[:5]  # Limit to 5 findings per source
            
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return []
    
    async def _synthesize_results(self, research_state: Dict) -> WebResearchResult:

        try:
            synthesis_prompt = f"""
                Create a comprehensive research summary based on the collected findings.
                
                Original Query: {research_state['original_query']}
                Sources Analyzed: {len(research_state['analyzed_sources'])}
                
                Key Findings:
                {chr(10).join(research_state['key_findings'])}
                
                Create a well-structured summary that:
                1. Directly answers the research question
                2. Synthesizes information from multiple sources
                3. Acknowledges any limitations or contradictions
                4. Provides actionable insights where possible
                
                Keep the summary comprehensive but concise (300-500 words).
            """
            
            response = await self.client.client.responses.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert research synthesizer."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.4
            )
            
            summary = response.output_text.strip()
            
            # Determine research depth
            research_depth = self._determine_research_depth(research_state)
            
            # Create final result object
            result = WebResearchResult(
                query=research_state["original_query"],
                search_results=[
                    SearchResult(
                        url=source.get("url", ""),
                        title=source.get("title", ""),
                        snippet="", 
                        content=source.get("content", ""),
                    ) for source in research_state["analyzed_sources"]
                ],
                summary=summary,
                key_findings=research_state["key_findings"],
                sources_analyzed=len(research_state["analyzed_sources"]),
                research_depth=research_depth,
                metadata={
                    "iterations_completed": research_state["iteration"],
                    "total_sources_found": len(research_state["search_results"]),
                    "research_completed_at": datetime.now().isoformat(),
                    "agent_type": "web_research"
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            # Return a basic result even if synthesis fails
            return WebResearchResult(
                query=research_state["original_query"],
                search_results=[],
                summary=f"Research completed with {len(research_state['analyzed_sources'])} sources analyzed.",
                key_findings=research_state["key_findings"],
                sources_analyzed=len(research_state["analyzed_sources"]),
                research_depth="moderate",
                metadata={"error": str(e)}
            )
            
    def _determine_research_depth(self, research_state: Dict) -> str:
        sources_count = len(research_state["analyzed_sources"])
        findings_count = len(research_state["key_findings"])
        
        if sources_count >= 5 and findings_count >= 10:
            return "DEEP"
        elif sources_count >= 3 and findings_count >= 5:
            return "MODERATE"
        else:
            return "SURFACE"