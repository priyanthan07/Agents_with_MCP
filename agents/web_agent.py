import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum

from mcp_client.client import create_mcp_client
from util.logger import get_logger
from config import OPENAI_CONFIG, MCP_CONFIG

logger = get_logger(__name__)
    
class decisionTypes(str, Enum):
    Conclude = "CONCLUDE"
    Continue = "CONTINUE"

class decisionOutputFormat(BaseModel):
    decision : decisionTypes
    reasoning: str

@dataclass
class ReActStep:             # Represents one complete ReAct cycle
    iteration : int
    thought : str
    action : str
    action_params: Dict[str, Any]
    observation: str
    reflection: str
    timestamp: datetime = datetime.now()

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
    react_trace: List[ReActStep]
    metadata: Dict[str, Any]
    
class WebResearchAgent:
    """
        Web Research Agent that performs ReAct (Reasoning + Acting) loops to search, analyze, and synthesize web content for research queries.
    """
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        
        self.max_iterations = 3                 # Maximum ReAct loop iterations
        self.is_initialized = False
        
        self.available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information. Use when you need to find new sources or explore a topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {"type": "integer", "description": "Number of results (1-20)", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_webpage",
                    "description": "Analyze a webpage for detailed content. Use when you want to extract information from a specific URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to analyze"},
                            "extract_text": {"type": "boolean", "default": True},
                            "summarize": {"type": "boolean", "default": True}
                        },
                        "required": ["url"]
                    }
                }
            }
        ]

        logger.info("Web Research Agent initialized successfully")
        
    @classmethod
    async def create(cls) -> 'WebResearchAgent':
        agent = cls()
        await agent._initialize_mcp_connection()
        agent.is_initialized = True
        return agent
    
    async def _initialize_mcp_connection(self):
        try:
            self.mcp_client = create_mcp_client()
            await self.mcp_client._initialize_client(server="web_research")
            
            server_tools = [tool["name"] for tool in self.mcp_client.available_tools]
            expected_tools = [tool["function"]["name"] for tool in self.available_tools]
            
            for expected_tool in expected_tools:
                if expected_tool not in server_tools:
                    logger.warning(f"Expected tool '{expected_tool}' not available from MCP server")
            
            logger.info(f"MCP connection established. Available tools in server : {server_tools}")
        
        except Exception as e:
            logger.error(f"Failed to connect with web mcp client: {e}")
            raise RuntimeError(e)
    
    async def research(self, task_query: str, context: List[Dict] = None) -> WebResearchResult:
        logger.info(f"Starting web research for query: {task_query}")
        
        research_state = {
            "original_query": task_query,
            "context": context or [],
            "search_results": [],
            "analyzed_sources": [],
            "key_findings": [],
            "iteration": 0,
            "research_complete": False,
            "react_steps": []
        }
        
        while (research_state["iteration"] < self.max_iterations) and not research_state["research_complete"]:
            
            research_state["iteration"] += 1
            logger.info(f"ReAct iteration {research_state['iteration']}")
            
            # Execute one complete ReAct cycle
            react_step = await self._execute_react_cycle(research_state)
            research_state["react_steps"].append(react_step)
            
            if react_step.reflection == "CONCLUDE":
                research_state["research_complete"] = True
                logger.info("Agent decided to conclude ReAct loop and research")
                
        # Synthesize final results
        final_result = await self._synthesize_results(research_state)
        logger.info(f"Web research completed with {len(final_result.search_results)} sources")
        
        return final_result
    
    async def _execute_react_cycle(self, research_state: Dict) -> ReActStep:
        """
            Execute one complete ReAct cycle:
            1. THOUGHT: Reason about current state and what to do next
            2. ACTION: Take a specific action based on the thought
            3. OBSERVATION: Process and understand the action results
            4. REFLECTION: Evaluate progress and plan next steps
        """
        
        # THOUGHT: Analyze current state and plan next action
        thought = await self._generate_thought(research_state)
        logger.info(f"Thought completed: {thought}")
        
        # ACTION: Execute the planned action
        action_result = await self._execute_action(thought, research_state)
        logger.info(f"ACTION: {action_result['action_name']} with params {action_result['params']}")
        
        # OBSERVATION: Process and understand the action results
        observation = await self._generate_observation(action_result, research_state)
        logger.info(f"Observation: {observation}")
        
        # REFLECTION: Evaluate progress and decide next steps
        reflection = await self._generate_reflection(thought, action_result, observation, research_state)
        logger.info(f"REFLECTION: {reflection}")
        
        # Create ReAct step record
        react_step = ReActStep(
            iteration=research_state["iteration"],
            thought=thought,
            action=f"{action_result['action_name']}({action_result['params']})",
            action_params=action_result['params'],
            observation=observation,
            reflection=reflection
        )
        return react_step
    
    async def _generate_thought(self, research_state: Dict) -> str:
        
        state_summary = self._build_state_summary(research_state)
        
        thought_prompt = f"""
            You are in the THOUGHT phase of ReAct methodology. Analyze your current research state and reason about what to do next.
            
            Original Query: {research_state['original_query']}
            
            current research state: {state_summary}
            
            Your task in this THOUGHT phase:
            1. Analyze what information you currently have
            2. Identify what information is still missing or unclear
            3. Determine the most logical next step to advance your research
            
            Think step-by-step about your reasoning. What should you do next and why?
            Respond with your thought process in 2-3 sentences that clearly explain your reasoning.
            make sure the current year is 2025. 
        """
        try:
            
            response = self.client.chat.completions.create(
                model = OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert researcher in the THOUGHT phase of ReAct. Provide clear, logical reasoning about what to do next."},
                    {"role": "user", "content": thought_prompt}
                ],
                temperature=0.1
            )

            thought = response.choices[0].message.content
            return thought
        
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return f"I need to gather more information about {research_state['original_query']} to make progress."
            
    async def _execute_action(self, thought: str, research_state: Dict) -> Dict[str, Any]:
        """
            ACTION phase: Execute a specific action based on the thought.
            The agent chooses and executes a tool based on its reasoning.
        """
        
        action_prompt = f"""
            You are in the ACTION phase of ReAct methodology. Based on your previous thought, choose and execute the most appropriate action.
            
            YOUR THOUGHT WAS: {thought}
            
            Available actions:
            1. web_search(query, num_results) - Search for information online
            2. analyze_webpage(url, extract_text, summarize) - Analyze a specific webpage
            
            Based on your thought, which action should you take? Choose the action that directly addresses your reasoning.
            
            You must call exactly one function based on your thought.
            make sure the current year is 2025. 
        """
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are in the ACTION phase. Choose and execute one tool based on your thought."},
                    {"role": "user", "content": action_prompt}
                ],
                tools=self.available_tools,
                tool_choice="required", 
                temperature=0.1
            )
            
            tool_call = response.choices[0].message.tool_calls[0]
            action_name  = tool_call.function.name
            action_params = json.loads(tool_call.function.arguments)
            
            if action_name == "web_search":
                result = await self._execute_web_search(action_params, research_state)
            
            elif action_name == "analyze_webpage":
                result = await self._execute_webpage_analysis(action_params, research_state)
            
            else:
                result = {"error": f"Unknown action: {action_name}"}
                
            return {
                "action_name": action_name,
                "params": action_params,
                "result": result,
                "success": result.get("success", True)
            }
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return {
                "action_name": "error",
                "params": {},
                "result": {"error": str(e)},
                "success": False
            }
            
    async def _generate_observation(self, action_result: Dict, research_state: Dict) -> str:
        observation_prompt = f"""
            You are in the OBSERVATION phase of ReAct methodology. Analyze the results of your recent action and understand what they mean for your research.
            
            ACTION TAKEN: {action_result['action_name']} with parameters {action_result['params']}
            ACTION RESULTS: {json.dumps(action_result['result'], indent=2)}
            SUCCESS: {action_result['success']}
            
            Your task in this OBSERVATION phase:
            1. Interpret what these results tell you about your research query
            2. Identify key information or patterns in the results
            3. Note any problems or limitations with the results
            4. Consider how these results relate to information you already have
            
            Provide a clear observation about what you learned from this action. Focus on the meaning and implications, not just repeating the raw results.
            
            Respond in 2-3 sentences that capture the key insights from this action.
            make sure the current year is 2025. 
        """
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are in the OBSERVATION phase. Interpret the action results and explain what you learned."},
                    {"role": "user", "content": observation_prompt}
                ],
                temperature=0.2
            )
            
            observation = response.choices[0].message.content.strip()
            return observation
            
        except Exception as e:
            logger.error(f"Error generating observation: {e}")
            return f"Action completed but encountered an error: {str(e)}"
    
    def _build_state_summary(self, research_state: Dict) -> str:
        summary_parts = []
        
        summary_parts.append(f"Iteration: {research_state['iteration']}/{self.max_iterations}")
        summary_parts.append(f"Search results found: {len(research_state['search_results'])}")
        summary_parts.append(f"Sources analyzed: {len(research_state['analyzed_sources'])}")
        summary_parts.append(f"Key findings: {len(research_state['key_findings'])}")
        
        if research_state['key_findings']:
            summary_parts.append("Recent findings:")
            for finding in research_state['key_findings'][-3:]:  # Last 3 findings
                summary_parts.append(f"  - {finding}")
        
        if research_state['react_steps']:
            last_step = research_state['react_steps'][-1]
            summary_parts.append(f"Last action: {last_step.action}")
            summary_parts.append(f"Last observation: {last_step.observation}")
        
        return "\n".join(summary_parts)
    
    async def _generate_reflection(self, thought: str, action_result: Dict, observation: str, research_state: Dict) -> str:
        state_summary = self._build_state_summary(research_state)
        
        reflection_prompt = f"""
            You are in the REFLECTION phase of ReAct methodology. Step back and evaluate your overall research progress.
            
            RESEARCH QUERY: {research_state['original_query']}
            
            THIS CYCLE:
            - THOUGHT: {thought}
            - ACTION: {action_result['action_name']}({action_result['params']})
            - OBSERVATION: {observation}
            
            OVERALL RESEARCH STATE:
            {state_summary}
            
            Your task in this REFLECTION phase:
            1. Evaluate how well this cycle advanced your research goals
            2. Assess whether you have sufficient information to answer the query
            3. Identify what aspects of the research still need attention
            4. Decide whether to continue research or conclude
            
            Consider: Do you have enough comprehensive information to provide a thorough answer to the research query? 
            
            Provide your reflection on progress and whether to continue or conclude research. Be specific about what you've accomplished and what might still be needed.
            
            End your reflection with either "CONTINUE" or "CONCLUDE" based on your assessment.
        """
        try:
            response = self.client.responses.parse(
                model=OPENAI_CONFIG["default_model"],
                input=[
                    {"role": "system", "content": "You are in the REFLECTION phase. Evaluate progress and decide whether to continue or conclude."},
                    {"role": "user", "content": reflection_prompt}
                ],
                text_format=decisionOutputFormat,
                temperature=0.3
            )
            
            reflection = json.loads(response.output[0].content[0].text)
            return reflection["decision"]
            
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            return "Need to continue research to gather more information. CONTINUE research"
        
    async def _execute_web_search(self, args, research_state: Dict) -> Dict[str, Any]:
        try:
           
            search_response = await self.mcp_client.call_tool("web_search",args)
            
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
                    "message": f"Found {len(new_results)} search results",
                    "results": [{"url": r.url, "title": r.title, "snippet": r.snippet} for r in new_results],
                    "success": True
                }
            else:
                return {"success": False, "error": search_response.get("error", "Search failed")}
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return {"success": False, "error": str(e)}
            
    async def _execute_webpage_analysis(self, args, research_state: Dict) -> Dict[str, Any]:

        try:
            analysis_response = await self.mcp_client.call_tool("analyze_webpage", args)
            
            if analysis_response.get("success"):
                content = analysis_response.get("content", "")
                summary = analysis_response.get("summary", "")
                url = args["url"]
                
                # Store analyzed source
                analysis_result = {
                    "url": url,
                    "title": analysis_response.get("title", ""),
                    "content": content,
                    "summary": summary,
                    "word_count": analysis_response.get("word_count", 0)
                }
                research_state["analyzed_sources"].append(analysis_result)
                
                # Extract key findings
                key_findings = await self._extract_key_findings(content, research_state["original_query"])
                research_state["key_findings"].extend(key_findings)
                
                return {
                    "success": True,
                    "title": analysis_response.get("title", ""),
                    "summary": summary,
                    "word_count": analysis_response.get("word_count", 0),
                    "key_findings": key_findings
                }
            else:
                return {"success": False, "error": analysis_response.get("error", "Analysis failed")}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
            
        
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
            
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert at extracting key research findings."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.3
            )
            
            findings_text = response.choices[0].message.content.strip()
            findings = [f.strip() for f in findings_text.split('\n') if f.strip()]
            
            return findings[:5]  # Limit to 5 findings per source
            
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return []
    
    async def _synthesize_results(self, research_state: Dict) -> WebResearchResult:

        try:
            react_summary = []
            for step in research_state["react_steps"]:
                react_summary.append(f"Iteration {step.iteration}:")
                react_summary.append(f"  THOUGHT: {step.thought}")
                react_summary.append(f"  ACTION: {step.action}")
                react_summary.append(f"  OBSERVATION: {step.observation}")
                react_summary.append(f"  REFLECTION: {step.reflection}")
                react_summary.append("")
            
            synthesis_prompt = f"""
                Create a comprehensive research summary based on the ReAct research process:
                
                Original Query: {research_state['original_query']}
                REACT REASONING TRACE: {chr(10).join(react_summary)}
                
                Key Findings discovered:
                {chr(10).join(research_state['key_findings'])}
                
                Create a well-structured summary that:
                1. Directly answers the research question
                2. Synthesizes information from multiple sources
                3. Acknowledges any limitations or contradictions
                4. Provides actionable insights where possible
                
                Keep the summary comprehensive but concise (300-500 words).
            """
            
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "Create a comprehensive research summary showing how ReAct methodology produced thorough results."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.4
            )
            
            summary = response.choices[0].message.content.strip()
            
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
                react_trace=research_state["react_steps"],
                metadata={
                    "iterations_completed": research_state["iteration"],
                    "total_sources_found": len(research_state["search_results"]),
                    "react_cycles": len(research_state["react_steps"]),
                    "research_completed_at": datetime.now().isoformat(),
                    "methodology": "ReAct (Reasoning and Acting)"
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error synthesizing ReAct results: {e}")
            # Return a basic result even if synthesis fails
            return WebResearchResult(
                query=research_state["original_query"],
                search_results=[],
                summary=f"ReAct research completed with {len(research_state['analyzed_sources'])} sources.",
                key_findings=research_state["key_findings"],
                sources_analyzed=len(research_state["analyzed_sources"]),
                research_depth="moderate",
                react_trace=research_state["react_steps"],
                metadata={"error": str(e)}
            )
            
    def _determine_research_depth(self, research_state: Dict) -> str:
        cycles = len(research_state["react_steps"])
        sources_count = len(research_state["analyzed_sources"])
        findings_count = len(research_state["key_findings"])
        
        if cycles >= 4 and sources_count >= 5 and findings_count >= 10:
            return "DEEP"
        elif cycles >= 2 and sources_count >= 3 and findings_count >= 5:
            return "MODERATE"
        else:
            return "SURFACE"