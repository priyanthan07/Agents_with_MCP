import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel

from mcp_client.client import create_mcp_client
from util.logger import get_logger
from config import OPENAI_CONFIG

logger = get_logger(__name__)

class TopicList(BaseModel):
    topics: List[str]
    reason: str
    
@dataclass
class TopicResult:
    topic: str
    search_results: Dict[str, Any]
    paper_details: List[Dict[str, Any]]
    key_insights: List[str]
    total_papers_found: int
    
@dataclass
class GlobalResearchResult:
    query: str
    generated_topics: List[str]
    topic_results: List[TopicResult]
    global_synthesis: str
    total_papers_analyzed: int
    
class ArxivResearchAgent:
    """
        Agentic ArXiv Research Agent that autonomously generates topic expansions
        for comprehensive research coverage
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])                
        self.is_initialized = False
        
        self.available_tools = ['search_papers', 'get_paper_details']
        
    @classmethod
    async def create(cls) -> 'ArxivResearchAgent':
        agent = cls()
        await agent._initialize_mcp_connection()
        agent.is_initialized = True
        return agent
    
    async def _initialize_mcp_connection(self):
        try:
            self.mcp_client = create_mcp_client()
            await self.mcp_client._initialize_client(server="arxiv_research") 
            
            server_tools = [tool["name"] for tool in self.mcp_client.available_tools]
            expected_tools = self.available_tools
            
            for expected_tool in expected_tools:
                if expected_tool not in server_tools:
                    logger.warning(f"Expected tool '{expected_tool}' not available from MCP server")
            
            logger.info(f"MCP connection established. Available tools in server : {server_tools}")
        
        except Exception as e:
            logger.error(f"Failed to connect with web mcp client: {e}")
            raise RuntimeError(e)
        
    async def research(self, task_query:str) -> GlobalResearchResult:
        """
            Main agentic research process:
            1. Agent autonomously generates 10 research topics
            2. Agent researches with comprehensive tool execution
            3. Agent synthesizes global understanding
        """
        
        logger.info(f"Starting ArXiv research for query: {task_query}")
        
        # Generate comprehensive topic expansion
        generated_topics = await self._generate_research_topics(task_query)
        logger.info(f"Agent generated {len(generated_topics)} research topics")
        
        topic_results = []
        for i, topic in enumerate(generated_topics, 1):
            logger.info(f"Executing research pipeline for topic {i}/{len(generated_topics)}: {topic}")
            
            topic_result = await self._execute_research_pipeline(topic, task_query)
            topic_results.append(topic_result)
            
            logger.info(f"Completed autonomous research for topic: {topic}")
            
        # Synthesize global understanding
        global_synthesis = await self._synthesize_global_understanding(task_query, generated_topics, topic_results)
        
        final_result = GlobalResearchResult(
            query=task_query,
            generated_topics=generated_topics,
            topic_results=topic_results,
            global_synthesis=global_synthesis,
            total_papers_analyzed= sum(result.total_papers_found  for result in topic_results)
        )
        
        logger.info(f"Global research completed: {len(generated_topics)} topics, {final_result.total_papers_analyzed} papers")
        return final_result
        
    async def _generate_research_topics(self, task_query) -> List[str]:
        
        try:
            topic_generation_prompt = f"""
                    You are an expert research agent tasked with comprehensive academic research.
                    
                    Given this research query: "{task_query}"
                    
                    Your agentic task is to autonomously identify 10 specific research topics that need to be 
                    explored to gain a complete, global understanding of this query. 
                    
                    Think like a research strategist - what are the different angles, sub-domains, 
                    methodologies, applications, and related areas that should be investigated?
                    
                    Generate 10 distinct, specific research topics that together will provide 
                    comprehensive coverage of the research landscape.
                    
                    Each topic should be:
                    1. Specific enough to yield focused search results
                    2. Different from other topics (no overlap)
                    3. Relevant to building global understanding
                    4. Searchable in academic databases
                    
                    Provide your reasoning for why these 10 topics give comprehensive coverage.
                """
            response = self.client.responses.parse(
                model=OPENAI_CONFIG["default_model"],
                input=[
                    {"role": "system", "content": "You are an autonomous research agent that makes intelligent decisions about research strategy."},
                    {"role": "user", "content": topic_generation_prompt}
                ],
                text_format=TopicList,
                temperature=0.3
            )
            
            result = json.loads(response.output[0].content[0].text)
            logger.info(f"Agent reasoning: {result['reason']}")    
            return result['topics']
            
        except Exception as e:
            logger.error(f"Error in topic generation: {e}") 
            keywords = task_query.split()
            return [f"{word} research applications" for word in keywords[:10]]
            
    async def _execute_research_pipeline(self, topic: str, original_query: str) -> TopicResult:
        """
            AGENTIC BEHAVIOR: Execute comprehensive research pipeline for each topic
            Agent makes autonomous decisions at each step based on previous results
        """
        logger.info(f"Starting research pipeline for topic: {topic}")
        try:
            
            topic_result = TopicResult(
                topic=topic,
                search_results={},
                paper_details=[],
                key_insights=[],
                total_papers_found=0
            )

            # STEP 1: Search Papers
            search_results = await self._execute_search_papers(topic)
            topic_result.search_results = search_results
            
            if (not search_results["success"]) or (not search_results["papers"]):
                logger.warning(f"No papers found for topic: {topic}")
                return topic_result
            
            papers = search_results["papers"]
            paper_ids = [paper["paper_id"] for paper in papers[:5]]
            topic_result.total_papers_found = len(papers)
            
            # STEP 2: Get Paper Details
            paper_details = await self._execute_get_paper_details(paper_ids)
            topic_result.paper_details = paper_details
            
            # STEP 6: Extract insights from all pipeline results
            topic_result.key_insights = await self._synthesize_topic_insights(topic, original_query, topic_result)    
            
            logger.info(f"Completed comprehensive pipeline for topic: {topic}")
            return topic_result    
        
        except Exception as e:
            logger.error(f"Error in autonomous topic research: {e}")
            return topic_result
    
        
    async def _execute_search_papers(self, topic: str) -> Dict[str, Any]:
        try:
            result = await self.mcp_client.call_tool("search_papers", {
                "query": topic,
                "max_results": 8
            })
            return result
        except Exception as e:
            logger.error(f"Error in search_papers: {e}")
            return {"success": False, "papers": []}
        
        
    async def _execute_get_paper_details(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        try:
            result = await self.mcp_client.call_tool("get_paper_details", {
                "paper_ids": paper_ids
            })
            return result.get("paper_details", []) if result.get("success") else []
        except Exception as e:
            logger.error(f"Error in get_paper_details: {e}")
            return []
    
    
    async def _synthesize_topic_insights(self, topic: str, original_query: str, topic_result: TopicResult) -> List[str]:
        try:
            topic_synthesis_prompt = f"""
                Synthesize key insights from comprehensive research on topic: "{topic}"
                (Part of broader research on: "{original_query}")
                
                Available data:
                - Search Results:{topic_result.search_results.get('papers', [])} and length {len(topic_result.search_results.get('papers', []))} papers found
                - Paper Details: {topic_result.paper_details} and {len(topic_result.paper_details)} papers analyzed in detail
                
                Extract 3-5 key insights that:
                1. Synthesize findings across all research tools
                2. Relate specifically to the topic "{topic}"
                3. Contribute to understanding the broader query
                4. Highlight important patterns, methods, or trends
                
                Return only the insights, one per line.
            """
            
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert at synthesizing research findings into comprehensive understanding."},
                    {"role": "user", "content": topic_synthesis_prompt}
                ],
                temperature=0.2
            )
            
            insights_text = response.choices[0].message.content.strip()
            insights = [insight.strip() for insight in insights_text.split('\n') if insight.strip()]
            return insights[:5]
            
        except Exception as e:
            logger.error(f"Error synthesizing topic insights: {e}")
            return [f"Comprehensive analysis completed for topic: {topic}"]
    
        
    async def _synthesize_global_understanding(self, original_query: str, topics: List[str], results: List[TopicResult]) -> str:
        """
        AGENTIC BEHAVIOR: Agent synthesizes all findings into global understanding
        """
        synthesis_prompt = f"""
            You are an expert research synthesizer conducting comprehensive academic analysis.
            
            Original research query: "{original_query}"
            
            You executed a comprehensive 5-step research pipeline for each of these {len(topics)} topics:
            {chr(10).join([f"- {topic}" for topic in topics])}
            
            For each topic, you executed:
            1. Paper Search → Paper Details
            
            Comprehensive findings summary:
            {chr(10).join([
                f"Topic: {r.topic}\\n"
                f"  Papers Found: {r.total_papers_found}\\n"
                f"  Key Insights: {r.key_insights}\\n"
                for r in results
            ])}
            
            Your agentic task is to synthesize ALL findings into a comprehensive, 
            global understanding of the original research query.
            
            Provide a synthesis that:
            1. Directly answers the original query with depth and nuance
            2. Integrates insights from all topics AND all research tools
            3. Identifies overarching patterns, methodologies, and trends
            4. Highlights key findings and their implications
            5. Notes research gaps and future directions
            6. Demonstrates the value of comprehensive analysis
            
            This should be a comprehensive research summary (500-700 words).
        """
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "You are an expert at synthesizing research findings into comprehensive understanding."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error in global synthesis: {e}")
            return f"Research completed across {len(topics)} topics with {sum(len(r.papers_found) for r in results)} papers analyzed."