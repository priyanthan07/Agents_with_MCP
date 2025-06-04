import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

# Import from src directory
from src.orchestrator import OrchestratorAgent, ResearchReport
from util.logger import get_logger

logger = get_logger(__name__)

class MultiAgentResearchSystem:
    """
        Main entry point for the multi-agent research system.
        This wraps the existing OrchestratorAgent with a simplified interface.
    """
    
    def __init__(self):
        self.orchestrator: Optional[OrchestratorAgent] = None
        logger.info("Multi-Agent Research System created")
    
    async def initialize(self):
        try:
            logger.info("Initializing Multi-Agent Research System...")
            
            # Create orchestrator with dependencies
            self.orchestrator = OrchestratorAgent()
            await self.orchestrator.initialize()
            
        except Exception as e:
            logger.error(f" Failed to initialize system: {e}")
            raise RuntimeError(f"System initialization failed: {e}")
    
    async def research(self, query: str) -> ResearchReport:        
        logger.info(f" Starting research for: '{query}'")
        
        try:
            # Execute research through orchestrator
            result = await self.orchestrator.research(query)
            
            logger.info(f" Research completed for: '{query}'")
            logger.info(f" Sources analyzed: {result.sources_analyzed}")
            logger.info(f" Insights generated: {len(result.web_insights) + len(result.academic_insights) + len(result.media_insights)}")
            logger.info(f" Contradictions found: {len(result.contradictions_found)}")
            logger.info(f" Contradictions resolved: {len(result.resolutions)}")
            
            return result
            
        except Exception as e:
            logger.error(f" Research failed: {e}")
            raise
    
    async def get_research_status(self) -> Dict[str, Any]:
        try:
            status = {
                "system_initialized": self.orchestrator.is_initialized,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "orchestrator": self.orchestrator is not None and self.orchestrator.is_initialized,
                    "memory_cache": self.orchestrator.memory_cache is not None if self.orchestrator else False,
                    "validator": self.orchestrator.validator is not None if self.orchestrator else False
                }
            }
            
            if self.orchestrator and self.orchestrator.is_initialized:
                status["agents"] = {
                    "web_agent": hasattr(self.orchestrator, 'web_agent') and self.orchestrator.web_agent is not None and self.orchestrator.web_agent.is_initialized,
                    "arxiv_agent": hasattr(self.orchestrator, 'arxiv_agent') and self.orchestrator.arxiv_agent is not None and self.orchestrator.arxiv_agent.is_initialized,
                    "multimodal_agent": hasattr(self.orchestrator, 'multimodal_agent') and self.orchestrator.multimodal_agent is not None and self.orchestrator.multimodal_agent.is_initialized
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
