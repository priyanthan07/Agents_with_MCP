import json
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from util.logger import get_logger
from config import OPENAI_CONFIG

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
    
class severityType(Enum, str):
    low = 'low'
    medium = 'medium'
    high = 'high'

class conData(BaseModel):
    topic: str
    severity: severityType
    
class ContradictionDetector(BaseModel):
    contradictions: List[conData]
    reasoning: str
    
class AnalyzeResolutionFormat(BaseModel):
    CONCLUSION = str
    CONFIDENCE = float
    
class ResearchValidator:
    """
        Validates research results and detects contradictions across agent outputs.
        Uses LLM-based analysis to identify conflicts and generate resolution strategies.
    """
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        
        logger.info("Validation layer initialized successfully")

    async def detect_contradictions(self, agent_results: Dict[str, Any]) -> List[Contradiction]:
        try:
            contradictions = []
            
            # Extract claims from each agent
            web_claims = self._extract_web_claims(agent_results["web_result"])
            arxiv_claims = self._extract_arxiv_claims(agent_results["arxiv_result"])
            media_claims = self._extract_media_claims(agent_results["multimodal_result"])

            if web_claims and arxiv_claims:
                contradictions.extend(await self._compare_sources("web", web_claims, "arxiv", arxiv_claims))
                
            if web_claims and media_claims:
                contradictions.extend(await self._compare_sources("web", web_claims, "media", media_claims))
            
            if arxiv_claims and media_claims:
                contradictions.extend(await self._compare_sources("arxiv", arxiv_claims, "media", media_claims))
            
            logger.info(f"Found {len(contradictions)} contradictions")
            return contradictions
        
        except Exception as e:
            logger.error(f"Error finding contradictions: {e}")
            return []
        
    def _extract_web_claims(self, web_result) -> List[str]:
        try:
            if not web_result:
                return []
            
            claims = []
            if hasattr(web_result, 'key_findings'):
                claims.extend(web_result.key_findings)
            if hasattr(web_result, 'summary'):
                claims.append(web_result.summary)
                
            return claims
        
        except Exception as e:
            logger.error(f"Error extract web claims: {e}")
            return []
        
    def _extract_arxiv_claims(self, arxiv_result) -> List[str]:
        try:
            if not arxiv_result:
                return []
            
            claims = []
            if hasattr(arxiv_result, 'topic_results'):
                for topic_result in arxiv_result.topic_results:
                    if hasattr(topic_result, 'key_insights'):
                        claims.extend(arxiv_result.key_findings)
                
            if hasattr(arxiv_result, 'global_synthesis'):
                claims.append(arxiv_result.global_synthesis)
                
            return claims
        
        except Exception as e:
            logger.error(f"Error extract arxiv claims: {e}")
            return []
        
    def _extract_media_claims(self, media_result) -> List[str]:
        try:
            if not media_result:
                return []
            
            claims = []
            if hasattr(media_result, 'key_insights'):
                claims.extend(media_result.key_insights)
            if hasattr(media_result, 'synthesis'):
                claims.append(media_result.synthesis)
            return claims
        
        except Exception as e:
            logger.error(f"Error extract media claims: {e}")
            return []
        
    async def _compare_sources(self, source1: str, claims1: List[str], source2: str, claims2: List[str]) -> List[Contradiction]:
        if not claims1 or not claims2:
            return []
        
        try:
            prompt = f"""
                Compare these claims from two research sources and identify contradictions.
                
                {source1.upper()} CLAIMS:
                {chr(10).join([f"- {claim}" for claim in claims1[:5]])}
                
                {source2.upper()} CLAIMS:
                {chr(10).join([f"- {claim}" for claim in claims2[:5]])}
                
                Find contradictions where claims make opposite statements about the same topic.
                For each contradiction, provide:
                - topic: The topic they contradict on
                - severity: low/medium/high
                - reasoning: Why they contradict
                
                Only identify clear contradictions, not minor differences.
            """
            response = self.client.responses.parse(
                model=OPENAI_CONFIG["default_model"],
                input=[
                    {"role": "system", "content": "You identify contradictions between research sources."},
                    {"role": "user", "content": prompt}
                ],
                text_format=ContradictionDetector,
                temperature=0.2
            )
            
            result = json.loads(response.output[0].content[0].text)
            contradictions = []
            
            for contradiction_data in result.contradictions:
                contradiction = Contradiction(
                    id=str(uuid.uuid4()),
                    source1=source1,
                    source2=source2,
                    claim1=claims1,
                    claim2=claims2,
                    topic=contradiction_data.get("topic", ""),
                    severity=contradiction_data.get("severity", "medium")
                )
                contradictions.append(contradiction)
            return contradictions
            
        except Exception as e:
            logger.error(f"Error comparing {source1} vs {source2}: {e}")
            return []
        
    async def generate_resolution_query(self, contradiction: Contradiction) -> str:
        try:
            prompt = f"""
                Generate a web search query to resolve this contradiction:
                
                Topic: {contradiction.topic}
                Claim 1: {contradiction.claim1}
                Claim 2: {contradiction.claim2}
                severity : {contradiction.severity}
                
                Create a specific search query to find authoritative sources that can determine which claim is correct.
                Return only the search query.
            """
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "Generate precise fact-checking search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating resolution query: {e}")
            return f"verify {contradiction.topic}"
        
    async def analyze_resolution(self, contradiction: Contradiction, evidence: str) -> Resolution:
        try:
            prompt = f"""
                Analyze this evidence to resolve the contradiction:
                
                Original Contradiction:
                - Topic: {contradiction.topic}
                - Claim 1: {contradiction.claim1}
                - Claim 2: {contradiction.claim2}
                
                Evidence: {evidence[:1000]}
                
                Provide:
                1. Which claim is more accurate (if determinable)
                2. Your confidence (0.0-1.0)
                3. Brief conclusion
                
                Format:
                CONCLUSION: conclusion
                CONFIDENCE: in between 0.0 and 1.0
            """
            response = self.client.responses.parse(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "Analyze evidence objectively to resolve contradictions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                text_format=AnalyzeResolutionFormat
            )
            
            analysis  = json.loads(response.output[0].content[0].text)
            conclusion = analysis["CONCLUSION"]
            confidence = float(analysis["CONFIDENCE"])
            
            return Resolution(
                contradiction_id=contradiction.id,
                resolution_query="",  # Set by orchestrator
                evidence=evidence[:500],
                conclusion=conclusion,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing resolution: {e}")
            return Resolution(
                contradiction_id=contradiction.id,
                resolution_query="",
                evidence="Analysis failed",
                conclusion="Unable to resolve",
                confidence=0.0
            )     