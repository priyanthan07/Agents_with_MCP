import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import requests
import json
from typing import Dict, List, Any, Optional
import aiohttp
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass, asdict
from mcp.server.fastmcp import FastMCP
from util.logger import get_logger
from config import TAVILY_CONFIG
from tavily import AsyncTavilyClient

logger = get_logger("WebServer")
    
mcp = FastMCP("ReAct Web Research Tools Server", port=8001)


# Global client variable - will be initialized in startup
tavily_client = None

async def initialize_tavily(): 
    global tavily_client
      
    try:
        tavily_client = AsyncTavilyClient(api_key=TAVILY_CONFIG["api_key"])
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Tavily client: {e}")
        return False

@mcp.tool()
async def web_search(query: str, num_results: int = 10) -> dict:
    
    if not tavily_client:
        return {"success": False, "error": "Tavily client not initialized", "results": []}
    
    try:
        response = await tavily_client.search(
            query=query,                    
            search_depth="advanced",        
            max_results=min(num_results, 20),  
            include_answer=True,           
            include_raw_content=True,      
            include_images=False,  
            include_image_descriptions= True,
            topic="general"                 
        )
        
        results = []
        for result in response["results"]:
            
            content = result.get("content", "")
            snippet = content[:500] if len(content) > 500 else content
            
            formatted_result = {
                "url" : result.get("url", ""),
                "title" : result.get("title", ""),
                "snippet" : snippet
            }
            results.append(formatted_result)
            
        if response["answer"]:
            summary = {
                "url" : "",
                "title" : "AI-Generated Research Summary",
                "snippet" : response["answer"]
            }
            results.insert(0, summary)
        logger.info(f" Tavily search for '{query}' returned {len(results)}")        
        return {"success": True,"results": results}        

    except Exception as e:
        logger.error(f"Unexpected error in web_search: {e}")
        return {"success": False,"error": str(e),"results": []}
    
@mcp.tool()
async def analyze_webpage(url: str, extract_text: bool = True, summarize: bool = True) -> dict:
    
    if not tavily_client:
        return {"success": False, "error": "Tavily client not initialized", "results": []}
    
    try:
        extract_response = await tavily_client.extract(
                    urls=[url],                    # Can extract from multiple URLs
                    include_images=False,    
                    extract_depth="basic",         
                )
        
        if extract_response["results"] and len(extract_response["results"]) > 0:
            result = extract_response["results"][0]
            content = result.get("raw_content", "")
            
            title = "Extracted Content"
                    
            logger.info(f" Tavily extraction successful for {url}")
            if summarize:
                summary = content[:1000] if len(content) >1000 else content
            
            return {
                "success": True,
                "title": title,
                "content": content,
                "summary": summary,
                "word_count": len(content.split()) if content else 0
            }
        else:
            {
                "success": False,
                "error": "No content extracted from URL",
                "url": url
            }
                       
        
    except Exception as e:
        return {"success" : False, "error" : str(e)}
        
def main():
    
    async def startup():
        success = await initialize_tavily()

    try:
        asyncio.run(startup())
        mcp.run(
            transport="streamable-http", 

        )
    except KeyboardInterrupt:
        logger.info("\n Server stopped by user")
    except Exception as e:
        logger.info(f" Server error: {e}")

if __name__ == "__main__":
    main()