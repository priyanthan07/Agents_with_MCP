import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass, asdict
from mcp.server.fastmcp import FastMCP
# from util.logger import get_logger

# logger = get_logger(__name__)

@dataclass
class ToolResult:
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""
    
mcp = FastMCP("ReAct Web Research Tools Server", port=8001)

@mcp.tool()
async def web_search(query: str, num_results: int = 10) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                'q':query,
                'format': 'json',
                'no_html': '1'
            }
            
            async with session.get("https://api.duckduckgo.com/", params=params) as response:
                
                if response.status == 200:
                    data = await response.json()
                
                    results = []
                    if data["RelatedTopics"]:
                        for topic in data["RelatedTopics"][:num_results]:
                            if (isinstance(topic, dict)) and ('Text' in topic):
                                results.append({
                                    "url" : topic.get('FirstURL', ''),
                                    "title" : topic.get('Text', '')[:100],
                                    "snippet" : topic.get('Text', '')
                                })
                                
                    # If no related topics, use abstract
                    if (not results) and data['Abstract']:
                        results.append({
                            "url" : data.get('AbstractURL', ''),
                            "title" : data.get('Heading', query),
                            "snippet" : data.get('Abstract', '')
                        })
                    
                    return {"success": True,"results": results}
                    
        return {"success": True,"results": []}        

    except Exception as e:
        return {"success": False,"error": str(e),"results": []}
    
@mcp.tool()
async def analyze_webpage(url: str, extract_text: bool = True, summarize: bool = False) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else "No Title"
                    
                    content = ""
                    if extract_text:
                        # Remove unwanted elements
                        for element in soup(["script", "style", "nav", "header", "footer"]):
                            element.decompose()
                        
                        main = soup.find('main') or soup.find('article') or soup.find('body')
                        if main:
                            content = main.get_text(separator=' ', strip=True)
                            
                    summary = ""
                    if summarize and content:
                        sentences = content.split('.')[:3]  # First 3 sentences
                        summary = '. '.join(sentences).strip() + '.'
                    
                    return {
                        "success": True,
                        "title": title_text,
                        "content": content,
                        "summary": summary,
                        "word_count": len(content.split()) if content else 0
                    }
                    
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}      
        
    except Exception as e:
        return {"success" : False, "error" : str(e)}
    
@mcp.tool()
async def validate_url(url: str) -> dict:
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {
                "success": True,
                "valid": False,
                "accessible": False,
                "error": "Invalid URL format"
            }
        
        # Check if URL is accessible
        async with aiohttp.ClientSession() as session:
            async with session.head(url, allow_redirects=True) as response:
                return {
                    "success": True,
                    "valid": True,
                    "accessible": response.status < 400,
                    "status_code": response.status
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Tool execution failed: {str(e)}",
            "url_checked": url
        }
        
def main():
    """Run the HTTP MCP server"""
    print("🚀 Starting MCP HTTP Server...")
    print("🔧 Available tools: web_search, analyze_webpage, validate_url")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        # Run server on HTTP transport
        mcp.run(
            transport="streamable-http", 

        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()