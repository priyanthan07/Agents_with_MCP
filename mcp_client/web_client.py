import aiohttp
import json
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

from utils.logger import get_logger
from config import MCP_CONFIG

logger = get_logger(__name__)

@dataclass
class MCPToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    timeout: int = 30

@dataclass
class MCPToolResult:
    success: bool
    result: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    
class MCPClient:
    
    def __init__(self):
        self.server_url = MCP_CONFIG["mcp_server_url"]
        self.session = None
        self.available_tools = {}
        
        logger.info(f"MCP Client initialized with server URL: {self.server_url}")
        
    async def __aenter__(self):
        """
            Async context manager entry
        """
        self.session = aiohttp.ClientSession()
        await self._discover_tools()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
            Async context manager exit
        """
        if self.session:
            await self.session.close()
            
    async def _discover_tools(self):
        try:
            async with self.session.get(f"{self.server_url}/tools") as response:
                if response.status == 200:
                    tools_data = await response.json()
                    self.available_tools = tools_data.get("tools", {})
                    logger.info(f"Discovered {len(self.available_tools)} tools from MCP server")
                else:
                    logger.warning(f"Failed to discover tools: HTTP {response.status}")
        
        except Exception as e:
            logger.error(f"Error discovering MCP tools: {e}")
            # Set default tools if discovery fails
            self.available_tools = {
                "web_search": {"description": "Search the web for information"},
                "webpage_analyzer": {"description": "Analyze and extract content from web pages"},
                "url_validator": {"description": "Validate and check URL accessibility"}
            }         
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        if not self.session:
            raise RuntimeError("MCP Client not initialized. Use async context manager.")
        
        if tool_name not in self.available_tools:
            logger.warning(f"Tool {tool_name} not available. Available tools: {list(self.available_tools.keys())}")
            return {
                "success": False,
                "error": f"Tool {tool_name} not available",
                "available_tools": list(self.available_tools.keys())
            }
        
        try:
            # Prepare the request payload
            payload = {
                "tool": tool_name,
                "parameters": parameters
            }
            
            # Make the request to the MCP server
            async with self.session.post(
                f"{self.server_url}/execute",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                if response.status == 200:
                    result_data = await response.json()
                    logger.info(f"Successfully called tool {tool_name}")
                    return {
                        "success": True,
                        **result_data
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Tool call failed: HTTP {response.status} - {error_text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"Tool call timed out after {timeout} seconds")
            return {
                "success": False,
                "error": f"Request timed out after {timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    async def web_search(self, query: str, num_results: int = 10, safe_search: str = "moderate") -> Dict[str, Any]:

        parameters = {
            "query": query,
            "num_results": num_results,
            "safe_search": safe_search
        }
        
        return await self.call_tool("web_search", parameters)
    
    async def analyze_webpage(self, url: str, extract_text: bool = True, summarize: bool = False) -> Dict[str, Any]:
        pass