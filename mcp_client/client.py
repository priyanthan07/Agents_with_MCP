import aiohttp
import json
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import AsyncExitStack
import sys
from datetime import datetime
from util.logger import get_logger
from config import MCP_CONFIG

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = get_logger(__name__)
    
class MCPClient:
    
    def __init__(self):
        self.server_url = None
        self.available_tools: List[Dict[str, Any]] = []
        self.tools_discovered = False
            
    async def _initialize_client(self, server: str):
        if self.tools_discovered:
            return 
        
        try:
            servers_config = MCP_CONFIG["servers"][server]
            self.server_url = servers_config["url"]
            
            logger.info(f"Initialize MCP Web Client of {server}server and URL: {self.server_url}")
            
            # discover tools
            async with streamablehttp_client(self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    response = await session.list_tools()
                    tools = response.tools
                    
                    self.available_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in tools]
                    
                    self.tools_discovered = True
                    logger.info(f"Discovered tools: {[tool.name for tool in tools]}")
                    
        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
            raise      
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        
        try:
            async with streamablehttp_client(self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=parameters)
                    
                    if hasattr(result, 'content') and result.content:
                        content_text = result.content[0].text if result.content else ""
                        try:
                            parsed_content = json.loads(content_text)
                            return parsed_content
                        except json.JSONDecodeError:
                            return {"success": True, "result": content_text}
                    else:
                        return {"success": True, "result": content_text}
        
        except Exception as e:
            logger.error(f" Tool call failed: {e}")
            return {"success": False, "error": str(e)}

# Factory function to match your current usage
def create_mcp_client() -> MCPClient:
    """Create MCP client - matches your current pattern"""
    return MCPClient()