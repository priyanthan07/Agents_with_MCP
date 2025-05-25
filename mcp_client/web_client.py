import aiohttp
import json
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import AsyncExitStack
import sys
from datetime import datetime
from util.logger import get_logger
from config import MCP_CONFIG

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

logger = get_logger(__name__)
    
class MCPClient:
    
    def __init__(self):
        self.servers_config = MCP_CONFIG["servers"]["web_research"]
        self.server_url = self.servers_config["url"]
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
        self.connected = False
        
        logger.info(f"MCP Web Client initialized with server URL: {self.server_url}")
            
    async def connect_to_server_and_setup(self):
        try:
            logger.info(f" Connecting to MCP server: {self.server_url}")
            
            async with streamablehttp_client(self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    
                    response = await session.list_tools()
                    tools = response.tools

                    logger.info(f" Connected to server with tools: {[tool.name for tool in tools]}")
                    
                    self.available_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in tools]
                    
                    self.connected = True
                    return session
            
        except Exception as e:
            logger.error(f" Failed to connect to MCP server: {e}")
            raise        
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if not self.connected or not self.session:
            return {"success": False, "error": "Not connected to MCP server", "available_tools": "" }
        
        if tool_name not in self.available_tools:
            logger.warning(f"Tool {tool_name} not available. Available tools: {list(self.available_tools.keys())}")
            return {"success": False, "error": f"Tool {tool_name} not available", "available_tools": list(self.available_tools.keys())}
        
        try:
            result = await self.session.call_tool(tool_name, arguments=parameters)
            
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