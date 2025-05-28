# Add this debug code to test the connection manually
import asyncio
from mcp_client.arxiv_client import create_mcp_client

async def debug_connection():
    try:
        client = create_mcp_client()
        print(f"Client created, server URL: {client.server_url}")
        
        await client._ensure_tools_discovered()
        print(f"Tools discovered: {[tool['name'] for tool in client.available_tools]}")
        
        # Test a simple tool call
        result = await client.call_tool("search_papers", {"query": "test", "max_results": 2})
        print(f"Test result: {result}")
        
        paper_ids =[paper['paper_id'] for paper in result["papers"]]
        
        result = await client.call_tool("get_paper_details", {"paper_ids": paper_ids})
        print(f"Test result: {result}")
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

# Run debug
asyncio.run(debug_connection())