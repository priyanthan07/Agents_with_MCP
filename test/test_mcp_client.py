# Add this debug code to test the connection manually
import asyncio
from mcp_client.client import create_mcp_client

async def debug_connection():
    try:
        client = create_mcp_client()
        print(f"Client created, server URL: {client.server_url}")
        
        await client._initialize_client(server="web_research")
        print(f"Tools discovered: {[tool['name'] for tool in client.available_tools]}")
        
        # Test a simple tool call
        result = await client.call_tool("web_search", {'query': 'latest AI trends 2023', 'num_results': 5})
        print(f"Test result: {result}")
        
        result = await client.call_tool("analyze_webpage", {'url': 'https://www.artificialintelligence-news.com/', 'extract_text': True, 'summarize': True})
        print(f"Test result: {result}")
        
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

# Run debug
asyncio.run(debug_connection())