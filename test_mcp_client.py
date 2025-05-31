# Add this debug code to test the connection manually
import asyncio
from mcp_client.client import create_mcp_client

async def debug_connection():
    try:
        client = create_mcp_client()
        print(f"Client created, server URL: {client.server_url}")
        
        await client._initialize_client(server="multimodal_analysis")
        print(f"Tools discovered: {[tool['name'] for tool in client.available_tools]}")
        
        # Test a simple tool call
        result = await client.call_tool("process_video_file", {
            'file_path': "data/pv.mp4",
            'analyze_audio': True,
            'analyze_visuals': True
        })
        print(f"Test result: {result}")
        
        result = await client.call_tool("process_audio_file", {
            'file_path': "data/ai-trends.mp3",
            'speaker_detection': True,
            'sentiment_analysis': True
        })
        print(f"Test result: {result}")
        
        result = await client.call_tool("process_image_file", {
            'file_path': "data/AI.jpg",
            'extract_text': True,
            'analyze_content': True
        })
        print(f"Test result: {result}")
        
        result = await client.call_tool("process_document_file", {
            'file_path': "data/report.pdf",
            'extract_images': True,
            'analyze_structure': True
        })
        print(f"Test result: {result}")
        
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

# Run debug
asyncio.run(debug_connection())