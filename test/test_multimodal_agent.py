import asyncio
import os
import json
from dataclasses import asdict
from dotenv import load_dotenv

load_dotenv()

from agents.multimodal_agent import MultiModalResearchAgent

async def test_multimodal_agent():
    """Simple test for multimodal agent functionality"""
    
    try:
        # Initialize agent
        agent = await MultiModalResearchAgent.create()
        
        # Check data directory
        if not os.path.exists(agent.data_directory):
            print(f"❌ Data directory not found: {agent.data_directory}")
            return False
        
        # Run research
        query = "Analyze all media content and extract key information"
        result = await agent.research(query)
        
        # Display results
        print(f"✅ Research completed")
        print(f"Files discovered: {len(result.files_discovered)}")
        print(f"Files processed: {result.files_processed}")
        print(f"Processing summary: {result.processing_summary}")
        
        output_file= "results.json"
        result_dict = asdict(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def main():
    success = await test_multimodal_agent()
    print("✅ Test passed" if success else "❌ Test failed")

if __name__ == "__main__":
    asyncio.run(main())