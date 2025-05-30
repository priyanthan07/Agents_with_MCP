import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your components
from agents.web_agent import WebResearchAgent

async def test_web_agent():
    print("🧪 Starting Web Agent Test...")
    print("=" * 50)
    
    try:
        
        # Step 2: Create web agent
        print("3️⃣ Initializing Web Research Agent...")
        web_agent = await WebResearchAgent().create()
        print("✅ Web agent ready!")
        
        # Step 3: Run a simple research query
        print("4️⃣ Starting research...")
        query = "What are the latest trends in artificial intelligence?"
        print(f"Research Query: {query}")
        
        # Execute the research
        result = await web_agent.research(query)
        
        # Step 4: Display results
        print("\n" + "=" * 50)
        print("🎉 RESEARCH COMPLETED!")
        print("=" * 50)
        
        print(f"Query: {result.query}")
        print(f"Research Depth: {result.research_depth}")
        print(f"Sources Analyzed: {result.sources_analyzed}")
        print(f"Key Findings Count: {len(result.key_findings)}")
        print(f"ReAct Cycles: {len(result.react_trace)}")
        
        print(f"\n📋 Research Summary:")
        print(f"{result.summary}")
        
        if result.key_findings:
            print(f"\n🔍 Key Findings:")
            for i, finding in enumerate(result.key_findings[:5], 1):
                print(f"  {i}. {finding}")
        
        print(f"\n🔄 ReAct Reasoning Trace:")
        for i, step in enumerate(result.react_trace, 1):
            print(f"  Cycle {i}:")
            print(f"    💭 THOUGHT: {step.thought}")
            print(f"    🎯 ACTION: {step.action}")
            print(f"    👁️ OBSERVATION: {step.observation}")
            print(f"    🤔 REFLECTION: {step.reflection}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    
    # Run the actual test
    success = await test_web_agent()
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed. Check the error messages above.")

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())