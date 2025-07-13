import asyncio
from datetime import datetime

from agents.arxiv_agent import ArxivResearchAgent

class SimpleResearchTest:
    """Simple test focusing on the main research method workflow"""
    
    def __init__(self):
        self.agent = None
    
    async def test_research_flow(self):
        """Test the main research method with minimal logging"""
        
        print("🚀 Testing ArXiv Research Agent - Main Flow")
        print("=" * 50)
        
        # Initialize Agent
        print("Initializing agent...")
        try:
            self.agent = await ArxivResearchAgent.create()
            print("✅ Agent initialized with MCP connection")
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return
        
        # Test the main research method
        test_query = "deep learning in computer vision"
        print(f"\n🔍 Research Query: '{test_query}'")
        print("-" * 30)
        
        try:
            # This will show the complete flow
            start_time = datetime.now()
            
            # Override to limit topics for demo 
            original_generate = self.agent._generate_research_topics
            async def demo_topics(query):
                topics = await original_generate(query)
                return topics
            
            self.agent._generate_research_topics = demo_topics
            
            # Call the main research method
            result = await self.agent.research(test_query)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Show the flow results
            print(f"\n📊 RESEARCH COMPLETED ({execution_time:.1f}s)")
            print("=" * 50)
            
            print(f"Query: {result.query}")
            print(f"Topics Generated: {len(result.generated_topics)}")
            
            # Show generated topics
            print("\n🧠 Generated Topics:")
            for i, topic in enumerate(result.generated_topics, 1):
                print(f"  {i}. {topic}")
            
            # Show research results for each topic
            print(f"\n📄 Research Results:")
            for i, topic_result in enumerate(result.topic_results, 1):
                print(f"  Topic {i}: {topic_result.topic}")
                print(f"    - Papers Found: {topic_result.total_papers_found}")
                print(f"    - Insights Generated: {len(topic_result.key_insights)}")
                
                # Show first insight
                if topic_result.key_insights:
                    print(f"    - Key Insight: {topic_result.key_insights[0][:80]}...")
                print()
            
            print(f"📊 Total Papers Analyzed: {result.total_papers_analyzed}")
            
            # Show global synthesis preview
            print(f"\n📝 Global Synthesis (Preview):")
            print("-" * 30)
            synthesis_preview = result.global_synthesis
            print(synthesis_preview)
            
            print("\n✅ Research method completed successfully!")
            
        except Exception as e:
            print(f"❌ Research failed: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Run the simple research test"""
    test = SimpleResearchTest()
    await test.test_research_flow()

if __name__ == "__main__":
    asyncio.run(main())