# app.py - Fixed version with proper event loop handling
import streamlit as st
import asyncio
import json
import concurrent.futures
import threading
from datetime import datetime
from main import MultiAgentResearchSystem

# Configure page
st.set_page_config(
    page_title="🔬 AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

def run_async_with_new_loop(coro):
    """
    Safely run async code in Streamlit by creating a completely new event loop
    in a separate thread, avoiding conflicts with Streamlit's internal loops.
    """
    def _run_in_thread():
        # Create fresh event loop in this thread
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            # Clean up the loop
            new_loop.close()
    
    # Execute in thread pool to isolate from Streamlit's threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        return future.result()

async def initialize_system():
    """Initialize the research system"""
    try:
        system = MultiAgentResearchSystem()
        await system.initialize()
        return system
    except Exception as e:
        st.error(f"❌ System initialization failed: {e}")
        return None

async def run_research(system, query):
    """Run research query"""
    try:
        result = await system.research(query)
        return result
    except Exception as e:
        st.error(f"❌ Research failed: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 style="text-align: center;">🔬 AI Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Powered by Multi-Agent AI System</p>', unsafe_allow_html=True)
    
    # Initialize system button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Initialize AI System", use_container_width=True, type="primary"):
            with st.spinner("🔄 Starting AI agents..."):
                try:
                    # Use our safe async runner
                    system = run_async_with_new_loop(initialize_system())
                    if system:
                        st.session_state.system = system
                        st.success("✅ AI Research System Ready!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Initialization failed: {e}")

    # Main research interface
    if st.session_state.system:
        st.markdown("### 💭 What would you like to research?")
        
        # Sample queries for inspiration
        with st.expander("💡 Need inspiration? Try these examples"):
            sample_queries = [
                "Latest developments in artificial intelligence and machine learning",
                "Impact of climate change on global food security",
                "Benefits and risks of renewable energy technologies",
                "Future of remote work and its effects on productivity",
                "Ethical implications of genetic engineering and CRISPR"
            ]
            
            for i, sample in enumerate(sample_queries):
                if st.button(f"📝 {sample}", key=f"sample_{i}"):
                    st.session_state.sample_query = sample
                    st.rerun()
        
        # Query input
        default_query = st.session_state.get('sample_query', '')
        query = st.text_area(
            "Enter your research question:",
            value=default_query,
            height=100,
            placeholder="e.g., What are the latest trends in quantum computing?",
        )
        
        # Research button
        if st.button("🔍 Start Research", use_container_width=True, type="primary", disabled=not query.strip()):
            if query.strip():
                with st.spinner("🤖 AI agents are researching... This may take a few minutes"):
                    try:
                        # Use our safe async runner for research
                        result = run_async_with_new_loop(run_research(st.session_state.system, query))
                        if result:
                            st.session_state.current_result = result
                            st.session_state.research_history.append(result)
                            st.success("✅ Research completed!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Research failed: {e}")
                        # Show the detailed error for debugging
                        with st.expander("🔍 Error Details"):
                            st.code(str(e))
    
    else:
        st.info("👆 Please initialize the AI system first to start researching")
    
    # Display results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        st.markdown("---")
        st.markdown("## 📊 Research Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sources Analyzed", result.sources_analyzed)
        with col2:
            total_insights = len(result.web_insights) + len(result.academic_insights) + len(result.media_insights)
            st.metric("Insights Generated", total_insights)
        with col3:
            st.metric("Contradictions Found", len(result.contradictions_found))
        with col4:
            cache_status = "📋 Cached" if result.used_cache else "🆕 Fresh"
            st.metric("Research Type", cache_status)
        
        # Executive Summary
        st.markdown("### 📋 Executive Summary")
        st.info(result.executive_summary)
        
        # Detailed Analysis
        st.markdown("### 📖 Detailed Analysis")
        st.markdown(result.detailed_analysis)
        
        # Insights by source
        tab1, tab2, tab3 = st.tabs(["🌐 Web Insights", "📚 Academic Insights", "🎬 Media Insights"])
        
        with tab1:
            if result.web_insights:
                for i, insight in enumerate(result.web_insights, 1):
                    st.markdown(f"**{i}.** {insight}")
            else:
                st.info("No web insights available")
        
        with tab2:
            if result.academic_insights:
                for i, insight in enumerate(result.academic_insights, 1):
                    st.markdown(f"**{i}.** {insight}")
            else:
                st.info("No academic insights available")
        
        with tab3:
            if result.media_insights:
                for i, insight in enumerate(result.media_insights, 1):
                    st.markdown(f"**{i}.** {insight}")
            else:
                st.info("No media insights available")
        
        # Contradictions
        if result.contradictions_found:
            st.markdown("### ⚠️ Contradictions & Resolutions")
            for i, contradiction in enumerate(result.contradictions_found, 1):
                with st.expander(f"Contradiction {i}: {contradiction.topic}"):
                    st.markdown(f"**Severity:** {contradiction.severity}")
                    st.markdown(f"**Sources:** {contradiction.source1} vs {contradiction.source2}")
                    
                    # Find resolution
                    resolution = next((r for r in result.resolutions if r.contradiction_id == contradiction.id), None)
                    if resolution:
                        st.markdown(f"**Resolution:** {resolution.conclusion}")
                        st.markdown(f"**Confidence:** {resolution.confidence:.2f}")
        
        # Export results
        st.markdown("### 💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            json_data = {
                "query": result.query,
                "executive_summary": result.executive_summary,
                "detailed_analysis": result.detailed_analysis,
                "web_insights": result.web_insights,
                "academic_insights": result.academic_insights,
                "media_insights": result.media_insights,
                "sources_analyzed": result.sources_analyzed,
                "timestamp": result.timestamp.isoformat()
            }
            
            st.download_button(
                label="📄 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            text_report = f"""
                RESEARCH REPORT
                ===============
                Query: {result.query}
                Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                Sources: {result.sources_analyzed}

                EXECUTIVE SUMMARY
                ================
                {result.executive_summary}

                DETAILED ANALYSIS
                ================
                {result.detailed_analysis}

                WEB INSIGHTS
                ============
                {chr(10).join([f"• {insight}" for insight in result.web_insights])}

                ACADEMIC INSIGHTS
                =================
                {chr(10).join([f"• {insight}" for insight in result.academic_insights])}

                MEDIA INSIGHTS
                ==============
                {chr(10).join([f"• {insight}" for insight in result.media_insights])}
            """
            
            st.download_button(
                label="📝 Download Report",
                data=text_report,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Research history sidebar
    if st.session_state.research_history:
        st.sidebar.markdown("## 📚 Recent Research")
        for i, research in enumerate(reversed(st.session_state.research_history[-5:]), 1):
            query_preview = research.query[:40] + "..." if len(research.query) > 40 else research.query
            cache_icon = "📋" if research.used_cache else "🆕"
            if st.sidebar.button(f"{cache_icon} {query_preview}", key=f"history_{i}"):
                st.session_state.current_result = research
                st.rerun()

if __name__ == "__main__":
    main()
    