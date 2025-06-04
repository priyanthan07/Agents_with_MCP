import streamlit as st
import asyncio
import json
from datetime import datetime
from main import MultiAgentResearchSystem

# Configure page
st.set_page_config(
    page_title="🔬 AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .query-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .result-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .insight-item {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .loading-text {
        text-align: center;
        color: #667eea;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

async def initialize_system():
    """Initialize the research system"""
    if st.session_state.system is None:
        try:
            st.session_state.system = MultiAgentResearchSystem()
            await st.session_state.system.initialize()
            return True
        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            return False
    return True

async def run_research(query):
    """Run research query"""
    try:
        result = await st.session_state.system.research(query)
        st.session_state.current_result = result
        st.session_state.research_history.append(result)
        return result
    except Exception as e:
        st.error(f"❌ Research failed: {e}")
        return None

def run_async(coro):
    """Helper to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 AI Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Powered by Multi-Agent AI System with Web, Academic & Multimodal Analysis</p>', unsafe_allow_html=True)
    
    # Initialize system button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Initialize AI System", use_container_width=True, type="primary"):
            with st.spinner("🔄 Starting AI agents..."):
                success = run_async(initialize_system())
                if success:
                    st.success("✅ AI Research System Ready!")
                    st.rerun()

    # Main research interface
    if st.session_state.system:
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        
        # Research form
        st.markdown("### 💭 What would you like to research?")
        
        # Sample queries for inspiration
        with st.expander("💡 Need inspiration? Try these examples"):
            sample_queries = [
                "Latest developments in artificial intelligence and machine learning",
                "Impact of climate change on global food security",
                "Benefits and risks of renewable energy technologies",
                "Future of remote work and its effects on productivity",
                "Ethical implications of genetic engineering and CRISPR",
                "How social media algorithms influence human behavior"
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
            placeholder="e.g., What are the latest trends in quantum computing and their potential applications?",
            help="Be specific about what you want to research. The AI will analyze web sources, academic papers, and multimedia content."
        )
        
        # Research button
        if st.button("🔍 Start Research", use_container_width=True, type="primary", disabled=not query.strip()):
            if query.strip():
                with st.spinner("🤖 AI agents are researching... This may take a few minutes"):
                    # Progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.markdown('<p class="loading-text">🌐 Web Research Agent is searching...</p>', unsafe_allow_html=True)
                    progress_bar.progress(33)
                    
                    status_text.markdown('<p class="loading-text">📚 Academic Research Agent is analyzing papers...</p>', unsafe_allow_html=True)
                    progress_bar.progress(66)
                    
                    status_text.markdown('<p class="loading-text">🎬 Multimodal Agent is processing content...</p>', unsafe_allow_html=True)
                    progress_bar.progress(100)
                    
                    # Run research
                    result = run_async(run_research(query))
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    if result:
                        st.success("✅ Research completed!")
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("👆 Please initialize the AI system first to start researching")
    
    # Display results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        st.markdown("---")
        st.markdown("## 📊 Research Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{result.sources_analyzed}</h3>
                <p>Sources Analyzed</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            total_insights = len(result.web_insights) + len(result.academic_insights) + len(result.media_insights)
            st.markdown(f'''
            <div class="metric-card">
                <h3>{total_insights}</h3>
                <p>Insights Generated</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{len(result.contradictions_found)}</h3>
                <p>Contradictions Found</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            cache_status = "📋 Cached" if result.used_cache else "🆕 Fresh"
            st.markdown(f'''
            <div class="metric-card">
                <h3>{cache_status}</h3>
                <p>Research Type</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Executive Summary
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("### 📋 Executive Summary")
        st.markdown(f"**{result.executive_summary}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Analysis
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("### 📖 Detailed Analysis")
        st.markdown(result.detailed_analysis)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Insights by source
        tab1, tab2, tab3 = st.tabs(["🌐 Web Insights", "📚 Academic Insights", "🎬 Media Insights"])
        
        with tab1:
            if result.web_insights:
                for i, insight in enumerate(result.web_insights, 1):
                    st.markdown(f'''
                    <div class="insight-item">
                        <strong>{i}.</strong> {insight}
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No web insights available")
        
        with tab2:
            if result.academic_insights:
                for i, insight in enumerate(result.academic_insights, 1):
                    st.markdown(f'''
                    <div class="insight-item">
                        <strong>{i}.</strong> {insight}
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No academic insights available")
        
        with tab3:
            if result.media_insights:
                for i, insight in enumerate(result.media_insights, 1):
                    st.markdown(f'''
                    <div class="insight-item">
                        <strong>{i}.</strong> {insight}
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No media insights available")
        
        # Contradictions and resolutions
        if result.contradictions_found:
            st.markdown("### ⚠️ Contradictions & Resolutions")
            for i, contradiction in enumerate(result.contradictions_found, 1):
                with st.expander(f"Contradiction {i}: {contradiction.topic} ({contradiction.severity} severity)"):
                    st.markdown(f"**Sources:** {contradiction.source1} vs {contradiction.source2}")
                    st.markdown(f"**Issue:** Conflicting information about {contradiction.topic}")
                    
                    # Find resolution
                    resolution = next((r for r in result.resolutions if r.contradiction_id == contradiction.id), None)
                    if resolution:
                        st.markdown(f"**Resolution:** {resolution.conclusion}")
                        confidence_color = "🟢" if resolution.confidence > 0.7 else "🟡" if resolution.confidence > 0.4 else "🔴"
                        st.markdown(f"**Confidence:** {confidence_color} {resolution.confidence:.2f}")
        
        # Download results
        st.markdown("### 💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_data = {
                "query": result.query,
                "executive_summary": result.executive_summary,
                "detailed_analysis": result.detailed_analysis,
                "web_insights": result.web_insights,
                "academic_insights": result.academic_insights,
                "media_insights": result.media_insights,
                "sources_analyzed": result.sources_analyzed,
                "timestamp": result.timestamp.isoformat(),
                "methodology": result.methodology
            }
            
            st.download_button(
                label="📄 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Text export
            text_report = f"""
RESEARCH REPORT
===============

Query: {result.query}
Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Sources Analyzed: {result.sources_analyzed}
Methodology: {result.methodology}

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