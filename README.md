# Multi-Agent Research System

A comprehensive AI research platform that combines web search, academic papers, and multimodal content analysis through specialized agents communicating via MCP (Model Context Protocol).

## Features

- **Web Research Agent**: ReAct-based web search and analysis
- **ArXiv Agent**: Academic paper discovery and synthesis  
- **Multimodal Agent**: Video, audio, image, and document processing
- **Smart Caching**: Redis + ChromaDB for similarity-based result reuse
- **Validation**: Automatic contradiction detection and resolution
- **Orchestration**: Intelligent coordination of all agents

## Quick Start

### 1. Setup Environment
```bash
# Clone and navigate
git clone <repo-url>
cd Agents_with_MCP

```

### 2. Start Infrastructure
```bash
docker-compose up -d  # Redis + ChromaDB
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt  # or use uv
```

### 4. Run Application
```bash
streamlit run app.py
```

### 5. Run MCP servers in separate CLIs
```bash
python arxiv_server.py
python multimodal_server.py
python web_server.py
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Orchestrator   │    │  Memory Cache   │
│      UI         │────│     Agent        │────│ Redis +ChromaDB │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌──────────┐ ┌─────────────┐ ┌──────────────┐
            │Web Agent │ │ArXiv Agent  │ │Multimodal    │
            │(ReAct)   │ │(Academic)   │ │Agent         │
            └──────────┘ └─────────────┘ └──────────────┘
                    │           │           │
                    ▼           ▼           ▼
            ┌──────────┐ ┌─────────────┐ ┌──────────────┐
            │Web MCP   │ │ArXiv MCP    │ │Multimodal    │
            │Server    │ │Server       │ │MCP Server    │
            └──────────┘ └─────────────┘ └──────────────┘
```

## Configuration

Create `.env` with required API keys:
```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key  
GEMINI_API_KEY=your_gemini_key
ASSEMBLYAI_API_KEY=your_assemblyai_key
REDIS_HOST=localhost
REDIS_PORT=6379
CHROMA_HOST=localhost
CHROMA_PORT=8000
DATA_DIRECTORY_PATH=./data  # For multimodal files
```

## Project Structure

```
├── agents/              # Specialized research agents
├── mcp_server/         # MCP protocol servers  
├── mcp_client/         # MCP client implementation
├── src/                # Core system components
├── util/               # Utilities (logging, etc.)
├── test/               # Test suite
├── app.py              # Streamlit interface
├── main.py             # System entry point
└── docker-compose.yml  # Infrastructure setup
```

## Blog Series

Read the complete 6-part blog series on Medium covering:
1. blog 1 : https://medium.com/@govindarajpriyanthan/from-theory-to-practice-building-a-multi-agent-research-system-with-mcp-part-1-d63e89ab8b0a
2. blog 2 : https://medium.com/@govindarajpriyanthan/from-theory-to-practice-building-a-multi-agent-research-system-with-mcp-part-2-811b0163e87c  
3. blog 3 : https://medium.com/@govindarajpriyanthan/from-theory-to-practice-building-a-multi-agent-research-system-with-mcp-part-3-be89bbadb7f1
4. blog 4 : https://medium.com/@govindarajpriyanthan/from-theory-to-practice-building-a-multi-agent-research-system-with-mcp-part-4-7ce70e588378
5. blog 5 : https://medium.com/@govindarajpriyanthan/from-theory-to-practice-building-a-multi-agent-research-system-with-mcp-part-5-6f9b2ed9eae8
6. blog 6 : https://medium.com/@govindarajpriyanthan/from-theory-to-practice-building-a-multi-agent-research-system-with-mcp-part-6-9e452cc5696b

## License

MIT License - see LICENSE file for details.