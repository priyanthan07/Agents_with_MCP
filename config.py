import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "default_model": "gpt-4o-2024-08-06",
    "temperature": 0.1,
}

TAVILY_CONFIG = {
    "api_key" : os.getenv("TAVILY_API_KEY")
}

REDIS_CONFIG = {
    "redis_host" : os.getenv("REDIS_HOST"),
    "redis_port" : os.getenv("REDIS_PORT"),
    "redis_db" : os.getenv("REDIS_DB")
}

CHROMA_CONFIG = {
    "chroma_persist_directory" : os.getenv("CHROMA_PERSIST_DIRECTORY")
}

MCP_CONFIG = {
    "servers": {
        "web_research": {
            "url": os.getenv("MCP_WEB_RESEARCH_URL")
        },
        "arxiv_research": {
            "url": os.getenv("MCP_ARXIV_RESEARCH_URL")
        },
        "document_analysis": {
            "url": os.getenv("MCP_DOCUMENT_ANALYSIS_URL")
        }
    },
    "default_server": "web_research",
    "connection_timeout": 30,
    "retry_attempts": 3
}
