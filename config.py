import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "default_model": "gpt-4o-2024-08-06",
    "temperature": 0.1,
}

REDIS_CONFIG = {
    "redis_host" : "localhost",
    "redis_port" : 6379,
    "redis_db" : 0
}

CHROMA_CONFIG = {
    "chroma_persist_directory" : "./data/chroma_db"
}

MCP_CONFIG = {
    "servers": {
        "web_research": {
            "url": "http://127.0.0.1:8001/mcp"
        },
        "arxiv_research": {
            "url": "http://127.0.0.1:8002/mcp"
        },
        "document_analysis": {
            "url": "http://127.0.0.1:8003/mcp"
        }
    },
    "default_server": "web_research",
    "connection_timeout": 30,
    "retry_attempts": 3
}
