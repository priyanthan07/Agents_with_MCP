import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "default_model": "gpt-4o-2024-08-06",
    "temperature": 0.1,
}

MCP_CONFIG = {
    "mcp_server_url" : "http://localhost:8000"
}

REDIS_CONFIG = {
    "redis_host" : "",
    "redis_port" : 6379,
    "redis_db" : 0
}

CHROMA_CONFIG = {
    "chroma_persist_directory" : "./data/chroma_db"
}