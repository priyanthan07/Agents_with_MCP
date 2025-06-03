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

GEMINI_CONFIG = {
    "api_key" : os.getenv("GEMINI_API_KEY")
}

ASSEMBLYAI_CONFIG = {
    "api_key" : os.getenv("ASSEMBLYAI_API_KEY")
}

REDIS_CONFIG = {
    "redis_host" : os.getenv("REDIS_HOST"),
    "redis_port" : os.getenv("REDIS_PORT"),
    "redis_db" : os.getenv("REDIS_DB")
}

CHROMA_CONFIG = {
    "chroma_host": os.getenv("CHROMA_HOST", "localhost"),
    "chroma_port": int(os.getenv("CHROMA_PORT", 8000)),
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
        "multimodal_analysis": {
            "url": os.getenv("MCP_MULTIMODAL_ANALYSIS_URL")
        }
    },
    "default_server": "web_research",
    "connection_timeout": 30,
    "retry_attempts": 3
}

# multimodal research
SUPPORTED_EXTENSIONS = [
    '.mp4', '.mpeg', '.avi', '.mov', '.wmv', '.x-flv', '.webm', '.mpg', '.3gpp',  # video
    '.mp3', '.wav', '.aiff', '.flac', '.aac', '.ogg',                            # audio
    '.jpeg', '.png', '.heic', '.heif', '.webp',                                  # image
    '.pdf', '.csv', '.md', '.txt', '.html', '.css', '.xml'                       # document
]

DATA_DIRECTORY_CONFIG = {
    "path" : os.getenv("DATA_DIRECTORY_PATH")
}