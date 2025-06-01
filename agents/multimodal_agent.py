import asyncio
import json
import os
import mimetypes
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI
from pathlib import Path

from mcp_client.client import create_mcp_client
from util.logger import get_logger
from config import OPENAI_CONFIG, SUPPORTED_EXTENSIONS, DATA_DIRECTORY_CONFIG

logger = get_logger(__name__)

@dataclass
class MediaFile:
    file_path: str
    file_type: str       # "video", "audio", "image", "document"
    mime_type: str       # Multipurpose Internet Mail Extension
    file_size: int
    
@dataclass
class ProcessingResult:
    file_path: str
    file_type: str
    content_extracted: str
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    
@dataclass
class MultiModalResearchResult:
    query: str
    data_directory: str
    files_discovered: List[MediaFile]
    processing_results: List[ProcessingResult]
    synthesis: str
    key_insights: List[str]
    files_processed: int
    processing_summary: Dict[str, int]
    
class MultiModalResearchAgent:
    """
        Sequential Multi-Modal Research Agent that processes files in order:
        1. Scan directory
        2. Process videos (if any)
        3. Process images (if any) 
        4. Process audio (if any)
        5. Process documents (if any)
        6. Synthesize all results
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        self.data_directory = DATA_DIRECTORY_CONFIG["path"]
        self.is_initialized = False
        self.available_tools = ["process_video_file", "process_audio_file", "process_image_file", "process_document_file"]
        self.processing_order = ["video", "image", "audio", "document"]
        
        logger.info(f"Sequential Multi-Modal Agent initialized.")
        
    @classmethod
    async def create(cls) -> 'MultiModalResearchAgent':
        agent = cls()
        await agent._initialize_mcp_connection()
        agent.is_initialized = True
        return agent
    
    async def _initialize_mcp_connection(self):
        try:
            self.mcp_client = create_mcp_client()
            await self.mcp_client._initialize_client(server="multimodal_analysis") 
            
            server_tools = [tool["name"] for tool in self.mcp_client.available_tools]
            expected_tools = self.available_tools
            
            for expected_tool in expected_tools:
                if expected_tool not in server_tools:
                    logger.warning(f"Expected tool '{expected_tool}' not available from MCP server")
            
            logger.info(f"MCP connection established. Available tools in server : {server_tools}")
        
        except Exception as e:
            logger.error(f"Failed to connect with web mcp client: {e}")
            raise RuntimeError(e)
    
    async def research(self, task_query: str) -> MultiModalResearchResult:
        logger.info(f"Starting research for: {task_query}")
        
        # Step 1: Scan directory for all files
        discovered_files = await self._scan_directory()
        
        # Step 2: Organize files by type
        files_by_type = self._organize_files_by_type(discovered_files)
        
        logger.info(f" Existing files : {files_by_type}")
        
        # Step 3: Process each type sequentially
        all_results = []
        processing_summary = {}
        
        for file_type in self.processing_order:
            if file_type in files_by_type:
                logger.info(f"Processing {len(files_by_type[file_type])} {file_type} files...")
                
                results = await self._process_file_type(file_type, files_by_type[file_type])
                all_results.extend(results)
                
                successful = len([result for result in results if result.success])
                processing_summary[file_type] = successful
                
                logger.info(f"Completed {file_type}: {successful}/{len(results)} successful") 
                
        # Step 4: Extract insights from all results
        all_insights = []
        for result in all_results:
            if result.success:
                insights = await self._extract_insights(result.content_extracted, task_query, result.file_type)
                all_insights.extend(insights)
                
        # Step 5: Create final synthesis
        synthesis = await self._create_synthesis(task_query, all_results, all_insights)
        
        return MultiModalResearchResult(
            query=task_query,
            data_directory=self.data_directory,
            files_discovered=discovered_files,
            processing_results=all_results,
            synthesis=synthesis,
            key_insights=all_insights,
            files_processed=len([r for r in all_results if r.success]),
            processing_summary=processing_summary
        )
    
    async def _scan_directory(self) -> List[MediaFile]:
        try:
            if not os.path.exists(self.data_directory):
                return {"success": False, "error": f"Directory not found: {self.data_directory}", "files": []}
            
            discovered_files = []
            for root, dirs, files in os.walk(self.data_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = Path(file).suffix.lower()
                    logger.info(f"file : {file}")
                    
                    if file_ext in SUPPORTED_EXTENSIONS:
                        try:
                            file_stats = os.stat(file_path)
                            mime_type, _ = mimetypes.guess_type(file_path)
                            
                            file_info = MediaFile(
                                file_path= file_path, 
                                file_type = await self.classify_file_type(file_path),
                                mime_type = mime_type or "application/octet-stream",
                                file_size = file_stats.st_size
                            )

                            discovered_files.append(file_info)
                            
                        except OSError:
                            continue
                        
            return discovered_files
        
        except Exception as e:
            logger.error(f"Error scanning directory: {e}")
            return []
        
    async def classify_file_type(self, file_path: str) -> str:
        
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.mp4', '.mpeg', '.avi', '.mov', '.wmv', '.x-flv', '.webm', '.mpg', '.3gpp']:
            return "video"
        elif ext in ['.mp3', '.wav', '.aiff', '.flac', '.aac', '.ogg']:
            return "audio"
        elif ext in ['.jpeg', '.png', '.heic', '.heif', '.webp']:
            return "image"
        elif ext in ['.pdf', '.csv', '.md', '.txt', '.html', '.css', '.xml']:
            return "document"
        else:
            return "unknown"
        
    def _organize_files_by_type(self, files: List[MediaFile]) -> Dict[str, List[MediaFile]]:
        files_by_type = {}
        for file in files:
            if file.file_type not in files_by_type:
                files_by_type[file.file_type] = []
            files_by_type[file.file_type].append(file)
        return files_by_type
    
    async def _process_file_type(self, file_type: str, files: List[MediaFile]) -> List[ProcessingResult]:
        
        results = []
        for file in files:
            try:
                if file_type == "video":
                    result = await self._process_video(file)
                    
                elif file_type == "audio":
                    result = await self._process_audio(file)
                    
                elif file_type == "image":
                    result = await self._process_image(file)
                
                elif file_type == "document":
                    result = await self._process_document(file)
                    
                else:
                    result = self._create_error_result(file, f"Unknown file type: {file_type}")
                    
                results.append(result)
            
            except Exception as e:
                error_result = self._create_error_result(file, str(e))
                results.append(error_result)
                
        return results
    
    async def _process_video(self, file: MediaFile) -> ProcessingResult:
        try:
            response = await self.mcp_client.call_tool("process_video_file", {
                "file_path": file.file_path,
                "analyze_audio": True,
                "analyze_visuals": True
            })
            
            if response.get("success"):
                data = response["processing_result"]
                return ProcessingResult(
                    file_path=file.file_path,
                    file_type="video",
                    content_extracted=data["content_extracted"],
                    metadata=data["metadata"],
                    success=True
                )
                 
            else:
                return self._create_error_result(file, response.get("error", "Video processing failed"))
            
        except Exception as e:
            return self._create_error_result(file, str(e))
        
    async def _process_audio(self, file: MediaFile) -> ProcessingResult:
        try:
            response = await self.mcp_client.call_tool("process_audio_file", {
                "file_path": file.file_path,
                "speaker_detection": True,
                "sentiment_analysis": True
            })
            
            if response.get("success"):
                data = response["processing_result"]
                return ProcessingResult(
                    file_path=file.file_path,
                    file_type=file.file_type,
                    content_extracted=data["content_extracted"],
                    metadata=data["metadata"],
                    success=True
                )
            else:
                return self._create_error_result(file, response.get("error", "Audio processing failed"))
            
        except Exception as e:
            return self._create_error_result(file, str(e))
        
    async def _process_image(self, file: MediaFile) -> ProcessingResult:
        try:
            response = await self.mcp_client.call_tool("process_image_file", {
                "file_path": file.file_path,
                "extract_text": True,
                "analyze_content": True
            })
            
            if response.get("success"):
                data = response["processing_result"]
                return ProcessingResult(
                    file_path=file.file_path,
                    file_type="image",
                    content_extracted=data["content_extracted"],
                    metadata=data["metadata"],
                    success=True
                )
            else:
                return self._create_error_result(file, response.get("error", "Image processing failed"))
            
        except Exception as e:
            return self._create_error_result(file, str(e))
        
    async def _process_document(self, file: MediaFile) -> ProcessingResult:
        try:
            response = await self.mcp_client.call_tool("process_document_file", {
                "file_path": file.file_path,
                "extract_images": True,
                "analyze_structure": True
            })
            
            if response.get("success"):
                data = response["processing_result"]
                return ProcessingResult(
                    file_path=file.file_path,
                    file_type="document",
                    content_extracted=data["content_extracted"],
                    metadata=data["metadata"],
                    success=True
                )
            else:
                return self._create_error_result(file, response.get("error", "Document processing failed"))
            
        except Exception as e:
            return self._create_error_result(file, str(e))
        
    def _create_error_result(self, file: MediaFile, error_message: str) -> ProcessingResult:
        """Create error result for failed processing"""
        return ProcessingResult(
            file_path=file.file_path,
            file_type=file.file_type,
            content_extracted="",
            metadata={},
            success=False,
            error_message=error_message
        )
        
    async def _extract_insights(self, content: str, query: str, content_type: str) -> List[str]:
        try:
            prompt = f"""
                Extract key insights from this {content_type} content for the research query: {query}
                
                Content: {content[:2000]}
                
                Provide 3-5 specific insights that relate to the query.
                Return only the insights, one per line.
            """
            
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": f"Extract research insights from {content_type} content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            insights = [i.strip() for i in response.choices[0].message.content.strip().split('\n') if i.strip()]
            return insights[:5]
        
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []
        
    async def _create_synthesis(self, query: str, results: List[ProcessingResult], insights: List[str]) -> str:
        try:
            content_by_type = {}
            for result in results:
                if result.success:
                    if result.file_type not in content_by_type:
                        content_by_type[result.file_type] = []
                    content_by_type[result.file_type].append(result.content_extracted[:500])
                        
            prompt = f"""
                Create a comprehensive research synthesis for: {query}
                
                Content processed:
                {json.dumps(content_by_type, indent=2)}
                
                Key insights found:
                {", ".join(insights)}
                
                Create a well-structured synthesis that:
                1. Answers the research query
                2. Integrates findings from all media types
                3. Highlights key patterns and conclusions
            """
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "Create comprehensive research synthesis from multi-modal content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
                        
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error creating synthesis: {e}")
            return f"Research completed with {len([r for r in results if r.success])} files processed for query: {query}"
        
        
    def _create_empty_result(self, query: str) -> MultiModalResearchResult:
        return MultiModalResearchResult(
            query=query,
            data_directory=self.data_directory,
            files_discovered=[],
            processing_results=[],
            synthesis="No supported media files found in the directory.",
            key_insights=[],
            files_processed=0,
            processing_summary={}
        )
        