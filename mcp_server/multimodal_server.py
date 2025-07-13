import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import assemblyai as aai
from google.genai import types
from google import genai
from mcp.server.fastmcp import FastMCP
from util.logger import get_logger
from config import GEMINI_CONFIG, ASSEMBLYAI_CONFIG

# Suppress all third-party logging to avoid duplicates
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR) 
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)

logger = get_logger("Multi-Modal")

mcp = FastMCP("Multi-Modal Research MCP Server", port=8003)

try:
    gemini_client = genai.Client(api_key=GEMINI_CONFIG["api_key"])
    
    # Configure AssemblyAI
    aai.settings.api_key = ASSEMBLYAI_CONFIG["api_key"]
    logger.info("Multi-modal Gemini and AssemblyAI clients initialized")
    
except Exception as e:
    logger.error(f"Error initializing Gemini and AssemblyAI clients: {e}")

@mcp.tool()
def process_video_file(file_path: str, analyze_audio: bool = True, analyze_visuals: bool = True) -> dict:
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"Video file not found: {file_path}"}
            
        visual_descriptions = ""
        transcript = ""
        
        # Exttract the visual descriptions   
        if analyze_visuals:
            prompt = "Provide visual descriptions of what's happening in this video with timestamps. Focus on scenes, objects, people, actions, and any visual elements that would be important for research analysis."
         
            video_bytes = open(file_path, 'rb').read()

            video_response = gemini_client.models.generate_content(
                model='models/gemini-2.0-flash',
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                        ),
                        types.Part(text=prompt)
                    ]
                )
            )
            visual_descriptions = video_response.candidates[0].content.parts[0].text
        
        # Exttract the trascript
        if analyze_audio:
            config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
            _transcript = aai.Transcriber(config=config).transcribe(file_path)
            if _transcript.status == "error":
                raise RuntimeError(f"Transcription failed: {_transcript.error}")
            
        transcript = _transcript.text
        
        # extract details
        if visual_descriptions and transcript:
            combined_content = f"VISUAL DESCRIPTIONS:\n{visual_descriptions}\n\nAUDIO TRANSCRIPT:\n{transcript}"
        elif visual_descriptions:
            combined_content = f"VISUAL DESCRIPTIONS:\n{visual_descriptions}"
        elif transcript:
            combined_content = f"AUDIO TRANSCRIPT:\n{transcript}"
        else:
            return {"success": False, "error": "No content could be extracted from video"}
        
        try:
            synthesis_prompt = f""" 
                You have been provided with visual descriptions and audio transcript from the same video.
                
                Content to analyze:
                {combined_content}
                
                Provide a comprehensive analysis that:
                1. Synthesizes information from both visual and audio elements
                2. Identifies key themes, topics, and insights
                3. Notes any correlations between what is seen and what is heard
                4. Extracts specific details that would be valuable for research
                5. Maintains chronological flow using the timestamps provided
                
                Focus on accuracy and detail. Do not add information not present in the provided content.
            """
            
            _response = gemini_client.models.generate_content(
                        model="gemini-2.0-flash",
                        config=types.GenerateContentConfig(
                            system_instruction="You are a research analyst specializing in video content analysis.",
                        ),
                        contents=synthesis_prompt
                    )
            
            content_extracted = _response.text
            
        except Exception as e:
            logger.warning(f"Failed to generate synthesis, using raw content: {e}")
            content_extracted = combined_content
        
        processing_result = {
            "success": True,
            "file_path": file_path,
            "file_type": "video",
            "content_extracted": content_extracted,
            "metadata": {
                "extract_audio": analyze_audio,
                "analyze_visuals": analyze_visuals,
                "processing_model": "gemini-2.0-flash",
                "file_size": os.path.getsize(file_path),
            },
        }
        logger.info(f"Successfully processed video: {file_path}")
        return {"success": True, "processing_result": processing_result}
        
    except Exception as e:
        logger.error(f"Error processing video {file_path}: {e}")
        return {"success": False, "error": str(e)}
    
@mcp.tool()
def process_audio_file(file_path: str, speaker_detection: bool = True, sentiment_analysis: bool = True) -> dict:
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"Audio file not found: {file_path}"}
        
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            speaker_labels=speaker_detection,
            sentiment_analysis=sentiment_analysis
        )
        
        transcript = aai.Transcriber(config=config).transcribe(file_path)
        if transcript.status == "error":
            return {"success": False, "error": f"Transcription failed: {transcript.error}"}
        
        content_extracted = transcript.text
        
        # Get additional features
        additional_features = {}
        if speaker_detection and hasattr(transcript, 'utterances') and transcript.utterances:
            speakers_info = []
            for utterance in transcript.utterances:
                speakers_info.append({
                    "speaker": utterance.speaker,
                    "text": utterance.text,
                    "start": utterance.start,
                    "end": utterance.end
                })
            additional_features["speakers"] = speakers_info
            
        if sentiment_analysis and hasattr(transcript, 'sentiment_analysis_results') and transcript.sentiment_analysis_results:
            sentiment_info = []
            for sentiment in transcript.sentiment_analysis_results:
                sentiment_info.append({
                    "text": sentiment.text,
                    "sentiment": sentiment.sentiment,
                    "confidence": sentiment.confidence
                })
            additional_features["sentiment"] = sentiment_info
            
        processing_result = {
            "success": True,
            "file_path": file_path,
            "file_type": "audio",
            "content_extracted": content_extracted,
            "metadata": {
                "speaker_detection": speaker_detection,
                "sentiment_analysis": sentiment_analysis,
                "processing_model": "assemblyai_universal-2",
                "additional_features": additional_features,
                "confidence": getattr(transcript, 'confidence', 0),
                "file_size": os.path.getsize(file_path)
            },
        }
        logger.info(f"Successfully processed audio: {file_path}")
        return {"success": True, "processing_result": processing_result}
    
    except Exception as e:
        logger.error(f"Error processing audio {file_path}: {e}")
        return {"success": False, "error": str(e)}
    
@mcp.tool()
def process_image_file(file_path: str, extract_text: bool = True, analyze_content: bool = True) -> dict:
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"Image file not found: {file_path}"}
        
        if extract_text and analyze_content:
            prompt = "Analyze this image. Extract any text (OCR) and describe the visual content in detail."
        elif extract_text:
            prompt = "Extract all text visible in this image (OCR)."
        elif analyze_content:
            prompt = "Describe the visual content of this image in detail."
        else:
            prompt = "Describe what you see in this image."
                
        my_file = gemini_client.files.upload(file=file_path)
        response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[my_file, prompt],
            )
        content_extracted = response.text  
        
        processing_result = {
            "success": True,
            "file_path": file_path,
            "file_type": "image",
            "content_extracted": content_extracted,
            "metadata": {
                "extract_text": extract_text,
                "analyze_content": analyze_content,
                "processing_model": "gemini-2.0-flash",
                "file_size": os.path.getsize(file_path),
                "token_count": response.usage_metadata.total_token_count if response.usage_metadata else 0
            },
        }  
        
        logger.info(f"Successfully processed image: {file_path}")
        return {"success": True, "processing_result": processing_result}   
           
    except Exception as e:
        logger.error(f"Error processing image {file_path}: {e}")
        return {"success": False, "error": str(e)}
    
@mcp.tool()
def process_document_file(file_path: str, extract_images: bool = True, analyze_structure: bool = True) -> dict:
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"Document file not found: {file_path}"}
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.pdf':
            with open(file_path, 'rb') as doc_file:
                doc_bytes = doc_file.read()
                
            if extract_images and analyze_structure:
                prompt = "Analyze this PDF document. Extract all text, describe any images/figures, and analyze the structure."
            elif extract_images:
                prompt = "Extract text from this PDF and describe any images or figures."
            elif analyze_structure:
                prompt = "Extract text from this PDF and analyze its structure."
            else:
                prompt = "Extract all text from this PDF."
                
            response = gemini_client.models.generate_content(
                    model='models/gemini-2.0-flash',
                    contents=types.Content(
                        parts=[
                            types.Part(inline_data=types.Blob(data=doc_bytes, mime_type='application/pdf')),
                            types.Part(text=prompt)
                        ]
                    )
                )
        else:
            # For text files, read as text
            try:
                with open(file_path, 'r', encoding='utf-8') as doc_file:
                    doc_text = doc_file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as doc_file:
                    doc_text = doc_file.read()
            
            # Create prompt for text analysis
            if analyze_structure:
                prompt = f"Analyze this document's structure and content:\n\n{doc_text}"
            else:
                prompt = f"Summarize this document:\n\n{doc_text}"
            
            # Process with Gemini
            response = gemini_client.models.generate_content(
                model='models/gemini-2.0-flash',
                contents=prompt
            )
        
        content_extracted = response.candidates[0].content.parts[0].text
        
        processing_result = {
            "success": True,
            "file_path": file_path,
            "file_type": "document",
            "content_extracted": content_extracted,
            "metadata": {
                "extract_images": extract_images,
                "analyze_structure": analyze_structure,
                "processing_model": "gemini-2.0-flash",
                "document_type": file_ext,
                "file_size": os.path.getsize(file_path),
                "token_count": response.usage_metadata.total_token_count if response.usage_metadata else 0
            }
        }
        
        logger.info(f"Successfully processed document: {file_path}")
        
        return {"success": True, "processing_result": processing_result}
        
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        return {"success": False, "error": str(e)}
    
def main():
    try:
        logger.info("Starting Multi-Modal Research MCP Server...")
        logger.info("Available tools: process_video_file, process_audio_file, process_image_file, process_document_file")
        mcp.run(transport="streamable-http")
        
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()