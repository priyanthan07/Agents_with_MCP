import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import arxiv
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from util.logger import get_logger
import logging

# Suppress all third-party logging to avoid duplicates
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR) 
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)

logger = get_logger(__name__)

mcp = FastMCP("Comprehensive ArXiv Research Server", port=8002)

@mcp.tool()
def search_papers(query: str, max_results: int = 8) -> dict:
    try:
        logger.info(f"Starting ArXiv search for query: '{query}' (max_results: {max_results})", context="ArXivServer")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for paper in client.results(search):
            paper_data = {
                'paper_id': paper.get_short_id(),
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'abstract': paper.summary,
                'published_date': str(paper.published.date()),
                'categories': paper.categories,
                'pdf_url': paper.pdf_url,
                'entry_id': paper.entry_id,
                'updated_date': str(paper.updated.date()) if paper.updated else None
            }
            papers.append(paper_data)
        
        logger.info(f"Successfully found {len(papers)} papers for query: '{query}'", context="ArXivServer")
        
        return {
            "success": True,
            "papers": papers,
            "query": query,
            "total_found": len(papers),
            "search_metadata": {
                "max_results": max_results
            }
        }   
        
    except Exception as e:
        logger.error(f"Failed to search papers for query '{query}': {str(e)}", context="ArXivServer")
        return {
            "success": False,
            "error": str(e),
            "papers": [],
            "query": query
        }


@mcp.tool()
def get_paper_details(paper_ids: List[str]) -> dict:
    try:
        logger.info(f"Retrieving details for {len(paper_ids)} papers", context="ArXivServer")
        
        client = arxiv.Client()
        paper_details = []
        
        for paper_id in paper_ids:
            try:
                search = arxiv.Search(id_list=[paper_id])
                paper = next(client.results(search))
                
                detailed_info = {
                    'paper_id': paper.get_short_id(),
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'published_date': str(paper.published.date()),
                    'updated_date': str(paper.updated.date()) if paper.updated else None,
                    'categories': paper.categories,
                    'primary_category': paper.primary_category,
                    'pdf_url': paper.pdf_url,
                    'entry_id': paper.entry_id,
                    'comment': getattr(paper, 'comment', ''),
                    'journal_ref': getattr(paper, 'journal_ref', ''),
                    'doi': getattr(paper, 'doi', ''),
                    'links': [link.href for link in paper.links],
                    'abstract_length': len(paper.summary.split()),
                    'author_count': len(paper.authors)
                }
                paper_details.append(detailed_info)
            
            except Exception as paper_error:
                logger.error(f"Failed to get details for paper {paper_id}: {str(paper_error)}", context="ArXivServer")
                paper_details.append({
                    'paper_id': paper_id,
                    'error': str(paper_error),
                    'success': False
                })
        
        successful_count = len([p for p in paper_details if 'error' not in p])     
        logger.info(f"Successfully retrieved {successful_count}/{len(paper_ids)} paper details", context="ArXivServer") 
        
        return {
            "success": True,
            "paper_details": paper_details,
            "total_processed": len(paper_ids),
            "successful_retrievals": len([p for p in paper_details if 'error' not in p])
        }
        
    except Exception as e:
        logger.error(f"Critical error retrieving paper details: {str(e)}", context="ArXivServer")
        return {
            "success": False,
            "error": str(e),
            "paper_ids": paper_ids
        }

def main():
    try:
        logger.info("ArXiv MCP Server starting on port 8002", context="ArXivServer")
        logger.info("Available tools: search_papers, get_paper_details", context="ArXivServer")
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        logger.info("ArXiv MCP Server stopped by user", context="ArXivServer")
    except Exception as e:
        logger.error(f"Critical server error: {str(e)}", context="ArXivServer")
        
if __name__ == "__main__":
    main()