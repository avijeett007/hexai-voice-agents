"""Vector memory system for NHS agent

This module provides a vector-based memory system using Qdrant for storing
and retrieving conversation history and important information.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

# Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from .utils import get_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nhs_agent.memory")

# Environment settings
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_TLS = os.getenv("QDRANT_TLS", "False").lower() in ("true", "1", "t")

class VectorMemory:
    """Vector memory system using Qdrant"""
    
    def __init__(self, collection_name: str, file_logger=None):
        self.collection_name = collection_name
        self.file_logger = file_logger  # Will be set by NHSAgent
        self._init_collection()
    
    def set_file_logger(self, file_logger):
        """Set the file logger for this memory system"""
        self.file_logger = file_logger
    
    async def _log_to_file(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log memory operations to file if logger is available"""
        if self.file_logger:
            await self.file_logger.log_event(event_type, content, metadata)
    
    def _init_collection(self):
        """Initialize the vector collection"""
        try:
            # Connect to Qdrant
            if not QDRANT_HOST:
                logger.warning("QDRANT_HOST not set, memory features will be disabled")
                return
            
            self.client = QdrantClient(
                url=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY,
                prefer_grpc=False,
                https=QDRANT_TLS
            )
            
            vector_size = 1536  # OpenAI embedding size
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE
                    )
                )
                logger.info(f"Created new memory collection: {self.collection_name}")
            else:
                logger.info(f"Using existing memory collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing vector memory: {str(e)}")
    
    def add_to_memory(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add text to vector memory"""
        try:
            if not hasattr(self, 'client'):
                logger.warning("Qdrant client not initialized, skipping memory storage")
                return False
            
            # Get embedding for text
            embedding = get_embedding(text)
            if not embedding:
                logger.error("Failed to get embedding for text")
                return False
            
            # Insert into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=abs(hash(text + str(metadata))) % (2**63 - 1),  # Unique ID
                        vector=embedding,
                        payload={
                            "text": text,
                            "metadata": metadata or {}
                        }
                    )
                ]
            )
            
            logger.info(f"Added to memory collection {self.collection_name}: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to memory: {str(e)}")
            return False
    
    def query_memory(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Query memory for relevant information"""
        try:
            if not hasattr(self, 'client'):
                logger.warning("Qdrant client not initialized, skipping memory query")
                return []
            
            # Get embedding for query
            query_embedding = get_embedding(query)
            if not query_embedding:
                logger.error("Failed to get embedding for query")
                return []
            
            # Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Format results
            results = []
            for result in search_result:
                payload = result.payload
                results.append({
                    "text": payload.get("text", ""),
                    "similarity": result.score,
                    "metadata": payload.get("metadata", {})
                })
            
            # Log memory retrieval
            if results and self.file_logger:
                memory_text = "\n\n".join([f"- {r['text'][:100]}..." for r in results])
                asyncio.create_task(self.file_logger.log_memory_retrieval(query, memory_text))
            
            logger.info(f"Retrieved {len(results)} memories for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error querying memory: {str(e)}")
            return []

# Add import at the top after we handle the circular import issue
import asyncio
