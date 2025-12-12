"""Knowledge base for NHS agent

This module provides a vector database-backed knowledge retrieval system
for medical information, integrating with Qdrant for vector search.
"""

import os
import asyncio
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple

# Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# OpenAI for embeddings and query refinement
from openai import OpenAI

# Cerebras for cost-efficient knowledge base selection
import importlib.util
is_cerebras_available = importlib.util.find_spec("cerebras") is not None
if is_cerebras_available:
    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError:
        is_cerebras_available = False

# Local imports
from .utils import get_embedding
from .constants import PATIENT_KNOWLEDGE_BASE_MAP, DOCTOR_KNOWLEDGE_BASE_MAP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nhs_agent.knowledge")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Cerebras client if available
cerebras_client = None
if is_cerebras_available and os.getenv("CEREBRAS_API_KEY"):
    try:
        cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        logger.info("Initialized Cerebras client for cost-efficient knowledge selection")
    except Exception as e:
        logger.warning(f"Failed to initialize Cerebras client: {e}")
        cerebras_client = None

# Environment settings
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_TLS = os.getenv("QDRANT_TLS", "False").lower() in ("true", "1", "t")

class KnowledgeBase:
    """Vector database based medical knowledge retrieval system
    
    This system has been simplified to:
    1. Use common vector databases instead of individual knowledge base IDs
    2. Select document metadata (title, authors) for citation rather than knowledge base IDs
    3. Return results with document metadata for citation
    4. If no results are found, return appropriate rejection message
    """
    
    def __init__(self, knowledge_map=None, file_logger=None):
        """Initialize with the appropriate knowledge base map based on user type"""
        self.knowledge_map = knowledge_map or DOCTOR_KNOWLEDGE_BASE_MAP
        self.file_logger = file_logger  # Optional file logger (will be set by NHSAgent)
        self._last_ai_analysis = {
            "refined_query": "",
            "selected_kb_ids": [],
            "explanation": ""
        }
        
        # Initialize Qdrant client
        try:
            if QDRANT_HOST:
                self.client = QdrantClient(
                    url=QDRANT_HOST,
                    port=QDRANT_PORT,
                    api_key=QDRANT_API_KEY,
                    prefer_grpc=False,
                    https=QDRANT_TLS
                )
                logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
            else:
                logger.warning("QDRANT_HOST not set, knowledge base features will be limited")
                self.client = None
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None
    
    def set_file_logger(self, file_logger):
        """Set the file logger for this knowledge base"""
        self.file_logger = file_logger
    
    async def _log_knowledge_event(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log knowledge base event if file logger is available"""
        if self.file_logger:
            await self.file_logger.log_event(event_type, content, metadata)
    
    def ai_analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Use AI to analyze the query and select appropriate document references
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with analysis results including refined query and selected documents
        """
        try:
            knowledge_map_docs = self.knowledge_map["documents"]
            
            # Create a simplified document list for the AI to select from
            kb_options = []
            for i, doc in enumerate(knowledge_map_docs):
                kb_options.append(f"{i+1}. {doc['domain']}: {doc['title']} ({doc['source']}, {doc['year']})")
            
            kb_options_text = "\n".join(kb_options)
            
            # Try Cerebras first (cost-efficient)
            result = None
            explanation = ""
            refined_query = query
            selected_indices = []
            
            if cerebras_client:
                try:
                    # Prompt for Cerebras
                    prompt = f"""Analyze this medical query and help me determine which knowledge sources would be most relevant for answering it.

QUERY: {query}

Available knowledge sources:
{kb_options_text}

Please provide:
1. A refined search query that focuses on the medical aspects
2. The numbers of the 1-3 most relevant knowledge sources (just the numbers)
3. A brief explanation of your selection

Response format:
Refined Query: [refined query]
Selected Sources: [comma-separated numbers of selected sources]
Explanation: [brief explanation]"""
                    
                    # Call Cerebras
                    response = cerebras_client.complete(prompt=prompt, max_tokens=300)
                    text_response = response.completions[0].text.strip()
                    
                    # Parse the response
                    lines = text_response.split("\n")
                    for line in lines:
                        if line.startswith("Refined Query:"):
                            refined_query = line.replace("Refined Query:", "").strip()
                        elif line.startswith("Selected Sources:"):
                            selected_str = line.replace("Selected Sources:", "").strip()
                            # Extract numbers, handling various formats
                            import re
                            selected_indices = [int(x.strip()) for x in re.findall(r'\d+', selected_str)]
                        elif line.startswith("Explanation:"):
                            explanation = line.replace("Explanation:", "").strip()
                    
                    logger.info(f"Cerebras analysis - Selected sources: {selected_indices}")
                    
                except Exception as cerebras_error:
                    logger.warning(f"Cerebras analysis failed, falling back to OpenAI: {cerebras_error}")
                    # Continue to OpenAI fallback
            
            # If not successful with Cerebras or selected sources are empty, try OpenAI
            if not cerebras_client or not selected_indices:
                try:
                    # Prompt for OpenAI
                    messages = [
                        {"role": "system", "content": "You are a medical knowledge specialist helping to match user queries with appropriate knowledge sources."},
                        {"role": "user", "content": f"""Analyze this medical query and help me determine which knowledge sources would be most relevant for answering it.

QUERY: {query}

Available knowledge sources:
{kb_options_text}

Please provide:
1. A refined search query that focuses on the medical aspects
2. The numbers of the 1-3 most relevant knowledge sources (just the numbers)
3. A brief explanation of your selection

Response format:
Refined Query: [refined query]
Selected Sources: [comma-separated numbers of selected sources]
Explanation: [brief explanation]"""}
                    ]
                    
                    # Call OpenAI
                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.3,
                        max_tokens=300
                    )
                    
                    text_response = response.choices[0].message.content.strip()
                    
                    # Parse the response
                    lines = text_response.split("\n")
                    for line in lines:
                        if line.startswith("Refined Query:"):
                            refined_query = line.replace("Refined Query:", "").strip()
                        elif line.startswith("Selected Sources:"):
                            selected_str = line.replace("Selected Sources:", "").strip()
                            # Extract numbers, handling various formats
                            import re
                            selected_indices = [int(x.strip()) for x in re.findall(r'\d+', selected_str)]
                        elif line.startswith("Explanation:"):
                            explanation = line.replace("Explanation:", "").strip()
                    
                    logger.info(f"OpenAI analysis - Selected sources: {selected_indices}")
                    
                except Exception as openai_error:
                    logger.error(f"OpenAI analysis failed: {openai_error}")
                    # If everything fails, return all sources
                    selected_indices = list(range(1, len(knowledge_map_docs) + 1))
                    explanation = "Using all available sources due to analysis failure."
            
            # Adjust indices to be 0-based and ensure they're valid
            selected_indices = [i-1 for i in selected_indices if 0 < i <= len(knowledge_map_docs)]
            
            # If no valid indices, use all
            if not selected_indices:
                selected_indices = list(range(len(knowledge_map_docs)))
                explanation += " (Using all sources as fallback)"
            
            # Get the selected knowledge base IDs
            selected_kb_ids = [knowledge_map_docs[i]["title"] for i in selected_indices if i < len(knowledge_map_docs)]
            
            # Store the analysis result
            self._last_ai_analysis = {
                "refined_query": refined_query,
                "selected_kb_ids": selected_kb_ids,
                "explanation": explanation
            }
            
            # Log the analysis
            if self.file_logger:
                asyncio.create_task(self._log_knowledge_event(
                    "KNOWLEDGE_SELECTION",
                    f"Query: {query}\nRefined: {refined_query}\nSelected: {', '.join(selected_kb_ids)}",
                    {"explanation": explanation}
                ))
            
            return self._last_ai_analysis
            
        except Exception as e:
            error_msg = f"Error in AI query analysis: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # Return a default analysis using all documents
            knowledge_map_docs = self.knowledge_map["documents"]
            selected_kb_ids = [doc["title"] for doc in knowledge_map_docs]
            
            result = {
                "refined_query": query,
                "selected_kb_ids": selected_kb_ids,
                "explanation": "Using all sources due to analysis error."
            }
            
            self._last_ai_analysis = result
            return result
    
    async def _log_vector_query(self, query: str, vector_response: List[Dict[str, Any]], event_type: str = "VECTOR_QUERY"):
        """
        Log vector database query and response
        
        Args:
            query: The original query text
            vector_response: The response from the vector database
            event_type: Type of event to log
        """
        if not self.file_logger:
            return
        
        # Format relevant info from responses
        formatted_results = []
        for item in vector_response:
            # Truncate text if too long
            text = item.get("text", "")
            if len(text) > 300:
                text = text[:297] + "..."
                
            result_info = f"Score: {item.get('score', 0):.4f}\n{text}"
            
            # Add vector analysis if possible
            if "vector" in item and callable(getattr(self, '_describe_vector', None)):
                try:
                    vector_description = await self._describe_vector(item["vector"])
                    if vector_description:
                        result_info += f"\nVector interpretation: {vector_description}"
                except Exception as e:
                    logger.warning(f"Error describing vector: {e}")
            
            formatted_results.append(result_info)
        
        results_text = "\n\n".join([f"Result {i+1}:\n{r}" for i, r in enumerate(formatted_results)])
        
        # Log the query and results
        content = f"Query: {query}\n\n{results_text}"
        metadata = {
            "result_count": len(vector_response),
            "collection": self.knowledge_map["collection_name"]
        }
        
        # Non-blocking log
        await self.file_logger.log_event(event_type, content, metadata)
    
    async def _describe_vector(self, vector: List[float]) -> Optional[str]:
        """
        Convert a vector embedding to human-readable text description using OpenAI
        
        Args:
            vector: The vector embedding
            
        Returns:
            Text description of what the vector might represent
        """
        try:
            # Create a more focused prompt for medical embeddings
            messages = [
                {"role": "system", "content": "You are a vector embedding analyzer for medical information. Your task is to interpret what concepts or information an embedding vector likely represents."},
                {"role": "user", "content": f"This is an embedding vector from a medical query or document. Based on the vector values, provide a concise interpretation (1-2 sentences) of what medical concepts this vector likely represents. Keep your response under 100 characters.\n\nVector (first 10 values): {vector[:10]}"}
            ]
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=50
            )
            
            description = response.choices[0].message.content.strip()
            return description
            
        except Exception as e:
            logger.warning(f"Error describing vector: {e}")
            return None
    
    def get_relevant_knowledge_bases(self, query: str) -> List[Dict[str, Any]]:
        """
        Find relevant knowledge bases using vector similarity
        
        Args:
            query: The query to find relevant knowledge bases for
            
        Returns:
            List of relevant documents metadata
        """
        try:
            # Use AI to analyze query and select appropriate knowledge bases
            ai_analysis = self.ai_analyze_query(query)
            refined_query = ai_analysis["refined_query"]
            
            # Get embedding for the refined query
            query_embedding = get_embedding(refined_query)
            
            if not query_embedding:
                logger.error("Failed to get embedding for query")
                return []
            
            # Get collection name from knowledge map
            collection_name = self.knowledge_map["collection_name"]
            
            # Try to query Qdrant if client is initialized
            vector_results = []
            if self.client:
                try:
                    # Search in the collection
                    search_results = self.client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=5,  # Get top 5 results
                        score_threshold=0.7  # Minimum similarity threshold
                    )
                    
                    # Process and format results
                    for result in search_results:
                        # Get the payload
                        payload = result.payload
                        
                        # Skip if no text
                        if "text" not in payload:
                            continue
                        
                        # Format the result
                        vector_results.append({
                            "text": payload["text"],
                            "score": result.score,
                            "metadata": payload.get("metadata", {}),
                            "vector": query_embedding  # Include the query vector for logging
                        })
                    
                    # Log the vector query results
                    if self.file_logger:
                        asyncio.create_task(self._log_vector_query(refined_query, vector_results))
                    
                    logger.info(f"Found {len(vector_results)} vector results for query: {refined_query[:50]}...")
                    
                except Exception as qdrant_error:
                    logger.error(f"Qdrant search error: {qdrant_error}")
                    # Continue to fallback mechanism
            
            # If no results found from Qdrant or Qdrant not available, try local fallback
            if not vector_results:
                logger.info("No vector results from Qdrant, using local fallback")
                vector_results = self._fallback_local_similarity(refined_query, query_embedding, self.knowledge_map["documents"])
                
                # Log the fallback results
                if self.file_logger and vector_results:
                    asyncio.create_task(self._log_vector_query(
                        refined_query, 
                        vector_results,
                        "FALLBACK_SIMILARITY"
                    ))
            
            # Return the processed results
            return vector_results
            
        except Exception as e:
            error_msg = f"Error getting relevant knowledge bases: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return []
    
    def _fallback_local_similarity(self, query: str, query_embedding: List[float], knowledge_map_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback method using local similarity matching
        
        Args:
            query: The original query
            query_embedding: The embedding of the query
            knowledge_map_docs: Documents from the knowledge map
            
        Returns:
            List of documents with similarity scores
        """
        try:
            results = []
            
            # Process each document in the knowledge map
            for doc in knowledge_map_docs:
                # Create document text from metadata
                doc_text = f"{doc['title']}. {doc['description']}. Source: {doc['source']} ({doc['year']})."
                
                # Get embedding for document text
                doc_embedding = get_embedding(doc_text)
                if not doc_embedding:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_cosine_similarity(query_embedding, doc_embedding)
                
                # Add to results if similarity is above threshold
                if similarity > 0.6:  # Threshold can be adjusted
                    results.append({
                        "text": doc_text,
                        "score": similarity,
                        "metadata": doc,
                        "vector": doc_embedding
                    })
            
            # Sort by similarity score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Take top 3 results
            return results[:3]
            
        except Exception as e:
            logger.error(f"Error in fallback similarity: {e}")
            return []
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            import numpy as np
            
            # Convert lists to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_comprehensive_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Get comprehensive knowledge from relevant knowledge bases
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with text and sources
        """
        try:
            # Get relevant knowledge bases
            vector_results = self.get_relevant_knowledge_bases(query)
            
            if not vector_results:
                # No knowledge found
                return {
                    "text": "I don't have verified information on this medical topic. Please consult a healthcare professional for accurate guidance.",
                    "sources": [],
                    "knowledge_found": False
                }
            
            # Extract text and sources from results
            comprehensive_text = ""
            sources = []
            
            for result in vector_results:
                comprehensive_text += result["text"] + "\n\n"
                
                # Extract metadata for citation
                metadata = result.get("metadata", {})
                if metadata:
                    source = {
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", ""),
                        "authors": metadata.get("authors", ""),
                        "year": metadata.get("year", ""),
                        "url": metadata.get("url", ""),
                        "domain": metadata.get("domain", "")
                    }
                    sources.append(source)
            
            return {
                "text": comprehensive_text.strip(),
                "sources": sources,
                "knowledge_found": True
            }
            
        except Exception as e:
            error_msg = f"Error getting comprehensive knowledge: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                "text": "I encountered an error retrieving information. Please try again or consult a healthcare professional.",
                "sources": [],
                "knowledge_found": False
            }
    
    def get_kb_info_by_id(self, kb_id: str) -> Dict[str, Any]:
        """Get knowledge base info by document title"""
        for doc in self.knowledge_map["documents"]:
            if doc["title"] == kb_id:
                return doc
        return None
