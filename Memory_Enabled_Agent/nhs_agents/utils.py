"""Utility functions for the NHS agent

This module provides utility functions for embedding generation,
string matching, and other helper functions.
"""

import os
import logging
from typing import List, Optional

# OpenAI for embeddings
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nhs_agent.utils")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding for text using OpenAI's embedding API
    
    Args:
        text: The text to get embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        text = text.replace("\n", " ")
        
        # Call OpenAI API
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"  # Use the appropriate model
        )
        
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None


def fuzzymatch(a: str, b: str, threshold: float = 0.3) -> bool:
    """Helper function to do fuzzy matching on strings"""
    # Quick equality check
    if a == b:
        return True
    
    a = a.lower().strip()
    b = b.lower().strip()
    
    # Simple substring check
    if a in b or b in a:
        return True
    
    # Word overlap check
    a_words = set(a.split())
    b_words = set(b.split())
    
    # If either set is empty, no overlap
    if not a_words or not b_words:
        return False
    
    # Calculate overlap ratio
    overlap = len(a_words.intersection(b_words))
    overlap_ratio = overlap / max(len(a_words), len(b_words))
    
    return overlap_ratio >= threshold
