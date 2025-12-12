"""Structured file logger for NHS agent conversations

This module provides a structured logging system for NHS agent interactions,
including user queries, agent responses, knowledge retrieval, and more.
"""

import os
import asyncio
import datetime
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nhs_agent")

# Create logs directory if it doesn't exist
logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

class StructuredFileLogger:
    """Structured file logger for NHS agent conversations"""
    
    def __init__(self, user_id: str, user_type: str):
        """Initialize the structured logger
        
        Args:
            user_id (str): User's ID (NHS number or registration number)
            user_type (str): Type of user ('patient' or 'doctor')
        """
        self.user_id = user_id
        self.user_type = user_type
        
        # Create unique log filename with timestamp and user info
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_id = str(uuid.uuid4())[:8]  # Short unique ID
        safe_user_id = "".join([c if c.isalnum() else "_" for c in user_id])
        
        filename = f"{timestamp}_{user_type}_{safe_user_id}_{self.log_id}.log"
        self.log_file = logs_dir / filename
        
        # Create the log file with header
        with open(self.log_file, "w", encoding="utf-8") as f:
            header = (
                f"=== NHS AGENT CONVERSATION LOG ===\n"
                f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"User Type: {user_type}\n"
                f"User ID: {user_id}\n"
                f"Log ID: {self.log_id}\n"
                f"===============================\n\n"
            )
            f.write(header)
        
        logger.info(f"Created structured log file: {self.log_file}")
        
        # Track knowledge bases used
        self.knowledge_bases_used: Set[str] = set()
    
    async def log_event(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log an event to the structured log file asynchronously
        
        Args:
            event_type (str): Type of event (e.g., 'USER_QUERY', 'CONTEXT', 'AGENT_RESPONSE')
            content (str): Main content of the event
            metadata (Dict[str, Any], optional): Additional metadata for the event
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Format the log entry
            log_entry = f"[{timestamp}] {event_type}\n"
            
            if content:
                # Format content with indentation for readability
                formatted_content = "\n".join(f"    {line}" for line in content.split("\n"))
                log_entry += f"{formatted_content}\n"
            
            # Add metadata if provided
            if metadata:
                # Format metadata as indented JSON for readability
                try:
                    metadata_str = json.dumps(metadata, indent=4)
                    formatted_metadata = "\n".join(f"    {line}" for line in metadata_str.split("\n"))
                    log_entry += f"METADATA:\n{formatted_metadata}\n"
                except:
                    # Fallback if JSON conversion fails
                    log_entry += f"METADATA: {str(metadata)}\n"
            
            log_entry += "---\n\n"
            
            # Write to file asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._write_to_file,
                log_entry
            )
            
        except Exception as e:
            logger.error(f"Error writing to structured log file: {e}")
    
    def _write_to_file(self, content: str):
        """Write content to log file (called by run_in_executor)"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
    
    async def log_user_query(self, query: str):
        """Log a user query"""
        await self.log_event("USER_QUERY", query)
    
    async def log_context(self, context: str, context_type: str = "GENERAL"):
        """Log context added to the agent"""
        await self.log_event("CONTEXT", context, {"type": context_type})
    
    async def log_agent_response(self, response: str):
        """Log agent text response"""
        await self.log_event("AGENT_RESPONSE", response)
    
    async def log_tts_output(self, text: str, duration_ms: int = None):
        """Log Text-to-Speech output"""
        metadata = {"duration_ms": duration_ms} if duration_ms else {}
        await self.log_event("TTS_OUTPUT", text, metadata)
    
    async def log_knowledge_retrieval(self, kb_id: str, query: str, content: str):
        """Log knowledge retrieval from vector database"""
        # Track that this knowledge base was used
        self.knowledge_bases_used.add(kb_id)
        
        await self.log_event(
            "KNOWLEDGE_RETRIEVAL", 
            content,
            {
                "knowledge_base_id": kb_id,
                "query": query
            }
        )
    
    async def log_memory_retrieval(self, query: str, content: str):
        """Log memory retrieval"""
        await self.log_event(
            "MEMORY_RETRIEVAL", 
            content,
            {"query": query}
        )
    
    async def log_conversation_summary(self, num_turns: int, duration_sec: int = None):
        """Log conversation summary at the end"""
        metadata = {
            "num_turns": num_turns,
            "duration_sec": duration_sec,
            "knowledge_bases_used": list(self.knowledge_bases_used)
        }
        
        summary = f"Conversation completed with {num_turns} turns"
        if duration_sec is not None:
            minutes = duration_sec // 60
            seconds = duration_sec % 60
            summary += f" over {minutes}m {seconds}s"
        
        await self.log_event("CONVERSATION_SUMMARY", summary, metadata)
