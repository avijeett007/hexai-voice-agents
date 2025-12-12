"""Core NHS agent implementation

This module contains the NHSAgent class which integrates all components
including memory, knowledge retrieval, and conversation management.
"""

import os
import re
import asyncio
import logging
import datetime
import json
import traceback
from typing import List, Dict, Any, Optional, Set, Tuple

# LiveKit imports
from livekit.agents import llm
from livekit.agents.pipeline import VoicePipelineAgent
import livekit.rtc as rtc

# Local imports
from .models import UserData, PatientData
from .memory import VectorMemory
from .knowledge import KnowledgeBase
from .logger import StructuredFileLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nhs_agent.agent")


class NHSFunctions:
    """Functions available to the NHS agent LLM"""
    
    def __init__(self, user_data: UserData, memory: VectorMemory, knowledge_base: KnowledgeBase):
        """Initialize with user data, memory, and knowledge base"""
        super().__init__()
        self.user_data = user_data
        self.memory = memory
        self.knowledge_base = knowledge_base


class NHSAgent:
    """NHS virtual assistant for doctors and patients"""
    
    def __init__(self, user_data: UserData):
        # Store user data
        self.user_data = user_data
        
        # Initialize the structured file logger
        self.file_logger = StructuredFileLogger(user_data.user_id, user_data.user_type)
        
        # Initialize vector memory with user-specific collection
        collection_name = f"{user_data.user_type}_{user_data.user_id.replace('-', '_')}"
        self.memory = VectorMemory(collection_name, self.file_logger)
        
        # Set logger for vector memory
        self.memory.set_file_logger(self.file_logger)
        
        # Load appropriate knowledge base map based on user type
        if user_data.user_type == "patient":
            logger.info("Initializing patient knowledge base")
            self.knowledge_base = KnowledgeBase(None, self.file_logger)  # Will default to patient map
        else:
            logger.info("Initializing doctor knowledge base")
            self.knowledge_base = KnowledgeBase(None, self.file_logger)  # Will default to doctor map
        
        # Set logger for knowledge base
        self.knowledge_base.set_file_logger(self.file_logger)
        
        # Initialize function context
        self.function_ctx = NHSFunctions(user_data, self.memory, self.knowledge_base)
        
        # Initialize conversation history
        self.conversation_history = []
        self.last_user_message = ""
        
        # Keep track of sources used in last query for validation
        self.last_query_sources = []
        
        # Flag to track if knowledge was found for the current query
        self.knowledge_found = False
        self.current_query_type = None
        
        # Log initialization
        logger.info(f"NHS agent initialized for {user_data}")
        
        # Log initialization in the structured file
        asyncio.create_task(self.file_logger.log_event(
            "AGENT_INITIALIZED", 
            f"NHS agent initialized for {user_data}",
            {
                "user_type": user_data.user_type,
                "user_id": user_data.user_id,
                "memory_collection": collection_name,
                "knowledge_base_type": "PATIENT" if user_data.user_type == "patient" else "DOCTOR"
            }
        ))
    
    async def before_llm_callback(self, assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """
        Add relevant context to the LLM conversation
        
        This method enhances the conversation with retrieved knowledge, memory details,
        and appropriate guardrails for medical information.
        """
        try:
            # Get the user's query
            last_message = chat_ctx.messages[-1] if chat_ctx.messages else None
            if not last_message or last_message.role != "user" or not last_message.content:
                return
            
            query = last_message.content
            logger.info(f"Processing query: {query[:50]}...")
            
            # Store the message for later reference
            self.last_user_message = query
            
            # Analyze the type of query
            self.current_query_type = self._analyze_query_type(query)
            
            # Log query analysis
            await self.file_logger.log_event(
                "QUERY_ANALYSIS",
                f"Query: {query}\nType: {self.current_query_type}",
            )
            
            # For medical questions, get knowledge from vector database
            if self.current_query_type == "medical":
                logger.info("Medical question detected, retrieving knowledge...")
                
                # Log vector database query attempt
                await self.file_logger.log_event(
                    "KNOWLEDGE_SEARCH_START",
                    f"Starting knowledge search for: {query}",
                    {"query_type": self.current_query_type}
                )
                
                # Get comprehensive knowledge
                knowledge = self.knowledge_base.get_comprehensive_knowledge(query)
                
                # Update knowledge found flag
                self.knowledge_found = knowledge.get("knowledge_found", False)
                
                # Get sources for citation
                self.last_query_sources = knowledge.get("sources", [])
                
                # If knowledge was found, add it to the context
                if self.knowledge_found and knowledge.get("text"):
                    # Format source information for citation
                    sources = knowledge.get("sources", [])
                    source_text = ""
                    if sources:
                        source_lines = []
                        for i, source in enumerate(sources):
                            source_line = f"[{i+1}] {source.get('title', 'Unknown')} - {source.get('source', 'Unknown')} ({source.get('year', 'n.d.')})." + \
                                        f" Domain: {source.get('domain', 'General')}"
                            source_lines.append(source_line)
                        source_text = "\n".join(source_lines)
                    
                    # Format the knowledge text
                    knowledge_text = knowledge.get("text", "")
                    
                    # Add knowledge context to the conversation
                    context_message = f"""I found the following verified medical information to help address this query:\n\n{knowledge_text}\n\nSources:\n{source_text}"""
                    
                    # Add context to chat
                    chat_ctx.addContextWithName("Medical Knowledge", context_message)
                    
                    # Log the knowledge that was found
                    await self.file_logger.log_event(
                        "KNOWLEDGE_FOUND",
                        context_message,
                        {"source_count": len(sources)}
                    )
                else:
                    # No knowledge found, add warning to context
                    warning = """I don't have verified information from my knowledge base to answer this medical question.
                    Please provide a soft response indicating you don't have specific information on this topic,
                    and suggest the user consult a healthcare professional. DO NOT use your pre-trained medical 
                    knowledge to answer this question - only use the verified NHS sources that I provide to you."""
                    
                    chat_ctx.addContextWithName("Medical Knowledge Warning", warning)
                    
                    # Log that no knowledge was found
                    await self.file_logger.log_event(
                        "NO_KNOWLEDGE_FOUND",
                        f"No verified information found for: {query}",
                        {"query_type": self.current_query_type}
                    )
            
            # Add additional guardrails to the context
            guardrails = """
            IMPORTANT HEALTHCARE INFORMATION GUIDELINES:
            
            1. NEVER cite sources that were not explicitly provided to you in the context.
            2. NEVER make up or invent medical information or advice.
            3. NEVER provide information based on your pre-trained knowledge for medical questions.
            4. ONLY use the verified sources that were explicitly provided to you in the context.
            5. If no verified information was provided in the context, acknowledge the limitations
               and suggest consulting a healthcare professional.
            6. ALWAYS make it clear when you're using information from verified NHS sources versus
               general conversation.
            7. NEVER attribute information to sources like "NHS guidelines" or "medical literature" unless
               these exact sources were provided to you in the context.
            """
            
            chat_ctx.addContextWithName("Healthcare Guidelines", guardrails)
            
            logger.info("Finished setting up context for LLM response")
            
        except Exception as e:
            error_msg = f"Error in before_llm_callback: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
    
    def _analyze_query_type(self, query: str) -> str:
        """
        Analyze the type of query to determine if it's a medical question.
        
        Args:
            query: The user query
            
        Returns:
            Query type: 'medical', 'personal', or 'general'
        """
        # Convert to lowercase for analysis
        query_lower = query.lower()
        
        # Define medical keywords and phrases
        medical_keywords = [
            "symptom", "symptoms", "disease", "treatment", "diagnose", "diagnosis",
            "medicine", "drug", "prescription", "health", "medical", "condition",
            "pain", "doctor", "hospital", "surgery", "procedure", "test", "scan",
            "vaccination", "vaccine", "illness", "disorder", "syndrome", "infection",
            "dose", "dosage", "medication", "therapy", "side effect", "contraindication"
        ]
        
        # Define personal information keywords
        personal_keywords = [
            "appointment", "record", "my ", "schedule", "booking", "cancel",
            "reschedule", "visit", "my doctor", "my treatment", "my medication",
            "my prescription", "my condition", "personal", "family", "history"
        ]
        
        # Check for medical terms
        for keyword in medical_keywords:
            if keyword in query_lower or f"{keyword}s" in query_lower:
                return "medical"
        
        # Check for personal information terms
        for keyword in personal_keywords:
            if keyword in query_lower:
                return "personal"
        
        # Check for medical question patterns
        medical_patterns = [
            r"what (is|are) [\w\s]+(symptom|disease|condition|treatment|medication)",
            r"how (to|do I|can I) [\w\s]+(treat|manage|cure|prevent|diagnose)",
            r"(is|are) [\w\s]+ (normal|serious|concerning|dangerous|safe)",
            r"(should|can) I [\w\s]+(take|use|try|stop|start) [\w\s]+(medication|treatment|drug)",
            r"(what|when|how) [\w\s]+(dose|dosage|amount)"
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, query_lower):
                return "medical"
        
        # Default to general if no patterns match
        return "general"
    
    async def add_user_message(self, message: str):
        """Add a user message to the conversation history"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Log user query
            await self.file_logger.log_user_query(message)
            
            logger.info(f"Added user message to conversation history: {message[:50]}...")
            
        except Exception as e:
            error_msg = f"Error adding user message to history: {e}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
    
    async def add_agent_message(self, message: str):
        """Add an agent message to the conversation history"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Log agent response
            await self.file_logger.log_agent_response(message)
            
            logger.info(f"Added agent message to conversation history: {message[:50]}...")
            
        except Exception as e:
            error_msg = f"Error adding agent message to history: {e}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
    
    async def process_conversation(self):
        """Process the entire conversation history at the end of the session"""
        try:
            logger.info("Processing complete conversation...")
            
            # Skip if no conversation happened
            if not self.conversation_history:
                logger.info("No conversation to process")
                await self.file_logger.log_event("INFO", "No conversation to process")
                return
            
            # Extract key information from the conversation
            # For both doctors and patients, we'll save the full conversation 
            # but filter out any sensitive medical information
            
            # Format the conversation for storage
            conversation_text = ""
            for message in self.conversation_history:
                prefix = "User" if message["role"] == "user" else "Assistant"
                conversation_text += f"{prefix}: {message['text']}\n\n"
            
            # Log what's about to be stored in memory
            memory_preview = conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text
            await self.file_logger.log_event(
                "MEMORY_PROCESSING", 
                f"Processing conversation for memory storage:\n{memory_preview}",
                {
                    "message_count": len(self.conversation_history),
                    "first_message_time": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
                    "last_message_time": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
                }
            )
            
            # Try to store the full conversation in memory for context continuity
            try:
                # Extract potential memory topics
                topics = []
                for msg in self.conversation_history:
                    if msg["role"] == "user":
                        # Simple topic extraction - first few words
                        text = msg["text"].strip()
                        topic = " ".join(text.split()[:5]) + "..." if len(text.split()) > 5 else text
                        topics.append(topic)
                
                # Prepare metadata for memory storage
                memory_metadata = {
                        "type": "conversation_history", 
                        "user_id": self.user_data.user_id,
                        "user_type": self.user_data.user_type,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message_count": len(self.conversation_history),
                    "topics": topics[:3]  # Store up to 3 topics
                }
                
                # Log detailed memory metadata
                await self.file_logger.log_event(
                    "MEMORY_METADATA", 
                    f"Memory storage metadata:\n{json.dumps(memory_metadata, indent=2)}",
                    memory_metadata
                )
                
                # Save to memory
                success = self.memory.add_to_memory(
                    text=conversation_text,
                    metadata=memory_metadata
                )
                
                if success:
                    logger.info("Successfully stored conversation history in memory")
                    await self.file_logger.log_event(
                        "MEMORY_SAVE_SUCCESS", 
                        "Successfully stored conversation history in memory",
                        {"collection_name": self.memory.collection_name}
                    )
                else:
                    logger.warning("Failed to store conversation history in memory")
                    await self.file_logger.log_event("WARNING", "Failed to store conversation history in memory")
            except Exception as e:
                error_msg = f"Error storing conversation: {e}"
                logger.error(error_msg)
                await self.file_logger.log_event("ERROR", error_msg)
            
            # Calculate conversation duration
            start_time = None
            end_time = None
            
            if len(self.conversation_history) > 0:
                start_time = datetime.datetime.fromisoformat(self.conversation_history[0]["timestamp"])
                end_time = datetime.datetime.fromisoformat(self.conversation_history[-1]["timestamp"])
            
            # Log conversation summary
            if start_time and end_time:
                duration_sec = (end_time - start_time).total_seconds()
                await self.file_logger.log_conversation_summary(len(self.conversation_history), duration_sec)
            else:
                await self.file_logger.log_conversation_summary(len(self.conversation_history))
            
            logger.info("Conversation processing complete")
            
        except Exception as e:
            error_msg = f"Error processing conversation: {e}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
    
    async def send_chat_message(self, room: rtc.Room, message: str):
        """Send a chat message to all participants in the room"""
        try:
            # Create data packet for the chat message
            data = json.dumps({
                "message": message,
                "sender": "NHS Virtual Assistant",
                "timestamp": datetime.datetime.now().isoformat()
            }).encode("utf-8")
            
            # Send message to all participants
            await room.local_participant.publish_data(data, reliability=rtc.Reliability.RELIABLE, topic="chat")
            logger.info(f"Sent chat message: {message[:50]}...")
            
        except Exception as e:
            error_msg = f"Error sending chat message: {e}"
            logger.error(error_msg)
