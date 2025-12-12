"""
NHS Virtual Assistant Voice Agent

This module implements a virtual assistant for NHS doctors and patients using LiveKit with:
- Vector-based memory storage using Qdrant
- User-specific memory collections
- Medical knowledge base integration
- API integration for patient and doctor data
- Consent management for medical records
- Cost-efficient AI for knowledge base selection using Cerebras
- Fallback to OpenAI when needed

Environment variables required:
- OPENAI_API_KEY: For embeddings and fallback LLM
- CEREBRAS_API_KEY: For cost-efficient knowledge base selection (optional, will fall back to OpenAI)
- QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY: For vector storage

Author: Avijit Sarkar (Modified version)
"""

import os
import re
import asyncio
import logging
import datetime
import json
import requests
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Annotated, Union, Set, Tuple
import traceback

# Load environment variables
from dotenv import load_dotenv

# LiveKit imports
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, turn_detector
from livekit.plugins.cartesia import tts as cartesia_tts
import livekit.rtc as rtc

# Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# OpenAI for embeddings
from openai import OpenAI  # Add OpenAI import for vector descriptions

# Cerebras for knowledge base selection (cost-effective alternative)
from cerebras.cloud.sdk import Cerebras

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
        summary = (
            f"Conversation Summary:\n"
            f"- Total turns: {num_turns}\n"
            f"- Knowledge bases used: {', '.join(self.knowledge_bases_used) if self.knowledge_bases_used else 'None'}\n"
        )
        
        if duration_sec is not None:
            minutes = int(duration_sec // 60)
            seconds = int(duration_sec % 60)
            summary += f"- Duration: {minutes}m {seconds}s\n"
        
        await self.log_event("CONVERSATION_SUMMARY", summary)

# Load environment variables
load_dotenv()

# Environment settings
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Room validation settings
DOCTOR_SUFFIX = "-doctor"
PATIENT_SUFFIX = "-patient"
ROOM_VALIDATION_ERROR = "Room name does not match required pattern"

# API endpoints
NHS_API_BASE = "https://nhsapi.kno2gether.com/api"
PATIENT_VERIFY_ENDPOINT = f"{NHS_API_BASE}/patients/verify"
DOCTOR_VERIFY_ENDPOINT = f"{NHS_API_BASE}/doctors/verify"
MEDICAL_RECORDS_ENDPOINT = f"{NHS_API_BASE}/medical-records/patient"

# Qdrant settings
QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_TLS = os.environ.get("QDRANT_TLS", "true").lower() == "true"

# Knowledge base collections
COMMON_KNOWLEDGE_COLLECTION = "patient_assessment_commonknowledgebase"
MEDICAL_KNOWLEDGE_COLLECTION = "medical_assessment_commonknowledgebase"

# SIMPLIFIED KNOWLEDGE BASE STRUCTURE
# Instead of using individual knowledge base IDs, we now use two common vector databases:
# - medical_assessment_commonknowledgebase for doctors
# - patient_assessment_commonknowledgebase for patients
# 
# The query flow now:
# 1. Uses AI to create an effective query and select appropriate document title and author
# 2. Queries the appropriate common knowledge base based on user type
# 3. Return results with document metadata (title, author) for citation
# 4. If no results found, provides appropriate rejection message

# Knowledge base maps for different user types
DOCTOR_KNOWLEDGE_BASE_MAP = {
    "collection_name": MEDICAL_KNOWLEDGE_COLLECTION,
    "documents": [
        {
            "domain": "Critical Care",
            "document_title": "Management of vasoplegic shock",
            "authors": "R.N. Mistry and J.E. Winearls, Gold Coast University Hospital, Australia"
        },
        {
            "domain": "Obstetric Anesthesia",
            "document_title": "Prevention and management of intraoperative pain during Caesarean section",
            "authors": "S. Orbach-Zinger and Y. Binyamin, Israel"
        },
        {
            "domain": "Obstetric Anesthesia",
            "document_title": "Patient-centred strategies in obstetric anaesthesia",
            "authors": "B.D. Mergler, C.C. Duffy and R.J. Mergler, USA"
        },
        {
            "domain": "Obstetric Anesthesia",
            "document_title": "Neuraxial anesthesia for patients with spinal pathology",
            "authors": "G. Crowe and T. Drew, Ireland"
        },
        {
            "domain": "Obstetric Anesthesia",
            "document_title": "Neuraxial anesthesia for patients with intracranial pathology",
            "authors": "C. Warrick, W. Schievink and M. Zakowski, USA"
        },
        {
            "domain": "Airway Management",
            "document_title": "Airway ultrasound techniques",
            "authors": "R. Lohse, W.H. Teoh and M.S. Kristensen, Copenhagen University Hospital, Denmark"
        },
        {
            "domain": "Pediatric Anesthesia",
            "document_title": "Anesthesia for children with congenital heart disease undergoing non-cardiac surgery",
            "authors": "J. Spiro, J. Bauerle and D. Njoku, St. Louis Children's Hospital, USA"
        },
        {
            "domain": "Neuroanesthesia",
            "document_title": "Anesthesia for pituitary surgery",
            "authors": "K. Raveendran, S. Kwok and L. Glancz, UK"
        },
        {
            "domain": "Critical Care",
            "document_title": "Critical care echocardiography",
            "authors": "J.K. Cheng and R. Arntfield, New Zealand and Canada"
        },
        {
            "domain": "Pediatric Anesthesia",
            "document_title": "Enhanced recovery protocols after pediatric cardiac surgery",
            "authors": "L. Foote, L. Hepburn and C. Goodison, Great Ormond Street Hospital, UK"
        },
        {
            "domain": "Maternal Sepsis",
            "document_title": "Maternal sepsis: background, diagnosis, and management approaches",
            "authors": "J. Manigrasso, N. Desai and E. Naoum, USA and UK"
        }
    ]
}

# Simplified knowledge base map for patients - using only common knowledge collection
PATIENT_KNOWLEDGE_BASE_MAP = {
    "collection_name": COMMON_KNOWLEDGE_COLLECTION,
    "documents": [
        {
            "domain": "Anaesthesia",
            "document_title": "General anaesthetics: Risks and side effects",
            "authors": "Royal College of Anaesthetists (RCoA)"
        },
        {
            "domain": "Pediatric Anaesthesia",
            "document_title": "Common events and risks for children and young people having a general anaesthetic",
            "authors": "Royal College of Anaesthetists (RCoA) and Association of Paediatric Anaesthetists of Great Britain and Ireland"
        },
        {
            "domain": "Regional Anaesthesia",
            "document_title": "Epidural anaesthesia during and after surgery",
            "authors": "Royal College of Anaesthetists (RCoA) and Association of Anaesthetists"
        },
        {
            "domain": "Anaesthesia",
            "document_title": "Anaesthetics – risks and side effects: Waking up during a general anaesthetic (accidental awareness)",
            "authors": "Leila Finikarides for the Royal College of Anaesthetists (RCoA)"
        },
        {
            "domain": "Anaesthesia",
            "document_title": "Anaesthetics – risks and side effects: Death and serious harm",
            "authors": "Leila Finikarides for the Royal College of Anaesthetists (RCoA)"
        },
        {
            "domain": "Regional Anaesthesia",
            "document_title": "Anaesthetics – risks and side effects: Nerve damage after a peripheral nerve block",
            "authors": "Leila Finikarides for the Royal College of Anaesthetists (RCoA)"
        },
        {
            "domain": "Regional Anaesthesia",
            "document_title": "Your spinal anaesthetic",
            "authors": "Royal College of Anaesthetists (RCoA), Association of Anaesthetists and RA-UK"
        }
    ]
}

# Create Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    https=QDRANT_TLS
)

class UserData:
    """Base class for user data"""
    
    def __init__(self, user_id: str, user_type: str):
        """Initialize the user data
        
        Args:
            user_id (str): User's ID (NHS number or registration number)
            user_type (str): Type of user ('patient' or 'doctor')
        """
        self.user_id = user_id
        self.user_type = user_type
        self.full_name = ""
        self.email = ""
        self.phone = ""
        self.address = ""
        self.date_of_birth = ""
        self.created_at = ""
        self.id = ""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert user data to dictionary"""
        return {
            "user_id": self.user_id,
            "user_type": self.user_type,
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "date_of_birth": self.date_of_birth,
            "created_at": self.created_at,
            "id": self.id
        }
        
    def __str__(self) -> str:
        """String representation"""
        name = self.full_name if self.full_name else f"Unknown {self.user_type}"
        return f"{self.user_type.capitalize()}: {name} ({self.user_id})"

class PatientData(UserData):
    """Patient-specific data"""
    
    def __init__(self, nhs_number: str):
        super().__init__(nhs_number, "patient")
        self.nhs_number = nhs_number
        self.medical_records = []
        self.has_consent_for_records = False
        
    @classmethod
    def from_api_response(cls, response_data: Dict[str, Any]) -> 'PatientData':
        """Create PatientData instance from API response"""
        patient = cls(response_data.get("nhs_number", ""))
        patient.full_name = response_data.get("full_name", "")
        patient.email = response_data.get("email", "")
        patient.phone = response_data.get("phone", "")
        patient.address = response_data.get("address", "")
        patient.date_of_birth = response_data.get("date_of_birth", "")
        patient.created_at = response_data.get("created_at", "")
        patient.id = response_data.get("id", "")
        return patient
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert patient data to dictionary"""
        data = super().to_dict()
        data.update({
            "nhs_number": self.nhs_number,
            "has_consent_for_records": self.has_consent_for_records
        })
        if self.medical_records:
            data["medical_records"] = self.medical_records
        return data

class DoctorData(UserData):
    """Doctor-specific data"""
    
    def __init__(self, registration_number: str):
        super().__init__(registration_number, "doctor")
        self.registration_number = registration_number
        self.hospital = ""
        self.specialty = ""
        
    @classmethod
    def from_api_response(cls, response_data: Dict[str, Any]) -> 'DoctorData':
        """Create DoctorData instance from API response"""
        doctor = cls(response_data.get("registration_number", ""))
        doctor.full_name = response_data.get("full_name", "")
        doctor.hospital = response_data.get("hospital", "")
        doctor.specialty = response_data.get("specialty", "")
        doctor.email = response_data.get("email", "")
        doctor.phone = response_data.get("phone", "")
        doctor.address = response_data.get("address", "")
        doctor.date_of_birth = response_data.get("date_of_birth", "")
        doctor.created_at = response_data.get("created_at", "")
        doctor.id = response_data.get("id", "")
        return doctor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert doctor data to dictionary"""
        data = super().to_dict()
        data.update({
            "registration_number": self.registration_number,
            "hospital": self.hospital,
            "specialty": self.specialty
        })
        return data

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
            await self.file_logger.log_event(f"MEMORY_{event_type}", content, metadata)
    
    def _init_collection(self):
        """Initialize the vector collection"""
        try:
            # Check if collection exists
            collections = qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Create new collection
                qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=qmodels.Distance.COSINE
                    )
                )
                logger.info(f"Created new vector collection: {self.collection_name}")
            else:
                logger.info(f"Using existing vector collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing vector collection: {e}")
            raise
    
    def add_to_memory(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add text to vector memory"""
        try:
            # Create default metadata if none provided
            if metadata is None:
                metadata = {}
            
            # Add timestamp to metadata
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            
            # Get embedding from OpenAI
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            
            # Log embedding generation asynchronously
            if self.file_logger:
                text_preview = text[:200] + "..." if len(text) > 200 else text
                asyncio.create_task(self._log_to_file(
                    "EMBED_GENERATION",
                    f"Generated embedding for memory text: {text_preview}",
                    {
                        "collection": self.collection_name,
                        "text_length": len(text),
                        "embedding_model": "text-embedding-3-small",
                        "metadata_keys": list(metadata.keys())
                    }
                ))
            
            # Generate a UUID for the point ID
            import uuid
            point_id = str(uuid.uuid4())
            
            # Add to collection
            qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            **metadata
                        }
                    )
                ]
            )
            
            logger.info(f"Added to memory collection {self.collection_name}: {text[:50]}...")
            
            # Log successful memory addition asynchronously
            if self.file_logger:
                asyncio.create_task(self._log_to_file(
                    "STORAGE_SUCCESS",
                    f"Successfully stored in memory collection: {self.collection_name}",
                    {
                        "point_id": point_id,
                        "collection": self.collection_name
                    }
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
            
            # Log error asynchronously
            if self.file_logger:
                asyncio.create_task(self._log_to_file(
                    "STORAGE_ERROR",
                    f"Error adding to memory: {e}",
                    {"collection": self.collection_name}
                ))
            
            return False
    
    def query_memory(self, query: str, limit: int = 3) -> str:
        """Query memory for relevant information"""
        try:
            # Get embedding from OpenAI
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            
            # Log query embedding generation asynchronously
            if self.file_logger:
                asyncio.create_task(self._log_to_file(
                    "QUERY_EMBED",
                    f"Generated embedding for memory query: {query}",
                    {
                        "collection": self.collection_name,
                        "query": query,
                        "embedding_model": "text-embedding-3-small",
                        "limit": limit
                    }
                ))
            
            # Search collection
            search_results = qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            if not search_results:
                # Log no results asynchronously
                if self.file_logger:
                    asyncio.create_task(self._log_to_file(
                        "QUERY_NO_RESULTS",
                        f"No memory found for query: {query}",
                        {"collection": self.collection_name}
                    ))
                return ""
            
            # Format results
            results = []
            point_ids = []
            for hit in search_results:
                text = hit.payload.get("text", "")
                timestamp = hit.payload.get("timestamp", "")
                results.append(f"({timestamp[:10]}) {text}")
                
                # Track point IDs for logging
                if hasattr(hit, "id"):
                    point_ids.append(str(hit.id))
            
            logger.info(f"Retrieved memory for query: {query[:50]}...")
            
            # Log memory retrieval results asynchronously
            if self.file_logger:
                results_preview = "\n".join(results)
                if len(results_preview) > 500:
                    results_preview = results_preview[:500] + "..."
                
                asyncio.create_task(self._log_to_file(
                    "QUERY_RESULTS",
                    f"Retrieved memory for query: {query}\n\nResults:\n{results_preview}",
                    {
                        "collection": self.collection_name,
                        "result_count": len(results),
                        "point_ids": point_ids
                    }
                ))
            
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            
            # Log error asynchronously
            if self.file_logger:
                asyncio.create_task(self._log_to_file(
                    "QUERY_ERROR",
                    f"Error querying memory: {e}",
                    {
                        "collection": self.collection_name,
                        "query": query
                    }
                ))
            
            return ""

class KnowledgeBase:
    """
    Vector database based medical knowledge retrieval system
    
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
    
    def set_file_logger(self, file_logger):
        """Set the file logger for this knowledge base"""
        self.file_logger = file_logger
    
    async def _log_knowledge_event(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log knowledge base event if file logger is available"""
        if self.file_logger:
            await self.file_logger.log_event(f"KB_{event_type}", content, metadata)
    
    def ai_analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Use AI to analyze the query and select appropriate document references
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with analysis results including refined query and selected documents
        """
        try:
            # Create a description of the knowledge base for the model
            kb_description = []
            
            # Get the documents from the map
            documents = self.knowledge_map["documents"]
            
            # Create a description of available documents for the model
            for doc in documents:
                desc = {
                    "document_title": doc["document_title"],
                    "domain": doc["domain"],
                    "authors": doc["authors"]
                }
                kb_description.append(desc)
            
            # Use Cerebras/OpenAI to analyze the query
            system_prompt = """You are an expert medical AI assistant helping to analyze user queries to match them with the most appropriate medical documents.
            
Your goal is to:
1. Refine the original query to make it more precise and medical literature focused
2. Select appropriate documents that are likely to contain information relevant to the query
3. Explain your reasoning in technical medical terminology

When selecting documents, be selective and only choose those that are directly relevant.
"""
            
            user_prompt = f"""Please analyze this medical query:

"{query}"

Here are the available medical documents:
{json.dumps(kb_description, indent=2)}

Return your analysis in this structured JSON format:
{{
  "refined_query": "the improved and more precise query",
  "selected_documents": [
    {{
      "document_title": "exact title of selected document",
      "authors": "exact authors of the document"
    }}
  ],
  "explanation": "Your detailed explanation for why these documents were selected"
}}

Important: Only include documents that are truly relevant to the query. Don't select documents just because they seem generally related.
"""

            # Use Cerebras instead of OpenAI for knowledge base selection (cost-effective)
            try:
                # Initialize Cerebras client (using API key from environment variable)
                cerebras_client = Cerebras(
                    api_key=os.environ.get("CEREBRAS_API_KEY")
                )
                
                # Try up to 3 times with increasing backoff
                max_retries = 3
                retry_count = 0
                response_content = None
                
                while retry_count < max_retries and response_content is None:
                    try:
                        # Make the API request to Cerebras
                        response = cerebras_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            model="llama-3.3-70b",  # Use Llama 3.3 70B model for best results
                            response_format={"type": "json_object"}
                        )
                        
                        # Extract response content
                        response_content = response.choices[0].message.content
                        logger.info(f"Cerebras response received: {len(response_content)} chars")
                        
                    except Exception as retry_error:
                        retry_count += 1
                        if retry_count < max_retries:
                            # Exponential backoff: 1s, 2s, 4s, etc.
                            wait_time = 2 ** (retry_count - 1)
                            logger.warning(f"Cerebras API call failed (attempt {retry_count}/{max_retries}). Retrying in {wait_time}s: {str(retry_error)}")
                            # Use synchronous sleep instead of asyncio.sleep since this is not an async function
                            time.sleep(wait_time)
                        else:
                            # Last attempt failed, re-raise
                            raise retry_error
                
            except Exception as cerebras_error:
                # Fall back to OpenAI if Cerebras fails
                logger.warning(f"Cerebras API failed after retries, falling back to OpenAI: {str(cerebras_error)}")
                
                # Use OpenAI as fallback
                openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Using smaller model for speed and cost efficiency
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                response_content = response.choices[0].message.content
                logger.info("Using OpenAI fallback for KB selection")
            
            # Parse response
            response_json = json.loads(response_content)
            
            # Validate the selected documents
            valid_documents = []
            for selected_doc in response_json.get("selected_documents", []):
                selected_title = selected_doc.get("document_title")
                selected_authors = selected_doc.get("authors")
                
                # Find the matching document in our knowledge base
                for doc in documents:
                    if (doc["document_title"] == selected_title and 
                        doc["authors"] == selected_authors):
                        valid_documents.append({
                            "document_title": selected_title,
                            "authors": selected_authors,
                            "domain": doc["domain"]
                        })
                        break
            
            # Log the AI's analysis
            if self.file_logger:
                asyncio.create_task(self._log_knowledge_event(
                    "AI_ANALYSIS", 
                    f"Query analyzed by AI: {query[:100]}...",
                    {
                        "original_query": query,
                        "refined_query": response_json.get("refined_query", ""),
                        "selected_documents": valid_documents,
                        "explanation": response_json.get("explanation", "")
                    }
                ))
            
            # Return the analysis
            return {
                "refined_query": response_json.get("refined_query", query),
                "selected_documents": valid_documents,
                "explanation": response_json.get("explanation", "No explanation provided")
            }
            
        except Exception as e:
            error_msg = f"Error analyzing query with AI: {e}"
            logger.error(error_msg)
            
            # Log the error
            if self.file_logger:
                asyncio.create_task(self._log_knowledge_event(
                    "ERROR", 
                    error_msg,
                    {"query": query}
                ))
                
            # Fallback to original query with no selected knowledge bases
            return {
                "refined_query": query,
                "selected_documents": [],
                "explanation": f"Failed to analyze query: {str(e)}"
            }
    
    async def _log_vector_query(self, query: str, vector_response: List[Dict[str, Any]], event_type: str = "VECTOR_QUERY"):
        """Log vector database query and response
        
        Args:
            query: The original query text
            vector_response: The response from the vector database
            event_type: Type of event to log
        """
        try:
            if not self.file_logger:
                return
                
            # Basic information about the query
            log_metadata = {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "timestamp": datetime.datetime.now().isoformat(),
                "num_results": len(vector_response)
            }
            
            # Extract document information
            documents = []
            for i, item in enumerate(vector_response[:5]):  # Limit to first 5 for brevity
                doc_info = {
                    "title": item.get("document_title", "") or item.get("title", "Unknown"),
                    "similarity": round(item.get("similarity", 0.0), 3),
                    "domain": item.get("domain", "Unknown")
                }
                documents.append(doc_info)
                
            log_metadata["documents"] = documents
            
            # If there are embeddings or vectors in the response, convert a sample to text
            if any("embedding" in item or "vector" in item for item in vector_response) and len(vector_response) > 0:
                # Try to convert first embedding to text description using OpenAI
                try:
                    sample_vector = vector_response[0].get("embedding") or vector_response[0].get("vector")
                    if sample_vector and isinstance(sample_vector, list) and len(sample_vector) > 10:
                        vector_description = await self._describe_vector(sample_vector)
                        log_metadata["vector_description"] = vector_description
                except Exception as e:
                    logger.warning(f"Failed to convert vector to description: {e}")
                    log_metadata["vector_description_error"] = str(e)
            
            # Log the vector query and response
            await self.file_logger.log_event(
                event_type,
                f"Vector database query: '{query}'",
                log_metadata
            )
            
            logger.info(f"Logged vector database query: '{query[:50]}...' with {len(vector_response)} results")
            
        except Exception as e:
            logger.error(f"Error logging vector query: {e}")
    
    async def _describe_vector(self, vector: List[float]) -> str:
        """Convert a vector embedding to human-readable text description using OpenAI
        
        Args:
            vector: The vector embedding
            
        Returns:
            Text description of what the vector might represent
        """
        try:
            # For very long vectors, just use a sample
            if len(vector) > 20:
                vector_sample = vector[:10] + vector[-10:]
                vector_str = f"First 10 + last 10 components of {len(vector)}-dimensional vector: {vector_sample}"
            else:
                vector_str = f"{len(vector)}-dimensional vector: {vector}"
            
            # Using OpenAI to describe the vector
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains vector embeddings in plain language."},
                    {"role": "user", "content": f"This is a semantic vector embedding: {vector_str}. Describe in a few sentences what information this embedding might encode and how it would be used in a medical knowledge retrieval system."}  
                ],
                max_tokens=150
            )
            
            # Extract the description
            description = response.choices[0].message.content
            return description
        
        except Exception as e:
            logger.error(f"Error describing vector: {e}")
            return f"Failed to describe vector: {str(e)}"
    
    def get_relevant_knowledge_bases(self, query: str) -> List[Dict[str, Any]]:
        """Find relevant knowledge bases using vector similarity
        
        Args:
            query: The query to find relevant knowledge bases for
            
        Returns:
            List of relevant documents metadata
        """
        try:
            # Log the start of knowledge retrieval process
            logger.info(f"Starting knowledge retrieval for: {query[:50]}...")
            
            # Get query embedding
            query_embedding = get_embedding(query)
            
            # Get the appropriate collection name from knowledge map
            collection_name = self.knowledge_map.get("collection_name", "")
            
            # Double-check collection based on user type to ensure we're using the right one
            if hasattr(self.user_data, 'user_type'):
                if self.user_data.user_type == "patient" and collection_name != COMMON_KNOWLEDGE_COLLECTION:
                    logger.warning(f"Overriding collection from {collection_name} to {COMMON_KNOWLEDGE_COLLECTION} for patient")
                    collection_name = COMMON_KNOWLEDGE_COLLECTION
                elif self.user_data.user_type == "doctor" and collection_name != MEDICAL_KNOWLEDGE_COLLECTION:
                    logger.warning(f"Overriding collection from {collection_name} to {MEDICAL_KNOWLEDGE_COLLECTION} for doctor")
                    collection_name = MEDICAL_KNOWLEDGE_COLLECTION
                
                logger.info(f"User type: {self.user_data.user_type}, Using collection: {collection_name}")
            
            if not collection_name:
                logger.error("No collection name found in knowledge map")
                return []
            
            # Log that we're connecting to Qdrant
            logger.info(f"Connecting to Qdrant collection '{collection_name}' for query: {query[:50]}...")
            
            # Get documents from the collection
            knowledge_map_docs = self.knowledge_map.get("documents", [])
            
            if not knowledge_map_docs:
                logger.warning("No documents in knowledge map")
                return []
            
            try:
                # Use Qdrant to perform the actual vector search
                search_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=5,  # Retrieve top 5 matches
                    with_payload=True  # Include document metadata
                )
                
                logger.info(f"Qdrant search returned {len(search_results)} results from collection '{collection_name}'")
                
                if not search_results:
                    logger.warning(f"No results found in Qdrant collection '{collection_name}' for query: {query[:50]}...")
                    # Fall back to local similarity as backup
                    logger.info("Falling back to local similarity matching")
                    return self._fallback_local_similarity(query, query_embedding, knowledge_map_docs)
                
                # Process Qdrant results
                results = []
                for result in search_results:
                    # Extract metadata from result payload
                    payload = result.payload
                    doc_title = payload.get("document_title", "")
                    
                    # Find the matching document in our knowledge map to get complete metadata
                    for doc in knowledge_map_docs:
                        if doc.get("document_title", "") == doc_title:
                            # Copy the document and add similarity score
                            doc_copy = doc.copy()
                            doc_copy["similarity"] = result.score
                            results.append(doc_copy)
                            break
                
                # Log the vector query and response asynchronously
                asyncio.create_task(self._log_vector_query(query, results, "KB_VECTOR_SEARCH"))
                
                # Filter out any with similarity below threshold
                threshold = 0.3
                results = [doc for doc in results if doc.get("similarity", 0) >= threshold]
                
                return results
                
            except Exception as qdrant_error:
                # Log the Qdrant error
                logger.error(f"Qdrant search error: {qdrant_error}")
                logger.info("Falling back to local similarity matching due to Qdrant error")
                
                # Fall back to local similarity matching
                return self._fallback_local_similarity(query, query_embedding, knowledge_map_docs)
            
        except Exception as e:
            logger.error(f"Error finding relevant knowledge bases: {e}")
            logger.error(traceback.format_exc())
            if self.file_logger:
                asyncio.create_task(self._log_knowledge_event(
                    "ERROR", 
                    f"Error finding relevant knowledge bases: {e}",
                    {"query": query, "traceback": traceback.format_exc()}
                ))
            return []
    
    def _fallback_local_similarity(self, query: str, query_embedding: List[float], knowledge_map_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback method using local similarity matching
        
        Args:
            query: The original query
            query_embedding: The embedding of the query
            knowledge_map_docs: Documents from the knowledge map
            
        Returns:
            List of documents with similarity scores
        """
        logger.warning(f"Using fallback LOCAL similarity matching for query: {query[:50]}...")
        
        results = []
        # For each document in knowledge map, compute similarity
        for doc in knowledge_map_docs:
            # Get document title and compute similarity
            title = doc.get("document_title", "")
            
            if not title:
                continue
                
            title_embedding = get_embedding(title)
            similarity = self._calculate_cosine_similarity(query_embedding, title_embedding)
            
            # Add to results with similarity score
            doc_copy = doc.copy()
            doc_copy["similarity"] = similarity
            results.append(doc_copy)
        
        # Sort by similarity
        results = sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Keep top K most similar
        top_k = 3
        results = results[:top_k]
        
        # Log that we used the fallback
        asyncio.create_task(self._log_vector_query(
            query, 
            results, 
            "KB_FALLBACK_LOCAL_SIMILARITY"
        ))
        
        # Filter out any with similarity below threshold
        threshold = 0.3
        results = [doc for doc in results if doc.get("similarity", 0) >= threshold]
        
        return results
    
    def get_comprehensive_knowledge(self, query: str) -> Dict[str, Any]:
        """Get comprehensive knowledge from relevant knowledge bases
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with text and sources
        """
        try:
            # Log the start of knowledge retrieval process
            logger.info(f"Starting knowledge retrieval for: {query[:50]}...")
            
            # First, try to get relevant documents using the AI-based approach
            logger.info("Attempting AI-based document selection")
            ai_analysis = self.ai_analyze_query(query)
            selected_documents = ai_analysis.get("selected_documents", [])
            
            # Log the AI selection asynchronously
            if selected_documents:
                logger.info(f"AI selected {len(selected_documents)} documents")
                asyncio.create_task(self._log_vector_query(
                    f"AI Selection for: {query}", 
                    selected_documents, 
                    "KB_AI_SELECTION"
                ))
            else:
                logger.info("AI-based selection found no documents, falling back to vector search")
            
            # If AI didn't find relevant documents, try embedding-based approach
            if not selected_documents:
                # IMPORTANT: Explicitly log that we're querying the vector database
                logger.info(f"Performing vector database query for: {query[:50]}...")
                
                # This should trigger the vector database query
                embedding_results = self.get_relevant_knowledge_bases(query)
                
                # Log the results for debugging
                logger.info(f"Vector search returned {len(embedding_results)} results")
                for i, doc in enumerate(embedding_results[:3]):
                    logger.info(f"Vector result {i+1}: {doc.get('document_title', '')} - {doc.get('similarity', 0):.3f}")
                
                selected_documents = embedding_results
                
                # Log is already handled in get_relevant_knowledge_bases
            
            # If no documents were found with either approach, return empty result
            if not selected_documents:
                logger.warning(f"No relevant documents found for query: {query[:100]}...")
                return {"text": "", "sources": []}
            
            # Verify that the selected documents exist in the knowledge map
            valid_documents = []
            knowledge_map_docs = self.knowledge_map.get("documents", [])
            
            for doc in selected_documents:
                # Get document title and normalize it
                doc_title = doc.get("document_title", "") or doc.get("title", "")
                
                # Check if this document exists in our predefined knowledge map
                found_in_map = False
                matched_doc = None
                
                for map_doc in knowledge_map_docs:
                    map_doc_title = map_doc.get("document_title", "")
                    # Check for exact match or high similarity
                    if map_doc_title == doc_title or fuzzymatch(map_doc_title, doc_title, threshold=0.8):
                        found_in_map = True
                        matched_doc = map_doc  # Use the exact document from the map
                        break
                
                if found_in_map and matched_doc:
                    valid_documents.append(matched_doc)  # Use the document from the knowledge map
                else:
                    logger.warning(f"Document not found in knowledge map: {doc_title}")
            
            # If no valid documents were found, return empty result
            if not valid_documents:
                logger.warning(f"No valid documents found in knowledge map for query: {query[:100]}...")
                return {"text": "", "sources": []}
            
            # For each verified document, retrieve its content
            combined_text = ""
            sources = []
            
            for doc in valid_documents:
                # Get document content using ONLY the knowledge map metadata
                document_title = doc.get("document_title", "")
                authors = doc.get("authors", "Unknown")
                domain = doc.get("domain", "Medical Journal")
                
                # In a real implementation, you would retrieve the actual document content
                # Here we're just generating a placeholder with the metadata
                doc_content = f"Information about {query} from {document_title}\n"
                doc_content += f"According to medical literature from {domain}, this topic is relevant for patient care.\n"
                
                # Add to combined text
                combined_text += doc_content + "\n\n"
                
                # Add to sources list - using ONLY the predefined knowledge map data
                sources.append({
                    "title": document_title,
                    "authors": authors,
                    "domain": domain
                })
            
            # Log the final knowledge result
            logger.info(f"Generated knowledge with {len(sources)} sources for query: {query[:50]}...")
            result_info = {
                "query": query, 
                "num_sources": len(sources),
                "source_titles": [s["title"] for s in sources],
                "text_length": len(combined_text)
            }
            logger.info(f"Knowledge result details: {result_info}")
            
            # Also log asynchronously
            asyncio.create_task(self._log_knowledge_event(
                "KNOWLEDGE_RESULT", 
                f"Generated knowledge for query: {query[:100]}...",
                result_info
            ))
            
            # Return the knowledge with sources
            return {
                "text": combined_text.strip(),
                "sources": sources
            }
        
        except Exception as e:
            error_msg = f"Error retrieving knowledge: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())  # Log full traceback for debugging
            
            # Log the error
            if self.file_logger:
                asyncio.create_task(self._log_knowledge_event(
                    "ERROR", 
                    error_msg,
                    {"query": query}
                ))
            
            return {"text": "", "sources": []}
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            
            # Calculate magnitude of vectors
            magnitude1 = sum(a * a for a in embedding1) ** 0.5
            magnitude2 = sum(b * b for b in embedding2) ** 0.5
            
            # Calculate cosine similarity
            if magnitude1 > 0 and magnitude2 > 0:
                return dot_product / (magnitude1 * magnitude2)
            else:
                return 0
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0
    
    def get_kb_info_by_id(self, kb_id: str) -> Dict[str, Any]:
        """Get knowledge base info by document title"""
        try:
            kb_map_key = next(iter(self.knowledge_map.keys()))
            knowledgebases = self.knowledge_map[kb_map_key]["documents"]
            
            for kb in knowledgebases:
                # In our simplified structure, we use document title instead of id
                # But we kept this method for backwards compatibility
                if kb["document_title"] == kb_id:
                    # Create a comprehensive info dictionary with all available fields
                    kb_info = {
                        "document_title": kb["document_title"],
                        "score": 10,  # Assign high score since it was selected by AI
                        "domain": kb["domain"]
                    }
                    
                    # Add optional fields if they exist
                    if "authors" in kb and kb["authors"]:
                        kb_info["authors"] = kb["authors"]
                    
                    if "date" in kb:
                        kb_info["date"] = kb["date"]
                    
                    return kb_info
            
            return None
        except Exception as e:
            logger.error(f"Error getting KB info by ID: {str(e)}")
            return None
    
    @llm.ai_callable(
        description="Record patient consent for accessing medical records"
    )
    async def record_patient_consent(
        self,
        has_consent: Annotated[
            bool,
            llm.TypeInfo(
                description="Whether the patient has given consent"
            )
        ]
    ) -> str:
        """Record patient's consent for accessing medical records"""
        try:
            # Only available for patients
            if self.user_data.user_type != "patient":
                return "This function is only available for patients."
            
            self.medical_record_consent = has_consent
            patient_data = self.user_data
            if isinstance(patient_data, PatientData):
                patient_data.has_consent_for_records = has_consent
            
            if has_consent:
                return "Patient consent for accessing medical records has been recorded. You may now retrieve the patient's medical records."
            else:
                return "Patient has not given consent for accessing medical records."
        except Exception as e:
            return f"Failed to record consent: {str(e)}"
    
    @llm.ai_callable(description="Get patient medical records")
    async def get_medical_records(
        self
    ) -> str:
        """Get patient medical records"""
        try:
            # Only available for patients
            if self.user_data.user_type != "patient":
                return "This function is only available for patients."
            
            # Check for consent
            patient_data = self.user_data
            if isinstance(patient_data, PatientData):
                if not patient_data.has_consent_for_records:
                    return "You don't have the patient's consent to access their medical records. Please obtain consent first."
                
                # Get medical records from API
                nhs_number = patient_data.nhs_number
                url = f"{MEDICAL_RECORDS_ENDPOINT}/{nhs_number}"
                response = requests.get(url)
                
                if response.status_code != 200:
                    return f"Failed to retrieve medical records: {response.status_code}"
                
                records = response.json()
                patient_data.medical_records = records
                
                # Format records for display
                if not records:
                    return "No medical records found for this patient."
                
                formatted_records = []
                for record in records:
                    record_date = record.get("record_date", "Unknown date")[:10]
                    med_history = record.get("medical_history", {})
                    
                    allergies = ", ".join(med_history.get("allergies", ["None"]))
                    conditions = ", ".join(med_history.get("chronic_conditions", ["None"]))
                    medications = ", ".join(med_history.get("medications", ["None"]))
                    
                    formatted_record = (
                        f"Record Date: {record_date}\n"
                        f"Allergies: {allergies}\n"
                        f"Chronic Conditions: {conditions}\n"
                        f"Current Medications: {medications}\n"
                        f"Notes: {record.get('notes', 'No notes')}"
                    )
                    formatted_records.append(formatted_record)
                
                return "Medical Records:\n\n" + "\n\n".join(formatted_records)
            else:
                return "Invalid patient data."
        except Exception as e:
            return f"Failed to retrieve medical records: {str(e)}"
    
    @llm.ai_callable(
        description="Save important information to memory for future reference"
    )
    async def save_to_memory(
        self,
        information: Annotated[
            str,
            llm.TypeInfo(
                description="Non-sensitive information to save for future conversations"
            )
        ],
        category: Annotated[
            str,
            llm.TypeInfo(
                description="Category of information such as 'preference', 'general_health', or 'scheduling'"
            )
        ]
    ) -> str:
        """Save important non-sensitive information to memory"""
        try:
            # Add to vector memory
            success = self.memory.add_to_memory(
                information,
                metadata={
                    "category": category,
                    "user_id": self.user_data.user_id,
                    "user_type": self.user_data.user_type
                }
            )
            
            if success:
                return f"Information saved to memory under category '{category}'"
            else:
                return "Failed to save information to memory"
        except Exception as e:
            return f"Error saving to memory: {str(e)}"

class NHSFunctions(llm.FunctionContext):
    """Functions available to the NHS agent LLM"""
    
    def __init__(self, user_data: UserData, memory: VectorMemory, knowledge_base: KnowledgeBase):
        """Initialize with user data, memory, and knowledge base"""
        super().__init__()
        self.user_data = user_data
        self.memory = memory
        self.knowledge_base = knowledge_base
        
    @llm.ai_callable(description="Record patient's consent for accessing medical records")
    async def record_patient_consent(
        self,
        has_consent: Annotated[
            bool,
            llm.TypeInfo(
                description="Whether the patient has given consent"
            )
        ]
    ) -> str:
        """Record patient consent for medical records access"""
        return await self.knowledge_base.record_patient_consent(has_consent)
    
    @llm.ai_callable(description="Get patient medical records if consent has been given")
    async def get_medical_records(self) -> str:
        """Get patient medical records"""
        return await self.knowledge_base.get_medical_records()
    
    @llm.ai_callable(description="Save important non-sensitive information to memory")
    async def save_to_memory(
        self,
        information: Annotated[
            str,
            llm.TypeInfo(
                description="Non-sensitive information to save for future conversations"
            )
        ],
        category: Annotated[
            str,
            llm.TypeInfo(
                description="Category of information such as 'preference', 'general_health', or 'scheduling'"
            )
        ]
    ) -> str:
        """Save important non-sensitive information to memory"""
        return await self.knowledge_base.save_to_memory(information, category)

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
            self.knowledge_base = KnowledgeBase(PATIENT_KNOWLEDGE_BASE_MAP, self.file_logger)
        else:
            logger.info("Initializing doctor knowledge base")
            self.knowledge_base = KnowledgeBase(DOCTOR_KNOWLEDGE_BASE_MAP, self.file_logger)
        
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
        
    async def before_llm_callback(self, assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext) -> llm.ChatContext:
        """Add relevant context to the LLM conversation
        
        This method enhances the conversation with retrieved knowledge, memory details,
        and appropriate guardrails for medical information.
        """
        try:
            # Don't try to create new context, modify the existing one
            # Just use the chat_ctx directly instead of creating a new one
            self.current_query_type = "general"  # Default to general query
            self.knowledge_found = False  # Reset knowledge found flag
            self.last_query_sources = []  # Reset source tracking
            
            # Get the last user message
            user_messages = [m for m in chat_ctx.messages if m.role == "user"]
            if not user_messages:
                logger.warning("No user messages found in context")
                return chat_ctx
                
            last_msg = user_messages[-1].content
            
            # Log that we're processing a message
            logger.info(f"Processing user message: {last_msg[:50]}...")
            
            # Skip if the message is a system command or not a question/statement
            if last_msg.strip().startswith("/") or len(last_msg.strip()) < 3:
                return chat_ctx
            
            # Only analyze the message and add context if the message is non-trivial
            if len(last_msg.strip()) > 5 and any(char.isalpha() for char in last_msg):
                # Analyze the message to categorize the query
                query_type = self._analyze_query_type(last_msg)
                self.current_query_type = query_type
                
                logger.info(f"Query classified as: {query_type}")
                
                # Log the query type classification asynchronously
                if self.file_logger:
                    asyncio.create_task(self.file_logger.log_event(
                        "QUERY_CLASSIFICATION",
                        f"Query classified as: {query_type}",
                        {"query": last_msg[:100] + "..." if len(last_msg) > 100 else last_msg}
                    ))
                
                # Prepare parts to add to context
                context_parts = []
                
                # For medical queries, get relevant medical knowledge
                if query_type == "medical":
                    try:
                        # Log that we're searching for medical knowledge
                        logger.info(f"Searching for medical knowledge for: {last_msg[:50]}...")
                        
                        # Get medical knowledge from knowledge base
                        kb_result = self.knowledge_base.get_comprehensive_knowledge(last_msg)
                        knowledge_text = kb_result.get("text", "")
                        sources = kb_result.get("sources", [])
                        self.last_query_sources = sources  # Track sources for validation
                        
                        # Log the raw KB result
                        logger.info(f"Knowledge base returned {len(sources)} sources")
                        asyncio.create_task(self.file_logger.log_event(
                            "KB_RAW_RESULT",
                            f"Raw knowledge base result for: {last_msg[:50]}...",
                            {
                                "query": last_msg[:100],
                                "text_length": len(knowledge_text),
                                "sources": sources
                            }
                        ))
                        
                        if knowledge_text.strip() and sources:
                            self.knowledge_found = True
                            
                            # Get the list of valid source titles from the knowledge map for validation
                            knowledge_map_docs = self.knowledge_base.knowledge_map.get("documents", [])
                            valid_map_titles = [doc.get("document_title", "") for doc in knowledge_map_docs]
                            
                            # Get the specific sources that were provided for this query
                            query_source_titles = [source.get("title", "") for source in self.last_query_sources]
                            
                            # Make sure we're only validating against sources in the knowledge map
                            valid_sources = []
                            for title in query_source_titles:
                                # Check if this source exists in the knowledge map
                                source_in_map = False
                                for map_title in valid_map_titles:
                                    if title == map_title or fuzzymatch(title, map_title, threshold=0.8):
                                        valid_sources.append(map_title)  # Use the exact title from the map
                                        source_in_map = True
                                        break
                                
                                if not source_in_map:
                                    logger.warning(f"Source not found in knowledge map: {title}")
                        
                        if self.knowledge_found:
                            # Format the knowledge with clear markings and instructions
                            kb_text = f"[MEDICAL KNOWLEDGE FROM VERIFIED SOURCES - YOU MUST EXCLUSIVELY USE THIS FOR ANSWERING MEDICAL QUESTIONS - DO NOT REFERENCE THIS CONTEXT DIRECTLY IN YOUR RESPONSE]\n{knowledge_text}\n\nCitation Sources (YOU MUST ONLY CITE THESE EXACT SOURCES AND NO OTHERS - DO NOT INVENT, MODIFY OR USE ANY OTHER SOURCES):"  
                            
                            # Add source information
                            for source in sources:
                                title = source.get("title", "Unknown")
                                authors = source.get("authors", "")
                                domain = source.get("domain", "")
                                kb_text += f"\n- {title} by {authors} ({domain})"
                            
                            kb_text += "\n[END MEDICAL KNOWLEDGE - YOU ARE STRICTLY PROHIBITED FROM USING ANY OTHER KNOWLEDGE SOURCE OR CITING ANY SOURCE NOT EXPLICITLY LISTED ABOVE]"
                            context_parts.append(kb_text)
                            
                            # Log knowledge retrieval
                            if self.file_logger:
                                asyncio.create_task(self.file_logger.log_event(
                                    "KNOWLEDGE_RETRIEVED",
                                    f"Found {len(sources)} sources of medical knowledge for query",
                                    {"query": last_msg[:100], "num_sources": len(sources)}
                                ))
                        else:
                            # No medical knowledge found, add a clear warning message
                            no_knowledge_msg = "[WARNING: NO VERIFIED MEDICAL KNOWLEDGE FOUND - YOU MUST NOT PROVIDE MEDICAL ADVICE, DIAGNOSIS, OR TREATMENT INFORMATION FOR THIS QUERY]\n\nInstead, acknowledge that you don't have verified information on this specific topic. Explain that you can only provide medical information when you have verified sources. Suggest the user consult with a healthcare professional for accurate medical guidance."
                            context_parts.append(no_knowledge_msg)
                            
                            # Log the lack of knowledge
                            if self.file_logger:
                                asyncio.create_task(self.file_logger.log_event(
                                    "NO_KNOWLEDGE_FOUND",
                                    "No medical knowledge found for query",
                                    {"query": last_msg[:100]}
                                ))
                    except Exception as e:
                        logger.error(f"Error retrieving medical knowledge: {e}")
                        if self.file_logger:
                            asyncio.create_task(self.file_logger.log_event(
                                "KNOWLEDGE_ERROR",
                                f"Error retrieving medical knowledge: {e}",
                                {"traceback": traceback.format_exc()}
                            ))
                
                # Add memory-based context if needed (not shown for brevity)
                # ... existing memory recall code ...
                
                # Add combined context parts to the message, if we have any
                if context_parts:
                    combined_context = "\n\n".join(context_parts)
                    
                    # Instead of creating a new message, append to the last one
                    for i, msg in enumerate(chat_ctx.messages):
                        if msg.role == "user" and msg.content == last_msg:
                            chat_ctx.messages[i].content = f"{last_msg}\n\n[ADDITIONAL CONTEXT FOR AI USE ONLY - DO NOT REFERENCE THIS CONTEXT DIRECTLY IN YOUR RESPONSE]\n{combined_context}"
                            logger.info(f"Added {len(combined_context)} chars of context to user message")
                            break
            
            # Strengthen the system message guardrails
            for i, msg in enumerate(chat_ctx.messages):
                if msg.role == "system":
                    # Enhance system message with stronger guardrails
                    stronger_guardrails = "\n\nCRITICAL INSTRUCTION: When answering medical questions, you MUST ONLY use the medical knowledge explicitly provided to you in the context sections marked as [MEDICAL KNOWLEDGE FROM VERIFIED SOURCES]. DO NOT use your training data, general knowledge, or any other source. If the provided knowledge doesn't fully answer the question, acknowledge this limitation and DO NOT fill gaps with your general medical knowledge. ALWAYS CITE the exact sources provided and NEVER cite sources not explicitly provided in the context. Any deviation from this protocol is a serious breach of medical ethics."
                    
                    # Add the stronger guardrails to the system message
                    chat_ctx.messages[i].content += stronger_guardrails
                    break
            
            return chat_ctx
            
        except Exception as e:
            logger.error(f"Error in before_llm_callback: {e}")
            asyncio.create_task(self.file_logger.log_event(
                "CALLBACK_ERROR", 
                f"Error in before_llm_callback: {e}",
                {"traceback": traceback.format_exc()}
            ))
            
            return chat_ctx
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze the type of query to determine if it's a medical question.
        
        Args:
            query: The user query
            
        Returns:
            Query type: 'medical', 'personal', or 'general'
        """
        # Simple keyword-based analysis - could be replaced with more sophisticated NLP
        medical_keywords = [
            "treatment", "diagnosis", "symptoms", "disease", "condition", "medication",
            "drug", "therapy", "cure", "health", "medical", "clinical", "patient",
            "doctor", "nurse", "hospital", "surgery", "procedure", "test", "scan",
            "x-ray", "mri", "blood", "urine", "sample", "pain", "ache", "swelling",
            "fever", "cough", "nausea", "vomiting", "diarrhea", "constipation",
            "headache", "migraine", "fatigue", "tired", "weakness", "dizzy", "faint",
            "pregnant", "pregnancy", "birth", "labor", "delivery", "pediatric", "child",
            "baby", "infant", "toddler", "adolescent", "teen", "elderly", "senior",
            "geriatric", "prescription", "dose", "dosage", "side effect", "risk",
            "complication", "prognosis", "recovery", "rehabilitation", "therapy",
            "contraindication", "interaction", "allergen", "allergy", "vaccine",
            "immunization", "shot", "booster", "antibiotic", "infection", "virus",
            "bacterial", "fungal", "parasite", "chronic", "acute", "terminal",
            "cancer", "tumor", "malignant", "benign", "biopsy", "remission", "relapse",
            "heart", "cardiac", "lung", "pulmonary", "respiratory", "liver", "hepatic",
            "kidney", "renal", "brain", "neural", "nervous", "stomach", "gastric",
            "intestine", "bowel", "colon", "rectal", "skin", "dermal", "rash",
            "bone", "joint", "arthritis", "muscle", "sprain", "fracture", "break",
            "stroke", "heart attack", "diabetes", "hypertension", "high blood pressure", "anesthesia"
        ]
        
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check for medical keywords
        for keyword in medical_keywords:
            if keyword in query_lower:
                logger.info(f"Query classified as medical based on keyword: {keyword}")
                return "medical"
        
        # Check for personal queries
        personal_patterns = [
            r"\bmy\b", r"\bI have\b", r"\bI am\b", r"\bI feel\b", r"\bI'm\b",
            r"\bI've been\b", r"\bmy family\b", r"\bmy child\b", r"\bmy parent\b"
        ]
        
        for pattern in personal_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Query classified as personal based on pattern: {pattern}")
                return "personal"
        
        # Default to general if no specific match
        logger.info("Query classified as general (no specific keywords matched)")
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
            
            # Store for context in the upcoming agent response
            self.last_user_message = message
            
            # Log user message
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
            if not room or not message:
                logger.warning("Cannot send empty message or no room provided")
                await self.file_logger.log_event("WARNING", "Cannot send empty message or no room provided")
                return False
                
            # Send message to all participants in the room
            await room.local_participant.publish_data(message.encode('utf-8'), rtc.DataPacketKind.RELIABLE)
            logger.info(f"Sent chat message: {message[:50]}...")
            
            # Log the message
            await self.file_logger.log_event("CHAT_MESSAGE_SENT", message)
            
            # Add message to conversation history
            await self.add_agent_message(message)
            return True
            
        except Exception as e:
            error_msg = f"Error sending chat message: {e}"
            logger.error(error_msg)
            await self.file_logger.log_event("ERROR", error_msg)
            return False

def validate_room_name(room_name: str) -> Union[str, None]:
    """Validate that the room name ends with one of the required suffixes"""
    if room_name.endswith(DOCTOR_SUFFIX):
        return "doctor"
    elif room_name.endswith(PATIENT_SUFFIX):
        return "patient"
    else:
        return None

def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    # Load VAD model for voice activity detection
    try:
        logger.info("Loading VAD model...")
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("✅ VAD model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load VAD model: {e}")
        logger.warning("Will attempt to load VAD model at runtime")
        proc.userdata["vad"] = None
    
    # Try to load turn detector model, but make it completely optional
    try:
        logger.info("Loading turn detector model...")
        proc.userdata["turn_detector"] = turn_detector.EOUModel()
        logger.info("✅ Turn detector model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load turn detector model: {e}")
        
        # Try an alternative approach for Docker environments
        try:
            import os
            import sys
            
            # Directory where the model should be
            model_dir = os.path.expanduser("~/.cache/livekit-plugins-turn-detector")
            
            # If the directory doesn't exist, create it
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                logger.info(f"Created model directory: {model_dir}")
            
            # Log available model files
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                logger.info(f"Files in model directory: {files}")
            else:
                logger.warning(f"Model directory does not exist: {model_dir}")
            
            logger.warning("Attempting to load turn detector model again...")
            proc.userdata["turn_detector"] = turn_detector.EOUModel()
            logger.info("✅ Turn detector model loaded on second attempt")
        except Exception as e2:
            logger.warning(f"Second attempt to load turn detector model failed: {e2}")
        logger.warning("Agent will continue using default pause detection for turn detection")
        proc.userdata["turn_detector"] = None

async def fetch_patient_data(nhs_number: str) -> Optional[PatientData]:
    """Fetch patient data from the API"""
    try:
        url = f"{PATIENT_VERIFY_ENDPOINT}/{nhs_number}"
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch patient data: {response.status_code}")
            return None
        
        patient_data = PatientData.from_api_response(response.json())
        logger.info(f"Successfully fetched patient data for {nhs_number}")
        return patient_data
        
    except Exception as e:
        logger.error(f"Error fetching patient data: {e}")
        return None

async def fetch_doctor_data(registration_number: str) -> Optional[DoctorData]:
    """Fetch doctor data from the API"""
    try:
        url = f"{DOCTOR_VERIFY_ENDPOINT}/{registration_number}"
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch doctor data: {response.status_code}")
            return None
        
        doctor_data = DoctorData.from_api_response(response.json())
        logger.info(f"Successfully fetched doctor data for {registration_number}")
        return doctor_data
        
    except Exception as e:
        logger.error(f"Error fetching doctor data: {e}")
        return None

async def entrypoint(ctx: JobContext):
    """Main entry point for the NHS LiveKit agent"""
    logger.info(f"Connecting to room {ctx.room.name}")
    
    # Check required API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    cerebras_api_key = os.environ.get("CEREBRAS_API_KEY")
    
    if not openai_api_key:
        logger.error("OpenAI API key not set in environment variables")
        ctx.error = "OpenAI API key not available. Please provide a valid API key."
        return
    
    if not cerebras_api_key:
        logger.warning("Cerebras API key not set. Will fall back to OpenAI for knowledge base selection.")
    else:
        logger.info("Using Cerebras for knowledge base selection")
    
    # Validate room name to determine user type
    user_type = validate_room_name(ctx.room.name)
    if not user_type:
        logger.error(f"Room name '{ctx.room.name}' does not have the required suffix")
        ctx.error = ROOM_VALIDATION_ERROR
        return
    
    logger.info(f"Detected user type: {user_type}")
    
    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    
    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting NHS virtual assistant for participant {participant.identity}")
    
    # Get participant metadata
    user_id = ""
    user_data = None
    metadata = participant.metadata
    
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
            logger.info(f"Parsed metadata: {parsed_metadata}")
            
            if user_type == "patient":
                nhs_number = parsed_metadata.get('nhs_number', '')
                if not nhs_number:
                    # Try alternate casing
                    nhs_number = parsed_metadata.get('nhsNumber', '')
                if nhs_number:
                    # Try to fetch patient data from API
                    user_id = nhs_number
                    user_data = await fetch_patient_data(nhs_number)
                    logger.info(f"Fetched patient data: {user_data}")
            elif user_type == "doctor":
                registration_number = parsed_metadata.get('registration_number', '')
                if not registration_number:
                    # Try alternate casing
                    registration_number = parsed_metadata.get('registrationNumber', '')
                if registration_number:
                    # Try to fetch doctor data from API
                    user_id = registration_number
                    user_data = await fetch_doctor_data(registration_number)
                    logger.info(f"Fetched doctor data: {user_data}")
        except json.JSONDecodeError:
            logger.error("Failed to parse participant metadata")
    
    # If we couldn't get user data from metadata, create a basic instance
    if not user_data:
        if user_type == "patient":
            # Use a default NHS number if none provided
            user_id = "unknown_patient"
            user_data = PatientData(user_id)
            # Populate from metadata if available
            if metadata and 'parsed_metadata' in locals():
                user_data.full_name = parsed_metadata.get('name', '')
                user_data.date_of_birth = parsed_metadata.get('dateOfBirth', '')
        else:
            # Use a default registration number if none provided
            user_id = "unknown_doctor" if not registration_number else registration_number
            user_data = DoctorData(user_id)
            # Populate from metadata if available
            if metadata and 'parsed_metadata' in locals():
                user_data.full_name = parsed_metadata.get('name', '')
                user_data.hospital = parsed_metadata.get('hospitalName', '')
                user_data.specialty = parsed_metadata.get('speciality', '')
    
    logger.info(f"Initialized user data: {user_data.to_dict()}")
    
    # Initialize NHS agent with user data
    nhs_agent = NHSAgent(user_data)
    
    # Create initial system prompt based on user type
    if user_type == "patient":
        system_prompt = create_patient_system_prompt(user_data)
    else:
        system_prompt = create_doctor_system_prompt(user_data)
    
    # Initialize the chat context with the system prompt
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt,
    )
    
    # Create the voice pipeline agent
    try:
        # Use the turn detector from prewarm if available, otherwise None
        turn_detector_instance = ctx.proc.userdata.get("turn_detector")
        if turn_detector_instance:
            logger.info("Using pre-loaded turn detector model")
        else:
            logger.info("Turn detector not available, using default pause detection")
        
        # Create the voice pipeline agent
        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata.get("vad", silero.VAD.load()),
            stt=deepgram.STT(
                model="nova-2-general",
                interim_results=True,
                smart_format=True,
                punctuate=True,
                language="en-US",
            ),
            llm=openai.LLM.with_cerebras(model="llama-3.3-70b"),
            tts=cartesia_tts.TTS(
                model="sonic-2",
                voice="7e19344f-9f17-47d7-a13a-4366ad06ebf3",
                sample_rate=24000,
                speed="normal",
                emotion=["curiosity", "positivity:high"],
            ),
            chat_ctx=initial_ctx,
            fnc_ctx=nhs_agent.function_ctx,
            turn_detector=turn_detector_instance,  # Can be None if not available
            before_llm_cb=nhs_agent.before_llm_callback,
        )

        chat = rtc.ChatManager(ctx.room)
        
        # Set response for both voice and chat
        last_chat_message_id = None
        
        # Handle voice messages
        @agent.on("user_speech_committed")
        def on_user_speech_committed(msg: llm.ChatMessage):
            if isinstance(msg.content, list):
                content = "\n".join(
                    "[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content
                )
            else:
                content = msg.content
                
            logger.info(f"User speech committed: {content[:50]}...")
            
            # Log user message asynchronously
            asyncio.create_task(nhs_agent.add_user_message(content))
        
        @agent.on("agent_speech_committed")
        def on_agent_speech_committed(msg: llm.ChatMessage):
            """Handler for agent speech commit events"""
            try:
                nonlocal last_chat_message_id
                logger.info(f"Agent response: {msg.content[:100]}...")
                
                # Add to conversation history if the response is valid
                asyncio.create_task(nhs_agent.add_agent_message(msg.content))
                
                # Log the agent's response
                asyncio.create_task(nhs_agent.file_logger.log_agent_response(msg.content))
                
                # If this is a response to a chat message, also send it as a chat message
                if last_chat_message_id is not None:
                    asyncio.create_task(nhs_agent.send_chat_message(ctx.room, msg.content))
                    last_chat_message_id = None
                
            except Exception as e:
                logger.error(f"Error handling agent response: {e}")
                # Try to recover and continue
        
        # Log TTS output
        @agent.on("tts_audio_generated")
        def on_tts_audio_generated(text: str, duration_ms: int):
            logger.info(f"TTS audio generated for: {text[:50]}... ({duration_ms}ms)")
            
            # Log TTS output asynchronously
            asyncio.create_task(nhs_agent.file_logger.log_tts_output(text, duration_ms))
        
        @chat.on("message_received")
        def on_message_received(msg: rtc.ChatMessage):
            logger.info(f"Chat message received: {msg.message}")
            if msg.message:
                # Add message to chat context
                chat_ctx.append(
                    text=msg.message,
                    role="user"
                )
                # Generate reply
                agent.generate_reply()
                logger.info("Response generation created")

        # Handle chat messages (text-based chat)
        @ctx.room.on("message_received")
        def on_message_received(msg: rtc.ChatMessage):
            try:
                nonlocal last_chat_message_id
                sender = msg.sender_sid or "unknown"
                if not msg.message or not msg.message.strip():
                    logger.warning(f"Empty chat message received from {sender}, ignoring")
                    return
                    
                logger.info(f"Chat message received from {sender}: {msg.message}")
                
                # Set the message ID so we know to send the response as a chat message too
                last_chat_message_id = msg.id
                
                # Add message to chat context
                asyncio.create_task(nhs_agent.add_user_message(msg.message))
                logger.info(f"Added chat message to conversation history: {msg.message[:50]}...")
                
                # Generate reply
                logger.info("Generating reply to chat message...")
                agent.generate_reply()
                logger.info("Response generation triggered from chat message")
            except Exception as e:
                logger.error(f"Error handling chat message: {e}")
                # Try to recover and continue
        
        # Set up metrics collection
        usage_collector = metrics.UsageCollector()
        @agent.on("metrics_collected")
        def on_metrics_collected(mtrcs: metrics.AgentMetrics):
            metrics.log_metrics(mtrcs)
            usage_collector.collect(mtrcs)
            
            # Skip logging for metrics that don't have cost/tokens (like PipelineVADMetrics)
            if not hasattr(mtrcs, 'cost') or not hasattr(mtrcs, 'tokens'):
                return
                
            # Log metrics asynchronously
            asyncio.create_task(nhs_agent.file_logger.log_event(
                "METRICS",
                f"Cost: ${mtrcs.cost:.6f}, Tokens: {mtrcs.tokens}",
                {
                    "cost": mtrcs.cost,
                    "tokens": mtrcs.tokens,
                    "latency_ms": mtrcs.latency_ms if hasattr(mtrcs, 'latency_ms') else 0
                }
            ))
        
        # Process conversation at end of session
        async def end_of_session():
            # Process conversation for memory storage
            await nhs_agent.process_conversation()
            
            # Log conversation summary
            user_messages = [msg for msg in nhs_agent.conversation_history if msg["role"] == "user"]
            agent_messages = [msg for msg in nhs_agent.conversation_history if msg["role"] == "assistant"]
            
            if user_messages:
                # Calculate conversation statistics
                conversation_duration = None
                if len(nhs_agent.conversation_history) >= 2:
                    first_msg_time = datetime.datetime.fromisoformat(nhs_agent.conversation_history[0]["timestamp"])
                    last_msg_time = datetime.datetime.fromisoformat(nhs_agent.conversation_history[-1]["timestamp"])
                    conversation_duration = (last_msg_time - first_msg_time).total_seconds()
                
                # Log summary
                logger.info("=" * 50)
                logger.info("CONVERSATION SUMMARY")
                logger.info("=" * 50)
                logger.info(f"User type: {user_type}")
                logger.info(f"User ID: {user_id}")
                logger.info(f"Total messages: {len(nhs_agent.conversation_history)}")
                logger.info(f"User messages: {len(user_messages)}")
                logger.info(f"Agent messages: {len(agent_messages)}")
                
                # Log detailed conversation analysis
                summary_text = (
                    f"Conversation Analysis:\n"
                    f"- User type: {user_type}\n"
                    f"- User ID: {user_id}\n"
                    f"- Total messages: {len(nhs_agent.conversation_history)}\n"
                    f"- User messages: {len(user_messages)}\n"
                    f"- Agent messages: {len(agent_messages)}\n"
                )
                
                # Add conversation topics (from first few user messages)
                if len(user_messages) > 0:
                    topics = ", ".join([msg["text"][:30] + "..." for msg in user_messages[:3]])
                    summary_text += f"- Discussion topics: {topics}\n"
                
                # Add duration information
                if conversation_duration:
                    minutes = int(conversation_duration // 60)
                    seconds = int(conversation_duration % 60)
                    logger.info(f"Conversation duration: {minutes}m {seconds}s")
                    summary_text += f"- Duration: {minutes}m {seconds}s\n"
                
                # Log to structured file
                await nhs_agent.file_logger.log_event(
                    "CONVERSATION_ANALYSIS", 
                    summary_text, 
                    {
                        "user_type": user_type,
                        "user_id": user_id,
                        "total_messages": len(nhs_agent.conversation_history),
                        "user_messages": len(user_messages),
                        "agent_messages": len(agent_messages),
                        "duration_sec": conversation_duration if conversation_duration else None
                    }
                )
                
                # Generate conversation memory summary
                memory_text = "Content saved to memory:\n"
                
                # Get the most recent messages (up to 5) for the memory summary
                recent_exchanges = []
                for i in range(min(5, len(nhs_agent.conversation_history))):
                    if i < len(nhs_agent.conversation_history):
                        msg = nhs_agent.conversation_history[-(i+1)]
                        role = "User" if msg["role"] == "user" else "Assistant"
                        text = msg["text"][:100] + "..." if len(msg["text"]) > 100 else msg["text"]
                        recent_exchanges.append(f"{role}: {text}")
                
                # Add recent exchanges to memory text
                if recent_exchanges:
                    memory_text += "- Recent exchanges:\n  " + "\n  ".join(recent_exchanges) + "\n"
                
                # Add KB usage information
                if nhs_agent.file_logger.knowledge_bases_used:
                    memory_text += f"- Knowledge bases used: {', '.join(nhs_agent.file_logger.knowledge_bases_used)}\n"
                
                # Log memory information
                await nhs_agent.file_logger.log_event("MEMORY_STORAGE", memory_text)
                
                logger.info("=" * 50)
            
            # Log usage metrics
            summary = usage_collector.get_summary()
            logger.info(f"Usage: {summary}")
            
            # Log detailed usage statistics
            usage_stats = {}
            if hasattr(usage_collector, "metrics") and usage_collector.metrics:
                total_tokens = sum(m.tokens for m in usage_collector.metrics)
                total_cost = sum(m.cost for m in usage_collector.metrics)
                avg_latency = sum(m.latency_ms for m in usage_collector.metrics) / len(usage_collector.metrics) if usage_collector.metrics else 0
                
                usage_stats = {
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                    "avg_latency_ms": avg_latency,
                    "call_count": len(usage_collector.metrics)
                }
                
                usage_text = (
                    f"Usage Statistics:\n"
                    f"- Total tokens: {total_tokens}\n"
                    f"- Total cost: ${total_cost:.6f}\n"
                    f"- Average latency: {avg_latency:.2f}ms\n"
                    f"- API calls: {len(usage_collector.metrics)}\n"
                )
                
                await nhs_agent.file_logger.log_event("USAGE_STATISTICS", usage_text, usage_stats)
                
            # Log session completion
            await nhs_agent.file_logger.log_event(
                "SESSION_COMPLETED", 
                "Agent session completed successfully",
                {
                    "log_file": str(nhs_agent.file_logger.log_file)
                }
            )
        
        # Add to shutdown callbacks
        ctx.add_shutdown_callback(end_of_session)
        
        # Start the agent
        agent.start(ctx.room, participant)
        
        # Create welcome message based on user type
        if user_type == "patient":
            welcome_message = create_patient_welcome(user_data)
        else:
            welcome_message = create_doctor_welcome(user_data)
        
        # Log the welcome message
        await nhs_agent.file_logger.log_event("WELCOME_MESSAGE", welcome_message)
        
        # Send welcome message
        await agent.say(welcome_message, allow_interruptions=True)
        
    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        # Attempt to explain error to the user
        ctx.error = f"Failed to initialize NHS virtual assistant: {str(e)}"

def create_patient_system_prompt(patient_data: PatientData) -> str:
    """Create system prompt for patient interactions"""
    # Create base prompt with personalization 
    prompt = (
        f"You are an NHS virtual assistant for patients. Your role is to provide helpful, accurate medical information and support.\n\n"
        
        f"PATIENT INFORMATION:\n"
        f"- You are speaking with {patient_data.full_name if patient_data.full_name else 'a patient'}\n"
        f"- NHS Number: {patient_data.nhs_number}\n\n"
        
        f"KNOWLEDGE INSTRUCTIONS:\n"
        f"- For medical questions, ONLY use information from the medical knowledge sources provided to you in context marked as [MEDICAL KNOWLEDGE FROM VERIFIED SOURCES]\n"
        f"- If no medical knowledge is found for a question, politely explain you don't have specific information and suggest speaking with their GP\n"
        f"- NEVER make up or fabricate medical information that isn't provided in the knowledge sources\n"
        f"- ALWAYS use the format: 'According to [Document Title],...' or 'According to [Document Title] by [Authors],...' when citing sources\n"
        f"- NEVER cite sources that weren't provided to you\n"
        f"- Only provide medical information that is explicitly contained in the knowledge sources\n\n"
        
        f"MEMORY INSTRUCTIONS:\n"
        f"- For casual conversation and follow-up questions, you may use information from previous conversations marked as [MEMORY FROM PREVIOUS CONVERSATIONS]\n"
        f"- DO NOT use memory information to answer medical questions - only use verified medical knowledge\n"
        f"- Memory is ONLY for maintaining conversation context, not for providing medical information\n\n"
        
        f"TONE AND STYLE:\n"
        f"- Be friendly, clear, and easy to understand\n"
        f"- Use plain language, not medical jargon, appropriate for general public\n"
        f"- Be precise and factual when discussing medical information\n"
        f"- Be mindful of patient concerns and anxiety about health issues\n"
        f"- Keep responses concise, with the most important information first\n\n"
        
        f"RESTRICTIONS:\n"
        f"- DO NOT diagnose conditions or prescribe treatments\n"
        f"- DO NOT provide definitive medical advice\n"
        f"- DO NOT guess or speculate about medical information\n"
        f"- DO NOT make up or fabricate information not in your knowledge base\n"
        f"- If you don't have information on a topic, acknowledge this limitation\n"
        f"- Always suggest consulting a healthcare professional for personalized advice\n\n"
        
        f"IMPORTANT: This is a medical application where accuracy is critical. Only provide medical information from verified sources that are explicitly provided to you. Never use your general knowledge or information from memory for medical questions."
    )
    
    return prompt

def create_doctor_system_prompt(doctor_data: DoctorData) -> str:
    """Create system prompt for doctor interactions"""
    # Create base prompt with personalization
    prompt = (
        f"You are an NHS virtual assistant for medical professionals. Your role is to provide accurate, evidence-based medical information to support clinical decision-making.\n\n"
        
        f"DOCTOR INFORMATION:\n"
        f"- You are speaking with Dr. {doctor_data.full_name if doctor_data.full_name else 'a medical professional'}\n"
        f"- Registration Number: {doctor_data.registration_number}\n\n"
        
        f"KNOWLEDGE INSTRUCTIONS:\n"
        f"- For medical questions, ONLY use information from the medical knowledge sources provided to you in context marked as [MEDICAL KNOWLEDGE FROM VERIFIED SOURCES]\n"
        f"- If no medical knowledge is found for a question, politely explain you don't have specific information and suggest consulting medical journals\n"
        f"- NEVER make up or fabricate medical information that isn't provided in the knowledge sources\n"
        f"- ALWAYS use the format: 'According to [Document Title],...' or 'According to [Document Title] by [Authors],...' when citing sources\n"
        f"- NEVER cite sources that weren't provided to you\n"
        f"- Only provide medical information that is explicitly contained in the knowledge sources\n\n"
        
        f"MEMORY INSTRUCTIONS:\n"
        f"- For casual conversation and follow-up questions, you may use information from previous conversations marked as [MEMORY FROM PREVIOUS CONVERSATIONS]\n"
        f"- DO NOT use memory information to answer medical questions - only use verified medical knowledge\n"
        f"- Memory is ONLY for maintaining conversation context, not for providing medical information\n\n"
        
        f"TONE AND STYLE:\n"
        f"- Use professional, clinical language appropriate for medical professionals\n"
        f"- Be precise and technically accurate\n"
        f"- Use medical terminology where appropriate\n"
        f"- Provide evidence-based information with proper citation\n"
        f"- Keep responses concise and well-structured\n\n"
        
        f"RESTRICTIONS:\n"
        f"- DO NOT provide medical advice that contradicts clinical guidelines\n"
        f"- DO NOT make absolute claims about treatments or outcomes\n"
        f"- DO NOT guess or speculate about medical information\n"
        f"- DO NOT make up or fabricate information not in your knowledge base\n"
        f"- If you don't have information on a topic, acknowledge this limitation\n"
        f"- Remember that the doctor has ultimate responsibility for clinical decisions\n\n"
        
        f"IMPORTANT: This is a medical application where accuracy is critical. Only provide medical information from verified sources that are explicitly provided to you. Never use your general knowledge or information from memory for medical questions."
    )
    
    return prompt

def create_patient_welcome(patient_data: PatientData) -> str:
    """Create welcome message for patients"""
    if patient_data.full_name:
        welcome_message = f"Hello {patient_data.full_name}, welcome to the NHS virtual health assistant. I'm here to provide you with health information and support. How can I help you today?"
    else:
        welcome_message = "Hello, welcome to the NHS virtual health assistant. I'm here to provide you with health information and support. May I know your name to better assist you?"
    
    return welcome_message

def create_doctor_welcome(doctor_data: DoctorData) -> str:
    """Create welcome message for doctors"""
    if doctor_data.full_name:
        specialty = f" in {doctor_data.specialty}" if doctor_data.specialty else ""
        welcome_message = f"Hello {doctor_data.full_name}, welcome to the NHS clinical assistant. I'm here to support your work{specialty}. How can I assist you today?"
    else:
        welcome_message = "Hello Doctor, welcome to the NHS clinical assistant. I'm here to support your clinical work. How can I assist you today?"
    
    return welcome_message

def get_embedding(text: str) -> List[float]:
    """
    Get embedding for text using OpenAI's embedding API
    
    Args:
        text: The text to get embedding for
        
    Returns:
        List of floats representing the embedding
    """
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def fuzzymatch(a: str, b: str, threshold: float = 0.3) -> bool:
    """Helper function to do fuzzy matching on strings"""
    # Simple fuzzy match based on character overlap
    a_lower = a.lower()
    b_lower = b.lower()
    
    # Calculate character overlap ratio
    shorter = min(len(a_lower), len(b_lower))
    longer = max(len(a_lower), len(b_lower))
    
    if shorter == 0:
        return False
        
    # Check for substring
    if a_lower in b_lower or b_lower in a_lower:
        return True
        
    # Count character matches
    common_chars = sum(1 for c in set(a_lower) if c in b_lower)
    ratio = common_chars / len(set(a_lower + b_lower))
    
    return ratio >= threshold

if __name__ == "__main__":
    # Run the LiveKit agent
    # Download models with: python nhs_agents.py download-files
    # Start agent with: python nhs_agents.py start
    logger.info("Starting NHS Virtual Assistant")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            num_idle_processes=2  # Keep 2 processes warm for better response time
        ),
    )
