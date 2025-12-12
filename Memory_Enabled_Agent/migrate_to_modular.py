#!/usr/bin/env python3
"""Migration script to replace the monolithic nhs_agents.py with the modular version"""

import os
import shutil

def migrate():
    print("Starting migration to modular NHS agent structure...")
    
    # Backup the original file
    src_file = "nhs_agents.py"
    backup_file = "nhs_agents.py.backup"
    
    if os.path.exists(src_file):
        print(f"Backing up {src_file} to {backup_file}")
        shutil.copy2(src_file, backup_file)
    
    # Create the new simplified file
    print("Creating new streamlined entry point file...")
    with open(src_file, "w") as f:
        f.write("""\
"""NHS Virtual Assistant Voice Agent

This module now serves as the main entry point to the modularized NHS agent.
The implementation has been split into separate modules for better maintainability.

Environment variables required:
- OPENAI_API_KEY: For embeddings and fallback LLM
- CEREBRAS_API_KEY: For cost-efficient knowledge base selection (optional, will fall back to OpenAI)
- QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY: For vector storage

Author: Avijit Sarkar (Modified version)
"""

# Re-export main components for backward compatibility
from nhs_agents.models import UserData, PatientData, DoctorData
from nhs_agents.agent import NHSAgent, NHSFunctions
from nhs_agents.memory import VectorMemory
from nhs_agents.knowledge import KnowledgeBase
from nhs_agents.logger import StructuredFileLogger
from nhs_agents.main import entrypoint, prewarm, validate_room_name, fetch_patient_data, fetch_doctor_data
from nhs_agents.constants import (
    COMMON_KNOWLEDGE_COLLECTION,
    DOCTOR_KNOWLEDGE_COLLECTION,
    PATIENT_KNOWLEDGE_BASE_MAP,
    DOCTOR_KNOWLEDGE_BASE_MAP,
    VALID_ROOM_SUFFIXES
)
from nhs_agents.utils import get_embedding, fuzzymatch
from nhs_agents.prompts import create_patient_system_prompt, create_doctor_system_prompt, create_patient_welcome, create_doctor_welcome

# LiveKit imports for CLI
from livekit.agents import cli

if __name__ == "__main__":
    # Run the LiveKit agent
    # Download models with: python nhs_agents.py download-files
    # Start agent with: python nhs_agents.py start
    cli.LKAgentCLI(
        agent_entrypoint=entrypoint,
        agent_prewarm=prewarm,
        description="NHS Virtual Assistant for Patients and Doctors",
    ).run()
""")
    
    print("Migration completed successfully!")
    print("\nOriginal file has been backed up to nhs_agents.py.backup")
    print("\nTo verify the new structure works:")
    print("1. Run: python nhs_agents.py download-files")
    print("2. Run: python nhs_agents.py start")
    print("\nIf you encounter any issues, you can restore the backup with:")
    print("cp nhs_agents.py.backup nhs_agents.py")

if __name__ == "__main__":
    migrate()
