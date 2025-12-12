"""Main entry point for NHS LiveKit agent

This module contains the main functions for initializing and running the NHS
LiveKit agent, handling room connections, and managing user validation.
"""

import os
import logging
import asyncio
import json
import traceback
import datetime
import requests
from typing import Dict, Any, Optional

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

# Local imports
from .models import PatientData, DoctorData
from .agent import NHSAgent
from .prompts import create_patient_system_prompt, create_doctor_system_prompt, create_patient_welcome, create_doctor_welcome
from .constants import VALID_ROOM_SUFFIXES, DOCTORS_API_ENDPOINT, PATIENTS_API_ENDPOINT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nhs_agent.main")

def validate_room_name(room_name: str) -> bool:
    """Validate that the room name ends with one of the required suffixes"""
    for suffix in VALID_ROOM_SUFFIXES:
        if room_name.endswith(f"_{suffix}"):
            return True
    return False

async def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    try:
        logger.info("Prewarming models...")
        
        # Prewarm voice activity detection
        turn_detector.init_voice_activity_detector()
        
        # Prewarm silero TTS model
        silero.download_models()
        
        # Load OpenAI models
        await proc.worker_state.load_plugin(
            "openai",
            llm_model="gpt-4",
            tts_model="tts-1",
        )
        
        # Preload Deepgram
        await proc.worker_state.load_plugin(
            "deepgram",
            model="nova-2",
            smart_format=True,
            language="en",
            enable_summarization=True,
        )
        
        # Preload Cartesia TTS
        await proc.worker_state.load_plugin("cartesia-tts")
        
        logger.info("Models prewarmed successfully")
        
    except Exception as e:
        logger.error(f"Error prewarming models: {e}")

def fetch_patient_data(nhs_number: str) -> Optional[PatientData]:
    """Fetch patient data from the API"""
    try:
        logger.info(f"Fetching patient data for NHS number {nhs_number}")
        response = requests.get(f"{PATIENTS_API_ENDPOINT}/{nhs_number}")
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch patient data: {response.status_code}")
            return None
        
        patient_data = PatientData.from_api_response(response.json())
        logger.info(f"Successfully fetched data for patient {patient_data.full_name}")
        return patient_data
        
    except Exception as e:
        logger.error(f"Error fetching patient data: {e}")
        return None

def fetch_doctor_data(registration_number: str) -> Optional[DoctorData]:
    """Fetch doctor data from the API"""
    try:
        logger.info(f"Fetching doctor data for registration number {registration_number}")
        response = requests.get(f"{DOCTORS_API_ENDPOINT}/{registration_number}")
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch doctor data: {response.status_code}")
            return None
        
        doctor_data = DoctorData.from_api_response(response.json())
        logger.info(f"Successfully fetched data for doctor {doctor_data.full_name}")
        return doctor_data
        
    except Exception as e:
        logger.error(f"Error fetching doctor data: {e}")
        return None

async def entrypoint(ctx: JobContext):
    """Main entry point for the NHS LiveKit agent"""
    try:
        # Extract room name and user ID from context
        room_name = ctx.room_name
        logger.info(f"Starting NHS agent for room: {room_name}")
        
        # Validate room name format
        if not validate_room_name(room_name):
            error_msg = f"Invalid room name format: {room_name}. Room name must end with _doctor or _patient."
            logger.error(error_msg)
            return error_msg
        
        # Parse room name for user type and ID
        room_parts = room_name.split("_")
        user_type = room_parts[-1]  # Last part is the user type (doctor/patient)
        user_id = "_".join(room_parts[:-1])  # The rest is the user ID
        
        logger.info(f"Room parsed: Type={user_type}, ID={user_id}")
        
        # Fetch user data based on type
        user_data = None
        welcome_message = ""
        system_prompt = ""
        
        if user_type == "patient":
            user_data = fetch_patient_data(user_id)
            if user_data:
                system_prompt = create_patient_system_prompt(user_data)
                welcome_message = create_patient_welcome(user_data)
            else:
                error_msg = f"Failed to fetch patient data for NHS number: {user_id}"
                logger.error(error_msg)
                return error_msg
                
        elif user_type == "doctor":
            user_data = fetch_doctor_data(user_id)
            if user_data:
                system_prompt = create_doctor_system_prompt(user_data)
                welcome_message = create_doctor_welcome(user_data)
            else:
                error_msg = f"Failed to fetch doctor data for registration number: {user_id}"
                logger.error(error_msg)
                return error_msg
        else:
            error_msg = f"Invalid user type: {user_type}. Must be 'doctor' or 'patient'."
            logger.error(error_msg)
            return error_msg
        
        # Create the NHS agent
        nhs_agent = NHSAgent(user_data)
        
        # Join the room
        options = WorkerOptions(
            type=WorkerType.VOICE,
            auto_subscribe=AutoSubscribe.SUBSCRIBED_TRACKS,
            start_paused=True,
        )
        
        room = await ctx.get_room(worker_options=options)
        logger.info(f"Joined room: {room_name}")
        
        # Subscribe to data messages
        async def on_data_received(data: bytes, participant: rtc.RemoteParticipant, topic: str):
            try:
                if topic == "chat":
                    # Parse JSON data
                    message_data = json.loads(data.decode("utf-8"))
                    message = message_data.get("message", "")
                    sender = message_data.get("sender", "Unknown")
                    
                    logger.info(f"Received chat message from {sender}: {message[:50]}...")
                    
                    # Process the message through the agent
                    asyncio.create_task(nhs_agent.add_user_message(message))
                    
                    # Let the response happen through the voice pipeline
                    
            except Exception as e:
                logger.error(f"Error processing data message: {e}")
        
        room.on("data_received", on_data_received)
        
        # Initialize voice pipeline agent
        agent = await ctx.create_agent(
            agent_type=VoicePipelineAgent,
            room=room,
            system_prompt=system_prompt,
            welcome_message=welcome_message,
            plugins={
                "deepgram": {
                    "model": "nova-2",
                    "smart_format": True,
                    "language": "en",
                    "enable_summarization": True,
                },
                "openai": {
                    "llm_model": "gpt-4-turbo",
                    "temperature": 0.1,
                    "tts_model": "tts-1",
                    "tts_voice": "nova",
                },
            },
            # Add turn detection
            turn_detector={
                "voice_activity_detection": {
                    "frame_duration_ms": 30,
                    "padding_duration_ms": 200,
                    "threshold": 0.5,
                },
                "silence_patience_ms": 900,
            },
        )
        
        # Set up callbacks
        agent.add_before_llm_callback(nhs_agent.before_llm_callback)
        
        # Log agent response and store in conversation history
        async def after_llm_callback(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext, response: llm.LLMResponse):
            try:
                text = response.message.content
                logger.info(f"Agent response: {text[:100]}...")
                
                # Add the message to the conversation history
                asyncio.create_task(nhs_agent.add_agent_message(text))
                
                # Also send as chat message
                asyncio.create_task(nhs_agent.send_chat_message(room, text))
                
            except Exception as e:
                logger.error(f"Error in after_llm_callback: {e}")
        
        agent.add_after_llm_callback(after_llm_callback)
        
        # Hook up TTS events for logging
        async def on_tts_event(event: Dict[str, Any]):
            try:
                if event.get("type") == "tts_output":
                    text = event.get("text", "")
                    duration_ms = event.get("duration_ms")
                    
                    # Log the TTS output
                    if text:
                        asyncio.create_task(nhs_agent.file_logger.log_tts_output(text, duration_ms))
                    
            except Exception as e:
                logger.error(f"Error processing TTS event: {e}")
        
        agent.on("tts_output", on_tts_event)
        
        # Hook up room leave events
        async def on_disconnected():
            try:
                logger.info(f"Disconnected from room: {room_name}")
                
                # Process the conversation history
                await nhs_agent.process_conversation()
                
                logger.info("NHS agent session complete")
                
            except Exception as e:
                logger.error(f"Error on disconnection: {e}")
        
        room.on("disconnected", on_disconnected)
        
        # Start the agent by unpausing
        await room.set_paused(False)
        
        # Wait for the agent to complete - will exit when the connection is closed
        await agent.wait_until_done()
        
        return "NHS agent session completed successfully"
        
    except Exception as e:
        error_msg = f"Error in NHS agent: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg
