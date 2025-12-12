"""NHS Virtual Assistant Package

This package implements a virtual assistant for NHS doctors and patients.
"""

# Import key classes and functions for easier access
from .models import UserData, PatientData, DoctorData
from .agent import NHSAgent
from .memory import VectorMemory
from .knowledge import KnowledgeBase
from .logger import StructuredFileLogger
from .main import entrypoint