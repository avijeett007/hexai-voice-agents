"""User data models for NHS Virtual Assistant

This module contains the data models for patients and doctors using the NHS virtual assistant.
"""

from typing import Dict, Any, List

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
        return f"{self.user_type.capitalize()}: {self.full_name} ({self.user_id})"


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
        patient.full_name = response_data.get("name", "")
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
        doctor.full_name = response_data.get("name", "")
        doctor.email = response_data.get("email", "")
        doctor.phone = response_data.get("phone", "")
        doctor.hospital = response_data.get("hospital", "")
        doctor.specialty = response_data.get("specialty", "")
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
