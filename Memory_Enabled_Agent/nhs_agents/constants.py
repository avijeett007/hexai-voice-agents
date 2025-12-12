"""Constants and configuration for NHS agent

This module contains knowledge base maps, API endpoints, and other constants
used throughout the NHS agent system.
"""

import os

# Environment settings
DOCTORS_API_ENDPOINT = os.getenv("DOCTORS_API_ENDPOINT", "http://localhost:8000/api/doctors")
PATIENTS_API_ENDPOINT = os.getenv("PATIENTS_API_ENDPOINT", "http://localhost:8000/api/patients")
MEDICAL_RECORDS_ENDPOINT = os.getenv("MEDICAL_RECORDS_ENDPOINT", "http://localhost:8000/api/medical-records")

# Collection names for different types of data
COMMON_KNOWLEDGE_COLLECTION = "nhs_knowledge"
DOCTOR_KNOWLEDGE_COLLECTION = "nhs_doctor_knowledge"  # Advanced medical knowledge for doctors

# Knowledge Base Maps
# These map collection names to document lists with metadata for citations

# Patient-friendly knowledge base with simplified medical information
PATIENT_KNOWLEDGE_BASE_MAP = {
    "collection_name": COMMON_KNOWLEDGE_COLLECTION,
    "documents": [
        {
            "domain": "Anaesthesia",
            "title": "Understanding Anaesthesia",
            "source": "Royal College of Anaesthetists Guide for Patients",
            "authors": "Royal College of Anaesthetists",
            "year": "2021",
            "url": "https://www.rcoa.ac.uk/patient-information",
            "description": "Patient-friendly information about anaesthesia procedures and risks"
        },
        {
            "domain": "Cardiology",
            "title": "Heart Health Guide",
            "source": "British Heart Foundation",
            "authors": "British Heart Foundation",
            "year": "2022",
            "url": "https://www.bhf.org.uk/informationsupport",
            "description": "Comprehensive guide on heart health for patients"
        },
        {
            "domain": "Diabetes",
            "title": "Living with Diabetes",
            "source": "Diabetes UK",
            "authors": "Diabetes UK",
            "year": "2022",
            "url": "https://www.diabetes.org.uk/guide-to-diabetes",
            "description": "Patient information on managing diabetes in daily life"
        },
        {
            "domain": "General Health",
            "title": "NHS Health A-Z",
            "source": "NHS UK",
            "authors": "NHS Digital",
            "year": "2023",
            "url": "https://www.nhs.uk/conditions",
            "description": "Comprehensive health information from the NHS"
        },
        {
            "domain": "Mental Health",
            "title": "Mental Health Support",
            "source": "Mind",
            "authors": "Mind",
            "year": "2022",
            "url": "https://www.mind.org.uk",
            "description": "Information and support for mental health issues"
        }
    ]
}

# Doctor-specific knowledge base with advanced medical information
DOCTOR_KNOWLEDGE_BASE_MAP = {
    "collection_name": DOCTOR_KNOWLEDGE_COLLECTION,
    "documents": [
        {
            "domain": "Cardiology",
            "title": "NICE Guidelines for Cardiovascular Disease",
            "source": "National Institute for Health and Care Excellence",
            "authors": "NICE",
            "year": "2023",
            "url": "https://www.nice.org.uk/guidance/conditions-and-diseases/cardiovascular-conditions",
            "description": "Clinical guidelines for cardiovascular disease management"
        },
        {
            "domain": "Endocrinology",
            "title": "NICE Guidelines for Diabetes Management",
            "source": "National Institute for Health and Care Excellence",
            "authors": "NICE",
            "year": "2022",
            "url": "https://www.nice.org.uk/guidance/ng28",
            "description": "Clinical guidelines for diabetes management"
        },
        {
            "domain": "Infectious Disease",
            "title": "NICE Guidelines for Antimicrobial Prescribing",
            "source": "National Institute for Health and Care Excellence",
            "authors": "NICE",
            "year": "2022",
            "url": "https://www.nice.org.uk/guidance/conditions-and-diseases/infections",
            "description": "Guidelines for antimicrobial prescribing and stewardship"
        },
        {
            "domain": "Neurology",
            "title": "NICE Guidelines for Neurological Conditions",
            "source": "National Institute for Health and Care Excellence",
            "authors": "NICE",
            "year": "2023",
            "url": "https://www.nice.org.uk/guidance/conditions-and-diseases/neurological-conditions",
            "description": "Clinical guidelines for neurological conditions"
        },
        {
            "domain": "Oncology",
            "title": "NICE Guidelines for Cancer Management",
            "source": "National Institute for Health and Care Excellence",
            "authors": "NICE",
            "year": "2023",
            "url": "https://www.nice.org.uk/guidance/conditions-and-diseases/cancer",
            "description": "Clinical guidelines for cancer management"
        },
        {
            "domain": "Pharmacology",
            "title": "British National Formulary",
            "source": "NICE and British Medical Association",
            "authors": "Joint Formulary Committee",
            "year": "2023",
            "url": "https://bnf.nice.org.uk",
            "description": "Authoritative pharmaceutical reference for UK healthcare professionals"
        },
        {
            "domain": "Psychiatry",
            "title": "NICE Guidelines for Mental Health",
            "source": "National Institute for Health and Care Excellence",
            "authors": "NICE", 
            "year": "2022",
            "url": "https://www.nice.org.uk/guidance/conditions-and-diseases/mental-health-and-behavioural-conditions",
            "description": "Clinical guidelines for mental health conditions"
        },
        {
            "domain": "Public Health",
            "title": "UK Health Security Agency Guidance",
            "source": "UK Health Security Agency",
            "authors": "UKHSA",
            "year": "2023",
            "url": "https://www.gov.uk/government/organisations/uk-health-security-agency",
            "description": "Public health guidelines and infectious disease surveillance"
        }
    ]
}

# Room name validation settings
VALID_ROOM_SUFFIXES = ["doctor", "patient"]
