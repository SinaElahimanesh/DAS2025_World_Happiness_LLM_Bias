"""
Configuration Module for LLM Audit System

This module provides centralized configuration, including API key management.
The API key is loaded from the .env file or OPENAI_API_KEY environment variable.

Usage:
    from config import get_api_key
    api_key = get_api_key()
"""

import os
from pathlib import Path

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def _load_env_file():
    """Load .env file from project root if python-dotenv is available"""
    if DOTENV_AVAILABLE:
        # Get project root (parent of llm_audit_data directory)
        # config.py is in llm_audit_data/, so go up one level to project root
        current_file = Path(__file__).resolve()  # Use absolute path
        project_root = current_file.parent.parent
        env_file = project_root / '.env'
        
        # Debug: print paths if needed (commented out)
        # print(f"Config file: {current_file}")
        # print(f"Project root: {project_root}")
        # print(f"Env file: {env_file}")
        # print(f"Env file exists: {env_file.exists()}")
        
        if env_file.exists():
            load_dotenv(env_file, override=True)  # override=True to ensure it loads
            return True
    return False


def get_api_key():
    """
    Get OpenAI API key from .env file or environment variable.
    
    First tries to load from .env file in project root.
    Falls back to OPENAI_API_KEY environment variable.
    
    Returns:
        str: The API key from .env file or environment variable
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    # Try to load from .env file first
    _load_env_file()
    
    # Get from environment (either from .env or system env)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        error_msg = (
            "OPENAI_API_KEY is not set. Please set it using one of these methods:\n"
            "  1. Create a .env file in the project root with: OPENAI_API_KEY=your-api-key-here\n"
            "  2. Export environment variable: export OPENAI_API_KEY='your-api-key-here'\n\n"
            "Note: If using .env file, install python-dotenv: pip install python-dotenv"
        )
        raise ValueError(error_msg)
    
    return api_key
