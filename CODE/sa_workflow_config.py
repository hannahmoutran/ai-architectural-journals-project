"""
Configuration settings for Southern Architect metadata workflow processing.
"""

import os

# Folder configuration.
# Change values here to update input/output directory names across all scripts at once.
FOLDER_CONFIG = {
    "input_dir": "image_folders",        # Parent directory containing input image folders
    "output_dir": "output_folders",      # Directory where output run folders are created
    "default_input_folder": "4_pages",  # Default input folder within input_dir
}

# Batch processing configuration.
BATCH_CONFIG = {
    "auto_threshold": 1,  # Use batch automatically when request count exceeds this number
}

# Default models for each step.
# Change values here to update the model used across all scripts at once.
DEFAULT_MODELS = {
    "step1": "gpt-4o",               # Used by step1_text and step1_image (individual mode)
    "step1_batch": "gpt-4.1",         # Used by step1_text and step1_image (batch mode)
    "step1_5": "gpt-4.1-mini",       # Used by step1.5 (batch cleanup)
    "step3": "gpt-4.1-mini",         # Used by step3 (vocabulary selection, individual mode)
    "step3_batch": "gpt-4.1-mini",   # Used by step3 (vocabulary selection, batch mode)
    "step4": "gpt-4.1-mini",         # Used by step4 (issue synthesis)
    "batch_default": "gpt-4o-mini-2024-07-18",  # Fallback model in batch_processor
}

# Portkey gateway configuration
# Set enabled to True to route individual (non-batch) OpenAI calls through Portkey.
# Requires PORTKEY_API_KEY and PORTKEY_VIRTUAL_KEY environment variables.
# Batch processing always uses OpenAI directly (Portkey does not support the Batch API).
PORTKEY_CONFIG = {
    "enabled": False,
    "api_key_env": "PORTKEY_API_KEY",
    "virtual_key_env": "PORTKEY_VIRTUAL_KEY"
}


def get_openai_client():
    """
    Return an API client for OpenAI calls.

    If PORTKEY_CONFIG['enabled'] is True and the required Portkey environment
    variables are set, returns a Portkey client (routes calls through the
    Portkey AI gateway). Otherwise returns a standard OpenAI client.

    Note: batch processing in batch_processor.py always uses OpenAI directly.
    """
    if PORTKEY_CONFIG.get("enabled"):
        portkey_api_key = os.getenv(PORTKEY_CONFIG["api_key_env"])
        portkey_virtual_key = os.getenv(PORTKEY_CONFIG["virtual_key_env"])
        if portkey_api_key and portkey_virtual_key:
            try:
                from portkey_ai import Portkey
                return Portkey(api_key=portkey_api_key, virtual_key=portkey_virtual_key)
            except ImportError:
                print("Warning: portkey_ai package not installed. Falling back to OpenAI.")
        else:
            print("Warning: Portkey enabled but PORTKEY_API_KEY/PORTKEY_VIRTUAL_KEY not set. Falling back to OpenAI.")

    from openai import OpenAI
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
