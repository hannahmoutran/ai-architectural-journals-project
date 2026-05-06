"""
Configuration settings for Southern Architect metadata workflow processing.
"""

import os

# Folder configuration.
# Change values here to update input/output directory names across all scripts at once.
FOLDER_CONFIG = {
    "input_dir": "image_folders",        # Parent directory containing input image folders
    "output_dir": "output_folders",      # Directory where output run folders are created
    "default_input_folder": "6_pages",  # Default input folder within input_dir
}

# Batch processing configuration.
BATCH_CONFIG = {
    "auto_threshold": 10,  # Use batch automatically when request count exceeds this number
}

# Provider configuration.
# Set "provider" to "openai" or "gemini" to switch between providers.
# Note: batch processing always uses the OpenAI Batch API directly regardless of this setting.
# Gemini models (individual mode only): gemini-3.1-flash-lite-preview, gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash
# OpenAI models: gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini
PROVIDER_CONFIG = {
    "provider": "openai",  # "openai" or "gemini"
}

# Default models for each step.
# Change values here to update the model used across all scripts at once.
DEFAULT_MODELS = {
    "step1": "gpt-4.1",  # Used by step1_text and step1_image (individual mode)
    "step1_batch": "gpt-4.1",                  # Used by step1_text and step1_image (batch mode)
    "step1_5": "gpt-4.1",       # Used by step1.5 (batch cleanup)
    "step3": "gpt-4.1-mini",         # Used by step3 (vocabulary selection, individual mode)
    "step3_batch": "gpt-4.1-mini",   # Used by step3 (vocabulary selection, batch mode)
    "step4": "gpt-4.1-mini",         # Used by step4 (issue synthesis)
    "batch_default": "gpt-4.1-mini",  # Fallback model in batch_processor
}

# Portkey gateway configuration
# Set enabled to True to route individual (non-batch) OpenAI calls through Portkey.
# Requires PORTKEY_API_KEY and PORTKEY_VIRTUAL_KEY environment variables.
# Batch processing always uses OpenAI directly (Portkey does not support the Batch API).
# Portkey is ignored when provider is set to "gemini".
PORTKEY_CONFIG = {
    "enabled": True,
    "api_key_env": "PORTKEY_API_KEY",
    "virtual_key_env": "PORTKEY_VIRTUAL_KEY"
}


class _GeminiUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _GeminiMessage:
    def __init__(self, content):
        self.content = content


class _GeminiChoice:
    def __init__(self, content):
        self.message = _GeminiMessage(content)


class _GeminiResponse:
    def __init__(self, native_response):
        self.choices = [_GeminiChoice(native_response.text or "")]
        usage = native_response.usage_metadata
        self.usage = _GeminiUsage(
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )


class _GeminiCompletions:
    def __init__(self, genai_client):
        self._client = genai_client

    def create(self, model, messages, max_tokens=None, temperature=None, **kwargs):
        import base64
        from google.genai import types

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg.get("content", "")
            if isinstance(content, str):
                parts = [types.Part(text=content)]
            else:
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append(types.Part(text=item["text"]))
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64data = url.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            image_bytes = base64.b64decode(b64data)
                            parts.append(
                                types.Part(
                                    inline_data=types.Blob(
                                        mime_type=mime_type, data=image_bytes
                                    )
                                )
                            )
            contents.append(types.Content(role=role, parts=parts))

        has_vision = any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for msg in messages
            for item in (
                msg.get("content", [])
                if isinstance(msg.get("content"), list)
                else []
            )
        )

        # Only disable thinking for models that support the thinking API (Pro/Flash, not Lite/preview variants)
        thinking_models = {"gemini-2.5-pro", "gemini-2.5-flash"}
        config_kwargs = {}
        if model in thinking_models:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if not has_vision:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)
        response = self._client.models.generate_content(
            model=model, contents=contents, config=config
        )
        return _GeminiResponse(response)


class _GeminiChat:
    def __init__(self, genai_client):
        self.completions = _GeminiCompletions(genai_client)


class _GeminiNativeClient:
    """Native google-genai client wrapped to present an OpenAI-compatible interface."""

    def __init__(self, api_key):
        from google import genai
        self._genai_client = genai.Client(api_key=api_key)
        self.chat = _GeminiChat(self._genai_client)


def get_openai_client():
    """
    Return an API client compatible with the OpenAI chat completions interface.

    When PROVIDER_CONFIG['provider'] is 'gemini', returns a native google-genai
    client wrapped to match the OpenAI interface (requires GEMINI_API_KEY).
    Thinking tokens are disabled and JSON mode is set for text-only requests.
    Portkey routing is skipped for Gemini.

    When provider is 'openai' and PORTKEY_CONFIG['enabled'] is True, returns a
    Portkey client if the required environment variables are set. Otherwise
    returns a standard OpenAI client.

    Note: batch processing in batch_processor.py always uses OpenAI directly.
    """
    if PROVIDER_CONFIG.get("provider") == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        return _GeminiNativeClient(api_key=gemini_api_key)

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
