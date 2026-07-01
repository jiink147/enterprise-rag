"""
LLM Client - Layer 4: Generation

Sends prompts to Ollama's local API and streams back the response.
"""

import json
import logging
import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"


def generate(prompt, model=DEFAULT_MODEL, timeout=120):
    """
    Send a prompt to Ollama and return the complete response text.

    Args:
        prompt: Full prompt string (already assembled by PromptBuilder).
        model: Ollama model name to use.
        timeout: Seconds to wait before giving up (CPU inference is slow).

    Returns:
        dict:
            {
                "text": str,       # the model's answer
                "model": str,      # model used
                "success": bool,
                "error": str|None
            }
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return {
            "text": data.get("response", "").strip(),
            "model": model,
            "success": True,
            "error": None,
        }
    except requests.exceptions.ConnectionError:
        msg = "Cannot connect to Ollama. Is 'ollama serve' running?"
        logger.error(msg)
        return {"text": "", "model": model, "success": False, "error": msg}
    except requests.exceptions.Timeout:
        msg = f"Ollama request timed out after {timeout}s."
        logger.error(msg)
        return {"text": "", "model": model, "success": False, "error": msg}
    except Exception as e:
        msg = f"Unexpected error calling Ollama: {e}"
        logger.error(msg)
        return {"text": "", "model": model, "success": False, "error": msg}


if __name__ == "__main__":
    print("Testing Ollama connection...")
    result = generate("Siapa kamu? Jawab dalam satu kalimat.")
    if result["success"]:
        print(f"Model: {result['model']}")
        print(f"Answer: {result['text']}")
    else:
        print(f"Error: {result['error']}")
