import logging
from typing import Any, Dict

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def generate_response(prompt: str, model: str = DEFAULT_MODEL, **extra_params: Any) -> str:
    """
    Send a prompt to a local Ollama instance and return the generated text.

    Ollama must be running locally, e.g.:
      - `ollama run llama3`  (first time pulls the model)
      - `ollama serve`       (if not already running)
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    payload.update(extra_params or {})

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    except Exception as e:
        logging.error(f"Ollama request failed: {e}")
        raise RuntimeError(f"Ollama request failed: {e}") from e

    if resp.status_code != 200:
        logging.error(f"Ollama returned {resp.status_code}: {resp.text}")
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

    data = resp.json()
    # Ollama's /api/generate returns a JSON object with a 'response' field
    text = data.get("response", "")
    return text.strip()


def build_triage_prompt_question(
    user_message: str,
    language_code: str,
    symptoms_summary: str | None = None,
    questions_asked_count: int | None = None,
) -> str:
    """
    Construct a structured prompt for the medical triage assistant.

    language_code examples: "en", "hi", "ta", "te", "bn", "kn", or "en-IN"/"hi-IN"/etc.
    """
    lang_short = language_code.split("-")[0].lower() if language_code else "en"

    qn = questions_asked_count or 0

    return (
        "You are an AI medical triage assistant for rural primary care.\n\n"
        f"User message (in their own language):\n{user_message}\n\n"
        f"Summary of symptoms so far (may be incomplete):\n{symptoms_summary or user_message}\n\n"
        f"User language code: {language_code} (short: {lang_short}).\n"
        f"Conversation state: You have already asked {qn} triage follow-up questions so far.\n\n"
        "Behavior rules:\n"
        "1. ALWAYS respond in the SAME language as the user message above.\n"
        "   - If the user message is in Kannada, respond in Kannada.\n"
        "   - If in Hindi, respond in Hindi.\n"
        "   - If in Tamil, respond in Tamil.\n"
        "   - Never switch to English unless the user message itself was in English.\n"
        "2. Act like a doctor doing medical TRIAGE, not giving full diagnosis.\n"
        "3. Ask EXACTLY ONE short follow-up question in each response.\n"
        "   - Do NOT ask multiple questions in a single message.\n"
        "4. Keep your message very short and conversational (1–2 short sentences).\n"
        "5. You are STILL in the QUESTION PHASE of triage.\n"
        "   - Do NOT suggest diseases.\n"
        "   - Do NOT mention urgency level yet.\n"
        "   - Do NOT give final advice yet.\n\n"
        "Now produce your next message:\n"
        "- Respond ONLY with ONE short follow-up question in the same language as the user.\n"
        "- Do NOT add extra explanation or summaries.\n"
        "- Do NOT ask more than one question."
    )


def build_triage_prompt_summary(
    user_message: str,
    language_code: str,
    symptoms_summary: str | None = None,
    answers_summary: str | None = None,
) -> str:
    """
    Prompt used AFTER the question limit is reached: generate a concise final triage summary.
    """
    lang_short = language_code.split("-")[0].lower() if language_code else "en"

    return (
        "You are an AI medical triage assistant for rural primary care.\n\n"
        f"User message (in their own language):\n{user_message}\n\n"
        f"Summary of symptoms and answers from the conversation:\n{symptoms_summary or user_message}\n"
        f"Additional details the user mentioned:\n{answers_summary or ''}\n\n"
        f"User language code: {language_code} (short: {lang_short}).\n\n"
        "Now you are in the FINAL SUMMARY PHASE, not the question phase.\n"
        "Behavior rules:\n"
        "1. ALWAYS respond in the SAME language as the user message.\n"
        "2. Do NOT ask any more questions.\n"
        "3. In a very short response (2–4 short sentences), provide:\n"
        "   - 1–3 possible conditions (make it clear this is not a confirmed diagnosis),\n"
        "   - a triage urgency level: LOW, MEDIUM, HIGH, or EMERGENCY,\n"
        "   - simple, safe advice and when to seek immediate or urgent care.\n"
        "4. Be clear and reassuring, and avoid medical jargon.\n"
        "5. If symptoms suggest an emergency (e.g., severe chest pain, difficulty breathing,\n"
        "   heavy bleeding, confusion, unconsciousness), clearly say it is an EMERGENCY and urge the user\n"
        "   to seek immediate care.\n\n"
        "Produce ONLY this short final summary in the user's language. Do NOT ask any more questions."
    )

