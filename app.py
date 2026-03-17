import os
import shutil
import warnings

import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_TTS import auto_play, text_to_audio
from streamlit_mic_recorder import speech_to_text

load_dotenv()

# Silence noisy pydub ffmpeg warnings; playback still depends on ffmpeg being available.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
UNKNOWN_SYMPTOM_MESSAGE = (
    "I'm not sure about this symptom. Please consult a doctor for a more accurate diagnosis."
)
DEFAULT_MAX_FOLLOW_UP_QUESTIONS = 3

conversation_history = []
user_profile = {"language": "en", "language_name": "English", "preferences": {}, "age": None, "weight": None}

medical_conditions_db = {
    "fever": (
        "You might be experiencing a viral infection or flu. Please monitor your temperature "
        "and stay hydrated. If it persists, consult a healthcare professional."
    ),
    "headache": (
        "A headache could be due to stress, dehydration, or lack of sleep. Rest and drink "
        "plenty of fluids. If it lasts longer than a few days, see a doctor."
    ),
}

language_mapping = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
}


def initialize_session_state():
    defaults = {
        "captured_input": "",
        "response_language": "en",
        "follow_up_questions": [],
        "follow_up_answers": {},
        "follow_up_index": 0,
        "spoken_question_keys": [],
        "analysis_ready": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_system_message(age=None, weight=None):
    details = []
    if age is not None:
        details.append(f"The user is {age} years old.")
    if weight is not None:
        details.append(f"The user weighs {weight} kg.")

    profile_context = " ".join(details)
    return (
        "You are a calm multilingual health assistant. "
        f"Always reply in {user_profile['language_name']}. "
        "Give brief, practical guidance, avoid claiming certainty, and encourage professional "
        "medical care for severe, urgent, or persistent symptoms. "
        "Do not present yourself as a doctor. "
        f"{profile_context}".strip()
    )


def call_ollama(messages, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, timeout=60):
    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
    }

    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    response_data = response.json()
    return response_data["message"]["content"].strip()


def build_case_summary(initial_input, follow_up_questions, follow_up_answers):
    lines = [f"Initial symptom description: {initial_input}"]
    for question in follow_up_questions:
        answer = follow_up_answers.get(question, "").strip() or "No answer provided."
        lines.append(f"Follow-up question: {question}")
        lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def parse_follow_up_questions(raw_text, limit):
    questions = []
    for line in raw_text.splitlines():
        cleaned = line.strip().lstrip("-*").strip()
        if not cleaned:
            continue
        if ". " in cleaned[:4]:
            cleaned = cleaned.split(". ", 1)[1].strip()
        if cleaned.endswith("?"):
            questions.append(cleaned)
        else:
            questions.append(f"{cleaned}?")
        if len(questions) >= limit:
            break
    return questions


def generate_follow_up_questions(user_input, age=None, weight=None, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, limit=3):
    """Generate a small capped list of clarifying questions for deeper symptom analysis."""
    prompt = (
        f"User symptom description: {user_input}\n"
        f"User age: {age}\n"
        f"User weight: {weight}\n\n"
        f"Ask up to {limit} short follow-up questions that would help clarify likely causes, "
        "severity, duration, and red flags. Return only the questions, one per line, with no intro."
    )

    try:
        raw_text = call_ollama(
            messages=[
                {"role": "system", "content": build_system_message(age, weight)},
                {"role": "user", "content": prompt},
            ],
            base_url=base_url,
            model=model,
            timeout=45,
        )
        questions = parse_follow_up_questions(raw_text, limit)
        if questions:
            return questions
    except requests.RequestException:
        pass
    except (KeyError, TypeError, ValueError):
        pass

    return [
        "How long have you had these symptoms?",
        "How severe are the symptoms right now?",
        "Do you have any other symptoms such as fever, breathing trouble, vomiting, or chest pain?",
    ][:limit]


def get_response(user_input, age=None, weight=None, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL):
    """Get an AI response from a locally running Ollama model."""
    conversation_history.append({"role": "user", "content": user_input})

    try:
        assistant_message = call_ollama(
            messages=[
                {"role": "system", "content": build_system_message(age, weight)},
                *conversation_history,
            ],
            base_url=base_url,
            model=model,
            timeout=60,
        )
    except requests.RequestException as exc:
        assistant_message = (
            "I couldn't reach your local Ollama server. Please make sure Ollama is running "
            f"at {base_url} and that the model '{model}' is available. Details: {exc}"
        )
    except (KeyError, TypeError, ValueError) as exc:
        assistant_message = f"Ollama returned an unexpected response format. Details: {exc}"

    st.write(f"Assistant: {assistant_message}")
    conversation_history.append({"role": "assistant", "content": assistant_message})
    return assistant_message


def check_symptom(symptom_description):
    """Check symptoms against a tiny local rule set before asking Ollama."""
    symptom_text = symptom_description.lower()
    for symptom, advice in medical_conditions_db.items():
        if symptom in symptom_text:
            return advice
    return UNKNOWN_SYMPTOM_MESSAGE


def reset_follow_up_state():
    st.session_state["follow_up_questions"] = []
    st.session_state["follow_up_answers"] = {}
    st.session_state["follow_up_index"] = 0
    st.session_state["spoken_question_keys"] = []
    st.session_state["analysis_ready"] = False


def capture_user_input(max_follow_up_questions, ollama_base_url, ollama_model):
    st.write("Press the button below and speak.")

    user_input = speech_to_text(
        language=user_profile["language"],
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True,
    )

    if not user_input:
        return

    st.session_state["captured_input"] = user_input
    st.session_state["response_language"] = user_profile["language"]
    reset_follow_up_state()
    st.session_state["follow_up_questions"] = generate_follow_up_questions(
        user_input,
        age=user_profile["age"],
        weight=user_profile["weight"],
        base_url=ollama_base_url,
        model=ollama_model,
        limit=max_follow_up_questions,
    )


def play_question_audio_once(question, question_index):
    question_audio_key = f"follow_up_question_audio_{question_index}"
    if question_audio_key in st.session_state["spoken_question_keys"]:
        return

    question_audio = text_to_audio(question, st.session_state["response_language"])
    auto_play(question_audio, key=question_audio_key)
    st.session_state["spoken_question_keys"].append(question_audio_key)


def render_follow_up_voice_flow():
    if not st.session_state["captured_input"]:
        return None

    st.write(f"You: {st.session_state['captured_input']}")
    st.write(f"Response language: {st.session_state['response_language']}")

    questions = st.session_state["follow_up_questions"]
    if not questions:
        st.session_state["analysis_ready"] = True
        return {}

    st.subheader("Voice Follow-up")

    for previous_index, previous_question in enumerate(
        questions[: st.session_state["follow_up_index"]],
        start=1,
    ):
        st.write(f"{previous_index}. {previous_question}")
        st.write(
            f"Your answer: {st.session_state['follow_up_answers'].get(previous_question, 'No answer provided.')}"
        )

    current_index = st.session_state["follow_up_index"]
    if current_index >= len(questions):
        st.session_state["analysis_ready"] = True
        return st.session_state["follow_up_answers"]

    current_question = questions[current_index]
    st.write(f"{current_index + 1}. {current_question}")
    play_question_audio_once(current_question, current_index)

    answer = speech_to_text(
        language=st.session_state["response_language"],
        start_prompt="Answer in voice",
        stop_prompt="Stop answer",
        just_once=True,
        key=f"follow_up_voice_{current_index}",
    )

    if answer and current_question not in st.session_state["follow_up_answers"]:
        st.session_state["follow_up_answers"][current_question] = answer
        st.write(f"Your answer: {answer}")
        st.session_state["follow_up_index"] = current_index + 1
        if st.session_state["follow_up_index"] >= len(questions):
            st.session_state["analysis_ready"] = True
        st.rerun()

    if current_question in st.session_state["follow_up_answers"]:
        st.write(f"Your answer: {st.session_state['follow_up_answers'][current_question]}")

    return st.session_state["follow_up_answers"]


def analyze_case(ollama_base_url, ollama_model):
    if not st.session_state["captured_input"] or not st.session_state["analysis_ready"]:
        return

    enriched_input = build_case_summary(
        st.session_state["captured_input"],
        st.session_state["follow_up_questions"],
        st.session_state["follow_up_answers"],
    )
    response_text = check_symptom(enriched_input)
    if response_text == UNKNOWN_SYMPTOM_MESSAGE:
        response_text = get_response(
            enriched_input,
            user_profile["age"],
            user_profile["weight"],
            base_url=ollama_base_url,
            model=ollama_model,
        )

    audio = text_to_audio(response_text, st.session_state["response_language"])
    auto_play(audio)


st.title("Village Health Care Companion")
st.caption("Local LLM mode powered by Ollama")

if not FFMPEG_AVAILABLE:
    st.warning("ffmpeg/ffprobe were not found on PATH. Text-to-speech playback may fail.")

st.sidebar.header("User Profile")
user_language = st.sidebar.selectbox("Select Your Language", list(language_mapping.keys()))
user_profile["language_name"] = user_language
user_profile["language"] = language_mapping[user_language]
user_profile["age"] = st.sidebar.number_input("Enter Your Age", min_value=0, max_value=120, value=30)
user_profile["weight"] = st.sidebar.number_input("Enter Your Weight (kg)", min_value=0, value=70)

st.sidebar.header("Ollama Settings")
ollama_base_url = st.sidebar.text_input("Ollama Base URL", value=OLLAMA_BASE_URL)
ollama_model = st.sidebar.text_input("Ollama Model", value=OLLAMA_MODEL)
max_follow_up_questions = st.sidebar.slider(
    "Max Follow-up Questions",
    min_value=1,
    max_value=5,
    value=DEFAULT_MAX_FOLLOW_UP_QUESTIONS,
)
st.sidebar.caption("Example: base URL http://127.0.0.1:11434, model llama3")

initialize_session_state()
capture_user_input(max_follow_up_questions, ollama_base_url, ollama_model)
render_follow_up_voice_flow()
analyze_case(ollama_base_url, ollama_model)
