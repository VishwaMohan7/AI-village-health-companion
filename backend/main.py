import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.db import create_patient, save_session, log_alert, supabase
from backend.voice.stt import transcribe_audio
from backend.voice.tts import synthesize_audio
from backend.voice.translate import translate_text
from backend.voice.language import detect_language_from_text, normalize_to_app_language
from backend.agent.graph import agent_executor
from backend.agent.tools import sms_alert_tool
from backend.triage.orchestrator import TriageOrchestrator
from backend.triage.schemas import TriageRequest, TriageResponse
from backend.routes.chatbot import router as chatbot_router
from backend.services.ollama_service import generate_response

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup actions on startup
    logging.info("Starting up AI Village Health Companion Backend...")
    yield
    # Cleanup on shutdown
    logging.info("Shutting down...")

app = FastAPI(title="AI Village Health Companion", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router)

# Models
class StartSessionRequest(BaseModel):
    name: str
    phone: str
    language: str
    medical_history: str = ""

class EscalateRequest(BaseModel):
    patient_id: str
    message: str
    phone: str


triage_orchestrator = TriageOrchestrator()

# Endpoints
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend is running"}


@app.post("/triage/text", response_model=TriageResponse)
def triage_text(req: TriageRequest):
    """
    New structured triage endpoint (text).
    Returns the JSON shape requested in the redesign spec.
    """
    lang = req.language
    if not lang:
        lang = normalize_to_app_language(detect_language_from_text(req.text).code)
    return triage_orchestrator.triage_text(text=req.text, language=lang, session_id=req.session_id)


@app.post("/triage/voice", response_model=TriageResponse)
async def triage_voice(
    audio_file: UploadFile = File(...),
    language: str = Form("auto"),
    session_id: str = Form(None),
):
    """
    New structured triage endpoint (voice).
    - STT uses existing backend.voice.stt (Sarvam/Google fallback).
    - Language is auto-detected from transcript if 'auto'.
    """
    import os as os_module

    os_module.makedirs("temp", exist_ok=True)
    temp_audio_path = os_module.path.join("temp", f"triage_{uuid.uuid4()}.wav")
    with open(temp_audio_path, "wb") as f:
        f.write(await audio_file.read())

    # If language is "auto", pass a default to STT and then detect from text.
    stt_lang = "en-IN" if language == "auto" else language
    transcript = transcribe_audio(temp_audio_path, stt_lang)
    if not transcript:
        raise HTTPException(status_code=400, detail="Failed to transcribe audio.")

    if language == "auto":
        language = normalize_to_app_language(detect_language_from_text(transcript).code)

    return triage_orchestrator.triage_text(text=transcript, language=language, session_id=session_id)

@app.post("/session/start")
def start_session(req: StartSessionRequest):
    """Creates a new patient and initializes a session state."""
    patient = create_patient(req.name, req.phone, req.language, req.medical_history)
    if not patient:
        patient = {"id": str(uuid.uuid4())} # Mock for prototype
        
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "patient_id": patient["id"],
        "message": "Session started successfully."
    }

@app.post("/session/message")
async def process_message(
    session_id: str = Form(...),
    patient_id: str = Form("unknown"),
    language: str = Form("en-IN"),
    verify: bool = Form(False),
    audio_file: UploadFile = File(None),
    text_input: str = Form(None)
):
    """Processes user audio/text through the agent loop."""
    try:
        # Create temp directory if it doesn't exist
        import os as os_module
        os_module.makedirs("temp", exist_ok=True)
        
        # 1. Voice-to-Text
        if audio_file:
            temp_audio_path = os_module.path.join("temp", f"audio_{session_id}.wav")
            try:
                with open(temp_audio_path, "wb") as f:
                    content = await audio_file.read()
                    if not content:
                        raise HTTPException(status_code=400, detail="Audio file is empty.")
                    f.write(content)
                logging.info(f"Audio file saved: {temp_audio_path}")
                transcription = transcribe_audio(temp_audio_path, language)
            except IOError as e:
                logging.error(f"Error writing audio file: {e}")
                raise HTTPException(status_code=400, detail="Failed to save audio file.")
        else:
            transcription = text_input or ""
            
        if not transcription:
            raise HTTPException(status_code=400, detail="No audio or text input provided.")

        # Optional verification mode: echo back what was heard and ask for confirmation
        if verify:
            try:
                verify_prompt = (
                    "You are a speech-to-text verification assistant.\n\n"
                    f"Recognized text (from the user): \"{transcription}\"\n"
                    f"User language code: {language}.\n\n"
                    "Respond ONLY in the user's language by briefly repeating what you heard "
                    "and asking if it is correct. Keep it to one short question."
                )
                verify_text = generate_response(verify_prompt)
            except Exception as e:
                logging.error(f"Verification via Ollama failed: {e}")
                # Fall back to a simple echo if LLM fails
                verify_text = f"{transcription}\n\nIs this correct?"

            return {
                "mode": "verification",
                "transcription": transcription,
                "verify_message": verify_text,
            }

        # 2. Translation (to English for Agent if not already)
        english_text = translate_text(transcription, source_lang=language, target_lang="en")
        
        # 3. Agentic Loop Execution
        initial_state = {
            "patient_id": patient_id,
            "patient_language": language,
            "symptoms": english_text,
            "messages": [],
            "urgency_level": "LOW",
            "actions_taken": [],
            "next_node": "symptom_node"
        }
        
        final_state = agent_executor.invoke(initial_state)
        
        # 4. Response back to User Language
        messages = final_state.get("messages", [])
        if not messages:
            # If no messages by the end, provide a default response in the user's language
            agent_response_local = translate_text(
                "I have processed your symptoms. Please consult with a healthcare provider for further guidance.",
                source_lang="en",
                target_lang=language,
            )
        else:
            last_message = messages[-1]
            # Our agent graph already uses Ollama with the correct patient_language,
            # so last_message.content should already be in the user's language.
            agent_response_local = last_message.get("content", "") if isinstance(last_message, dict) else getattr(last_message, 'content', str(last_message))
        
        # 5. Text-to-Speech
        audio_url = synthesize_audio(agent_response_local, language)
        
        # 6. Save Session to DB
        actions = final_state.get("actions_taken", [])
        if isinstance(actions, list):
            actions_str = "; ".join(actions)
        else:
            actions_str = str(actions)
            
        save_session(
            patient_id=patient_id, 
            transcript=transcription, 
            urgency_level=final_state.get("urgency_level", "LOW"),
            action_taken=actions_str
        )
        
        return {
            "transcription": transcription,
            "english_translation": english_text,
            "text_response": agent_response_local,
            "audio_response_url": audio_url,
            "urgency_level": final_state.get("urgency_level", "LOW"),
            "actions_taken": final_state.get("actions_taken", [])
        }
        
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/transcribe")
async def voice_transcribe(audio_file: UploadFile = File(...), language: str = Form("en-IN")):
    """Standalone transcription + translation endpoint as requested in Step 4."""
    temp_audio_path = f"temp_standalone_{uuid.uuid4()}.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(await audio_file.read())
    
    transcription = transcribe_audio(temp_audio_path, language)
    english_text = translate_text(transcription, source_lang=language, target_lang="en")
    
    return {"transcription": transcription, "translation": english_text}

@app.get("/session/{session_id}/history")
def get_session_history(session_id: str):
    """Returns conversation history (mocked mostly for prototype unless using real Supabase)."""
    if not supabase:
        return {"history": [{"role": "agent", "content": "How can I help you today?"}]}
        
    try:
        # Simplistic fetch, assumes 'sessions' table tracks individual messages if designed that way
        # Real impl would fetch all rows for a session_id
        data, count = supabase.table('sessions').select('*').eq('id', session_id).execute()
        return {"history": data[1]}
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return {"history": []}

@app.post("/session/escalate")
def manual_escalate(req: EscalateRequest):
    """Manually triggers emergency SMS alert."""
    try:
        result = sms_alert_tool.invoke({"phone": req.phone, "message": req.message, "patient_id": req.patient_id})
        return {"status": "success", "detail": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
