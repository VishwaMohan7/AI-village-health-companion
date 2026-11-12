# Village Health Care Companion
### A Multilingual, Voice-Enabled Health Guidance Prototype

A Streamlit-based voice assistant that accepts spoken health-related queries, checks a small set of predefined symptom rules, and uses a locally running Ollama model for broader responses. The assistant can speak its replies in the language selected by the user.

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-111827?style=flat)
![License](https://img.shields.io/badge/License-MIT-58A6FF?style=flat)

</div>

---

## Overview

Village Health Care Companion is a prototype designed to make basic health guidance accessible through voice interaction. Users can speak through the browser, have their speech converted to text, and receive a response that is played back as audio.

The application first checks a small in-memory symptom dictionary for a few known cases. For broader queries, it sends the conversation to a locally running Ollama model. The assistant can ask a limited number of follow-up questions before providing its response.

## Features

- **Voice input:** capture microphone audio through the browser.
- **Speech-to-text:** convert spoken input into a text transcript.
- **Basic symptom matching:** check a small set of predefined symptom keywords and cases.
- **Local LLM responses:** use Ollama for broader health-related queries.
- **Follow-up questions:** ask a capped number of additional questions for further context.
- **Text-to-speech:** play assistant responses aloud.
- **Multilingual interaction:** choose English, Hindi, or Kannada.
- **User profile inputs:** provide age and weight as profile context.
- **Configurable model settings:** set the Ollama base URL and model from the sidebar or environment variables.

## How It Works

1. **Speak:** the user records a query through the browser microphone.
2. **Transcribe:** the speech input is converted into text.
3. **Check:** the app checks its small local symptom rule set for a known case.
4. **Ask follow-ups:** when applicable, Ollama can generate a limited set of follow-up questions.
5. **Respond:** the assistant generates a response in the language selected in the sidebar.
6. **Play audio:** the response is converted to speech and played back.

## Architecture & Tools

| Component | Role |
|---|---|
| Streamlit | Web application interface |
| streamlit-mic-recorder | Browser microphone capture and speech-to-text |
| Ollama | Local LLM inference |
| requests | Sends chat requests to the local Ollama server |
| streamlit-TTS and gTTS | Converts assistant responses to audio |
| In-memory symptom dictionary | Handles a few predefined symptom cases |

## Requirements

- Python
- Ollama installed and running locally
- `ffmpeg` and `ffprobe` available on your system `PATH`
- Python dependencies listed in `requirements.txt`

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/VishwaMohan7/AI-village-health-companion.git
cd AI-village-health-companion
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Ollama

Install Ollama, start its local server, and download the model:

```bash
ollama serve
ollama pull llama3
```

By default, the application expects:

- **Ollama base URL:** `http://127.0.0.1:11434`
- **Model:** `llama3`

You can override these defaults with environment variables:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3
```

### 5. Check audio dependencies

Ensure both `ffmpeg` and `ffprobe` are installed and accessible through your system `PATH`.

## Run the Application

Start the Streamlit app from the project directory:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## Important Limitations & Safety

- This project is a prototype for basic health guidance, **not a medical diagnostic system**.
- The built-in symptom checker covers only a small number of hardcoded examples.
- Follow-up questions are capped to keep the interaction short and focused.
- Follow-up questions are spoken aloud, and the recognized speech transcript is printed by the app.
- The model is instructed to respond in the exact language selected in the sidebar.
- For urgent, severe, or persistent symptoms, seek guidance from a qualified medical professional.

Do not use this application as a substitute for professional medical evaluation.

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

### Making basic health guidance more accessible through voice and language.

</div>
