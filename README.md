# Village Health Care Companion

## Overview

This project is a Streamlit-based voice assistant for basic health guidance. It records speech from the browser, converts the spoken input to text, checks a small local symptom rule set, and falls back to a locally running Ollama model for broader responses. Replies are generated and spoken in the language selected by the user.

## Current Architecture

- `streamlit` provides the web UI.
- `streamlit-mic-recorder` captures microphone input and performs speech-to-text.
- `streamlit-TTS` and `gTTS` convert the assistant response to audio.
- `requests` sends chat requests to a local Ollama server.
- A small in-memory symptom dictionary handles simple known cases before the app asks Ollama.
- Ollama can generate a limited set of follow-up questions before final analysis.

## Features

- Voice input from the browser
- Response playback in the selected language
- Basic symptom keyword matching for a few predefined cases
- Local LLM inference through Ollama
- Limited voice follow-up questions for deeper symptom analysis
- User profile inputs for age and weight
- Language choices limited to English, Hindi, and Kannada
- Configurable Ollama base URL and model from the sidebar
- Configurable cap for follow-up questions

## Ollama Setup

Install Ollama and make sure the local server is running.

Example:

```sh
ollama serve
ollama pull llama3
```

By default, the app expects:

- Base URL: `http://127.0.0.1:11434`
- Model: `llama3`

You can also override these with environment variables:

```sh
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3
```

## Installation

Create and activate a virtual environment:

```sh
python -m venv venv
venv\Scripts\activate
```

Install Python dependencies:

```sh
pip install -r requirements.txt
```

Make sure `ffmpeg` and `ffprobe` are available on your system `PATH`, because browser audio playback depends on them.

## Run

```sh
streamlit run app.py
```

## Notes

- This is a prototype and should not be treated as a diagnostic medical system.
- The built-in symptom checker is intentionally small and only covers a few hardcoded examples.
- Follow-up questions are intentionally capped to keep the interaction short and focused.
- Follow-up questions are spoken aloud, and the app prints the recognized speech transcript for each answer.
- The model is instructed to answer in the exact language chosen in the sidebar.
- For urgent, severe, or persistent symptoms, users should be directed to a qualified medical professional.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
