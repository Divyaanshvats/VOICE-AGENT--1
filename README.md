# 🎙️ Voice Agent — AI-Powered Phone Support Assistant

A lightweight, real-time voice agent built in Python that listens to what you say, thinks using a large language model, and talks back — all within a few seconds. Think of it as a smart phone support bot that lives entirely on your machine.

---

## What It Does

This project brings together three core capabilities to create a fully conversational AI agent:

- **Listens** — Records your voice from the microphone and automatically stops when you go quiet.
- **Understands** — Transcribes your speech to text using OpenAI''s Whisper model (via Groq''s ultra-fast API).
- **Responds** — Passes the text to an LLM, which can answer questions, call tools, and reply in natural spoken language.
- **Speaks** — Converts the agent''s reply to audio using Microsoft Edge TTS and plays it back instantly.

The whole pipeline runs in a loop, so you can have a back-and-forth conversation just like a real phone call.

---

## How It Works — Under the Hood

```
You speak → [Microphone] → input.wav
                              ↓
                    [Whisper STT via Groq]
                              ↓
                       Transcribed text
                              ↓
                   [LLM (GPT-OSS via Groq)]
                    ↙              ↘
              Tool call?        Plain reply
             ↙      ↘               ↓
  get_order   book_appointment   reply.mp3
   _status()       ()               ↓
       ↘           ↙         [Edge TTS + Pygame]
        Tool result                 ↓
             ↓                  You hear it
        Back to LLM
```

### Key Components

| Layer | Technology | Purpose |
|---|---|---|
| Speech-to-Text | `whisper-large-v3-turbo` via Groq | Converts your voice to text |
| Language Model | `openai/gpt-oss-20b` via Groq | Understands intent and generates replies |
| Tool Calling | Groq function calling API | Lets the agent take real actions |
| Text-to-Speech | Microsoft Edge TTS (`edge_tts`) | Produces a natural-sounding voice |
| Audio Playback | `pygame.mixer` | Plays the reply audio |
| Recording | `sounddevice` + `numpy` | Captures mic input with silence detection |

---

## Built-In Tools

The agent ships with two example tools that the LLM can call automatically when needed:

### 1. `get_order_status(order_id)`
Looks up the status of a customer order. Currently backed by a small in-memory dictionary — easy to swap for a real database or API.

> Example: *"What''s the status of order 1001?"*

### 2. `book_appointment(date, time)`
Confirms an appointment booking for a given date and time.

> Example: *"Book me for Friday at 3 PM."*

Adding more tools is as simple as writing a Python function and registering it in the `TOOLS` list.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A [Groq API key](https://console.groq.com/) (free tier is sufficient)
- A working microphone

### Installation

```bash
# Clone the repository
git clone https://github.com/Divyaanshvats/VOICE-AGENT--1.git
cd VOICE-AGENT--1

# (Optional but recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS / Linux

# Install dependencies
pip install groq edge-tts pygame sounddevice scipy numpy python-dotenv
```

### Configuration

Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is already excluded via `.gitignore`.

### Running the Agent

```bash
python main.py
```

Press **Enter** when prompted, then speak. The agent will transcribe your words, reason about them, and reply out loud.

---

## Project Structure

```
VOICE-AGENT--1/
│
├── main.py          # Full pipeline: record → transcribe → think → speak
├── .env             # Your API keys (not tracked by Git)
├── .gitignore       # Prevents secrets and build files from being committed
└── README.md        # You are here
```

---

## Configuration Options

You can tweak a few constants at the top of `main.py` to adjust behaviour:

| Constant | Default | Description |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-oss-20b` | The language model used for reasoning |
| `STT_MODEL` | `whisper-large-v3-turbo` | The Whisper variant for transcription |
| `VOICE` | `en-US-AriaNeural` | The Edge TTS voice persona |
| `SAMPLE_RATE` | `16000` | Microphone sample rate in Hz |
| `threshold` | `500` | Amplitude level that counts as speech |
| `silence_limit` | `1.0` | Seconds of quiet before recording stops |

---

## Latency

The agent prints timing stats after every turn so you can see where time is being spent:

```
⏱ STT 0.8s | LLM 1.2s
```

Groq''s inference hardware is significantly faster than standard cloud providers, which keeps the total round-trip time under 3 seconds in most cases.

---

## Extending the Agent

### Add a new tool

1. Write a Python function that returns a string result.
2. Add it to `TOOL_FUNCS`.
3. Describe it in the `TOOLS` list using the standard function-calling schema.

The LLM will automatically decide when to use it based on what you say.

### Change the system prompt

Edit the `content` field inside the `messages` list at the top of `main.py` to give the agent a different personality, domain focus, or set of constraints.

### Switch to a different voice

Browse available voices by running:

```bash
python -c "import asyncio, edge_tts; asyncio.run(edge_tts.list_voices())"
```

Then update the `VOICE` constant with your preferred voice name.

---

## Dependencies

| Package | Purpose |
|---|---|
| `groq` | Official Groq Python SDK — LLM and Whisper |
| `edge-tts` | Microsoft Edge neural TTS |
| `pygame` | Audio playback |
| `sounddevice` | Microphone input |
| `scipy` | Writing WAV files |
| `numpy` | Audio signal processing |
| `python-dotenv` | Loading environment variables from `.env` |

---

## Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request for bug fixes, new tools, or quality-of-life improvements.

---

## License

This project is open source. Use it, modify it, and build on it freely.
