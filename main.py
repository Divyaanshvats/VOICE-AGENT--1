import os, json, asyncio, time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from groq import Groq
import edge_tts
import pygame

load_dotenv()  # reads GROQ_API_KEY from your .env file

LLM_MODEL = "openai/gpt-oss-20b"  # fast + free; if tool calls act up, use "openai/gpt-oss-120b"
STT_MODEL = "whisper-large-v3-turbo"
VOICE = "en-US-AriaNeural"
SAMPLE_RATE = 16000

client = Groq(api_key=os.environ["GROQ_API_KEY"])
pygame.mixer.init()

# ---------- Tools ----------
ORDERS = {"1001": "shipped, arriving Thursday", "1002": "processing", "1003": "delivered"}


def get_order_status(order_id: str) -> str:
    return ORDERS.get(str(order_id), "not found")


def book_appointment(date: str, time: str) -> str:
    return f"Appointment confirmed for {date} at {time}"


TOOL_FUNCS = {"get_order_status": get_order_status, "book_appointment": book_appointment}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_status",
        "description": "Look up the status of an order by its order ID",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "book_appointment",
        "description": "Book an appointment for a given date and time",
        "parameters": {"type": "object",
                       "properties": {"date": {"type": "string"}, "time": {"type": "string"}},
                       "required": ["date", "time"]}}},
]

# ---------- Conversation memory ----------
messages = [{"role": "system", "content":
             "You are a friendly phone support agent. Keep replies to 1-2 short spoken sentences. "
             "Use tools when needed. No markdown, no lists."}]


# ---------- Ears part 1: smart recording ----------
def record():
    print("\n🎙  Speak now...")
    chunk = int(0.1 * SAMPLE_RATE)   # listen in 0.1-second pieces
    threshold = 500                  # how loud counts as speech
    silence_limit = 1.0              # seconds of quiet = you're done
    max_seconds = 15
    frames, heard_speech, quiet = [], False, 0.0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=chunk) as stream:
        for _ in range(int(max_seconds / 0.1)):
            data, _overflow = stream.read(chunk)
            frames.append(data)
            if np.abs(data.astype(np.int32)).mean() > threshold:
                heard_speech, quiet = True, 0.0
            elif heard_speech:
                quiet += 0.1
                if quiet >= silence_limit:
                    break
    if not heard_speech:
        return None
    write("input.wav", SAMPLE_RATE, np.concatenate(frames))
    return "input.wav"


# ---------- Ears part 2: speech-to-text ----------
def transcribe(path: str) -> str:
    with open(path, "rb") as f:
        res = client.audio.transcriptions.create(file=("input.wav", f.read()), model=STT_MODEL)
    return res.text.strip()


# ---------- Brain: LLM + tool calling ----------
def think(user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})
    while True:
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS,
            tool_choice="auto", reasoning_effort="low")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"🔧 tool: {tc.function.name}({args})")
            result = TOOL_FUNCS[tc.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


# ---------- Mouth: text-to-speech ----------
def speak(text: str):
    asyncio.run(edge_tts.Communicate(text, VOICE).save("reply.mp3"))
    pygame.mixer.music.load("reply.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    pygame.mixer.music.unload()


# ---------- Main loop ----------
if __name__ == "__main__":
    print("Try: 'What's the status of order 1001?' or 'Book me for Friday at 3 PM.'")
    while True:
        input("Press Enter to talk...")
        path = record()
        if path is None:
            print("(didn't hear anything)")
            continue
        t0 = time.perf_counter()
        text = transcribe(path)
        t1 = time.perf_counter()
        if not text:
            print("(didn't catch that)")
            continue
        print(f"🧑 You: {text}")
        reply = think(text)
        t2 = time.perf_counter()
        print(f"🤖 Agent: {reply}")
        print(f"⏱ STT {t1-t0:.1f}s | LLM {t2-t1:.1f}s")
        speak(reply)