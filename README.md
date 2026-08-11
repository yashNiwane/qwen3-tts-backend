# Qwen3-TTS Backend

A production-ready FastAPI backend for **Qwen3-TTS** — exposes REST API endpoints for:

| Feature | Model | Endpoint |
|---|---|---|
| **Voice Cloning** | `Qwen3-TTS-12Hz-1.7B-Base` | `POST /api/v1/voice-clone` |
| **Voice Designing** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `POST /api/v1/voice-design` |
| **Custom Voice** | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | `POST /api/v1/custom-voice` |

Designed to run on **Kaggle (T4 GPU)** and expose itself to the internet via **ngrok**.

---

## Project Structure

```
backend/
├── main.py                        # FastAPI app entrypoint
├── run.py                         # Startup script (uvicorn + ngrok)
├── requirements.txt
└── app/
    ├── core/
    │   ├── config.py              # Settings (pydantic-settings + .env)
    │   ├── model_manager.py       # Singleton model loader for all 3 variants
    │   └── audio_utils.py         # Audio save / encode helpers
    ├── schemas/
    │   └── common.py              # Shared Pydantic schemas
    └── api/v1/
        ├── health.py              # GET  /api/v1/health
        ├── info.py                # GET  /api/v1/info
        ├── voice_clone.py         # POST /api/v1/voice-clone  (+ /stream, /batch)
        ├── voice_design.py        # POST /api/v1/voice-design (+ /stream)
        └── custom_voice.py        # POST /api/v1/custom-voice (+ /stream, /batch, /speakers)
```

---

## API Endpoints

### Health & Info
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service + model status |
| `GET` | `/api/v1/info` | Available models, speakers, languages |

### Voice Cloning
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/voice-clone` | Clone a voice from reference audio |
| `POST` | `/api/v1/voice-clone/stream` | Stream cloned audio (97ms first packet) |
| `POST` | `/api/v1/voice-clone/batch` | Batch synthesis with prompt caching |

**Form fields:**
- `text` — Text to synthesise
- `language` — Target language (e.g. `English`, `Chinese`)
- `ref_audio` — Reference WAV/MP3 file (3–15 s)
- `ref_text` — Transcript of reference audio (strongly recommended)

### Voice Designing
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/voice-design` | Create a new voice from a text description |
| `POST` | `/api/v1/voice-design/stream` | Stream voice-designed audio |

**JSON body:**
```json
{
  "text": "Hello, welcome to the future of speech.",
  "language": "English",
  "voice_description": "A warm, elderly British gentleman with a slight rasp and calm cadence."
}
```

### Custom Voice (9 Preset Speakers)
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/custom-voice/speakers` | List all 9 preset speakers |
| `POST` | `/api/v1/custom-voice` | Synthesise with a preset speaker |
| `POST` | `/api/v1/custom-voice/stream` | Stream preset-speaker audio |
| `POST` | `/api/v1/custom-voice/batch` | Batch synthesis across multiple speakers |

**Available speakers:** `Vivian`, `Serena`, `Uncle_Fu`, `Chelsie`, `Ethan`, `Cherry`, `Ryan`, `Aura`, `Daniel`

**JSON body:**
```json
{
  "text": "I'm thrilled to announce this!",
  "language": "English",
  "speaker": "Ryan",
  "instruction": "Speak in an energetic and excited tone."
}
```

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run without ngrok
python run.py

# Run with ngrok tunnel (reads NGROK_AUTHTOKEN from .env)
python run.py --ngrok

# Use smaller 0.6B model (less VRAM)
python run.py --model-size 0.6B --ngrok
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Environment Variables (`.env`)

```env
NGROK_AUTHTOKEN=your_ngrok_authtoken
NGROK_API_ID=your_ngrok_api_id
NGROK_API_TOKEN=your_ngrok_api_token

# Optional overrides
MODEL_SIZE=1.7B
USE_GPU=true
USE_FLASH_ATTENTION=true
MODEL_CACHE_DIR=/kaggle/working/models
AUDIO_OUTPUT_DIR=/kaggle/working/audio_outputs
```

---

## Supported Languages

`English`, `Chinese`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian`

## Hardware Requirements

| Model | VRAM (bfloat16) | Recommended GPU |
|---|---|---|
| 1.7B | ~6–8 GB | T4, RTX 3060, A10G |
| 0.6B | ~3–4 GB | GTX 1660, RTX 2060 |

Kaggle's free T4 (16 GB VRAM) runs all three 1.7B models simultaneously.
