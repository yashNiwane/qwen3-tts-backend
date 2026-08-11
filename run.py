"""
run.py — Local / Kaggle startup script
=======================================
Usage:
    python run.py                  # starts server on localhost:8000
    python run.py --ngrok          # starts server + opens ngrok tunnel
    python run.py --host 0.0.0.0   # custom host
    python run.py --port 8080       # custom port
"""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-TTS API Server")
    p.add_argument("--host",  default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port",  default=8000, type=int, help="Bind port (default: 8000)")
    p.add_argument("--ngrok", action="store_true",  help="Open an ngrok tunnel and print the public URL")
    p.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode only)")
    p.add_argument(
        "--model-size",
        choices=["0.6B", "1.7B"],
        default="1.7B",
        help="Which model size to load (default: 1.7B)",
    )
    return p.parse_args()


def start_ngrok(port: int) -> str:
    """Start an ngrok tunnel and return the public HTTPS URL."""
    from pyngrok import ngrok, conf

    # Load tokens from env / .env
    from dotenv import load_dotenv
    load_dotenv()

    authtoken = os.getenv("NGROK_AUTHTOKEN", "")
    if not authtoken:
        print("⚠️  NGROK_AUTHTOKEN not set — tunnel may be limited to 1 hour.")
    else:
        conf.get_default().auth_token = authtoken

    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url
    # Prefer https
    if public_url.startswith("http://"):
        public_url = public_url.replace("http://", "https://", 1)

    print(f"\n{'='*60}")
    print(f"  🌐  Public API URL : {public_url}")
    print(f"  📖  Swagger Docs  : {public_url}/docs")
    print(f"  ❤️   Health Check  : {public_url}/api/v1/health")
    print(f"{'='*60}\n")
    return public_url


def main():
    args = parse_args()

    # Inject model size into env before importing settings
    os.environ["MODEL_SIZE"] = args.model_size

    if args.ngrok:
        public_url = start_ngrok(args.port)
        # Write URL to a file so Kaggle notebooks can read it easily
        with open("ngrok_url.txt", "w") as f:
            f.write(public_url)

    import uvicorn
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
