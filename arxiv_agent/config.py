"""Environment loading and shared paths/constants for the agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("ARXIV_AGENT_DATA_DIR", BASE_DIR / "data")).resolve()
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.sqlite"

for _dir in (DATA_DIR, PDF_DIR, CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# arXiv Terms of Use (info.arxiv.org/help/api/tou.html): at most one request
# every 3 seconds, single connection at a time. Note this is 1/3s, not 3/s.
ARXIV_REQUEST_INTERVAL_SECONDS = 3.0
