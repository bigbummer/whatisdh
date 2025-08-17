import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

DRIVE_MODE = os.getenv("DRIVE_MODE", "public").strip().lower()
DRIVE_FOLDER_1 = os.getenv("DRIVE_FOLDER_1", "")
DRIVE_FOLDER_2 = os.getenv("DRIVE_FOLDER_2", "")

DRIVE_FOLDER_ID_1 = os.getenv("DRIVE_FOLDER_ID_1", "")
DRIVE_FOLDER_ID_2 = os.getenv("DRIVE_FOLDER_ID_2", "")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

DATA_DIR = "data_pdfs"
STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "index.faiss")
META_PATH = os.path.join(STORAGE_DIR, "meta.json")
PDF_CACHE = os.path.join(STORAGE_DIR, "pdf_cache.json")

# Safety checks at import time are avoided to allow tooling (e.g., building index separately)
