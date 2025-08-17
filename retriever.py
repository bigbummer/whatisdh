import os
import io
import json
import hashlib
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from pypdf import PdfReader

from openai import OpenAI
import config as C

# Гарантируем наличие директорий хранения
os.makedirs(C.DATA_DIR, exist_ok=True)
os.makedirs(C.STORAGE_DIR, exist_ok=True)


@dataclass
class Doc:
    id: str
    text: str
    source: str
    section: str


def _openai_client() -> OpenAI:
    if not C.OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")
    return OpenAI(api_key=C.OPENAI_API_KEY)


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 3072), dtype="float32")
    oai = _openai_client()
    BATCH = 64
    embs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = oai.embeddings.create(model=C.EMBED_MODEL, input=batch)
        embs.extend([np.array(e.embedding, dtype="float32") for e in resp.data])
    return np.vstack(embs)


def normalize(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


class FaissStore:
    def __init__(self, d: int):
        self.d = d
        self.index = faiss.IndexFlatIP(d)
        self.meta: List[Dict[str, Any]] = []

    def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]):
        if vecs.size == 0:
            return
        self.index.add(vecs)
        self.meta.extend(metas)

    def search(self, q: np.ndarray, k: int=None) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index.ntotal == 0:
            return []
        k = int(k or C.RETRIEVAL_K)
        # ограничим k, чтобы не запрашивать больше, чем есть
        k = max(1, min(k, max(1, self.index.ntotal)))
        D, I = self.index.search(q, k)

        # Приведём формы к ожидаемым (1, k)
        D = np.asarray(D)
        I = np.asarray(I)
        if D.ndim == 1:
            D = D.reshape(1, -1)
        if I.ndim == 1:
            I = I.reshape(1, -1)

        out: List[Tuple[float, Dict[str, Any]]] = []
        # Безопасно итерируем по первой строке
        for score, idx in zip(D[0].tolist(), I.tolist()):
            # idx может прийти как numpy тип — приведём к int
            try:
                idx_int = int(idx)
            except Exception:
                # если FAISS вернул что-то неожиданное — пропускаем
                continue
            if idx_int < 0:
                # -1 означает пустой слот
                continue
            if idx_int >= len(self.meta):
                # защита от несоответствия размеров
                continue
            out.append((float(score), self.meta[idx_int]))
        return out

    def save(self):
        faiss.write_index(self.index, C.INDEX_PATH)
        with open(C.META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load():
        if not (os.path.exists(C.INDEX_PATH) and os.path.exists(C.META_PATH)):
            return None
        store = FaissStore(d=0)
        store.index = faiss.read_index(C.INDEX_PATH)
        with open(C.META_PATH, "r", encoding="utf-8") as f:
            store.meta = json.load(f)
        store.d = store.index.d
        return store



# Google Drive utilities

def _list_and_download_public_folder(folder_url: str, out_dir: str) -> List[str]:
    """
    Публичная папка: пытаемся скачать через gdown.
    Если упрёмся в лимит (более 50 файлов) или другую ошибку — не падаем,
    просто работаем с тем, что уже успели скачать/что есть локально.
    """
    if not folder_url:
        return []
    import gdown
    os.makedirs(out_dir, exist_ok=True)

    try:
        # gdown имеет лимит ~50 файлов на папку, при превышении кидает FolderContentsMaximumLimitError
        gdown.download_folder(url=folder_url, output=out_dir, quiet=True, use_cookies=False)
    except Exception as e:
        # Не прерываем выполнение: переходим к использованию локально уже имеющихся PDF
        # При желании можно добавить логирование:
        # print(f"[warn] gdown error on public folder: {e}. Proceeding with local PDFs in {out_dir}")
        pass

    # Собираем локальные PDF, даже если загрузка прервалась
    pdfs: List[str] = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    return sorted(list(set(pdfs)))


def _list_and_download_private_folder(folder_id: str, out_dir: str, service_json: str) -> List[str]:
    """
    Приватная папка: скачиваем через Google Drive API (Service Account).
    """
    if not folder_id or not os.path.exists(service_json):
        return []
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_file(service_json, scopes=SCOPES)
    drive = build("drive", "v3", credentials=creds)

    os.makedirs(out_dir, exist_ok=True)
    pdf_paths: List[str] = []
    page_token = None
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    while True:
        resp = drive.files().list(
            q=query,
            fields="files(id, name, modifiedTime), nextPageToken",
            pageToken=page_token
        ).execute()
        files = resp.get("files", [])
        for f in files:
            file_id = f["id"]
            name = f["name"]
            local_path = os.path.join(out_dir, name)
            if not os.path.exists(local_path):
                request = drive.files().get_media(fileId=file_id)
                fh = io.FileIO(local_path, "wb")
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            pdf_paths.append(local_path)
        page_token = resp.get("nextPageToken", None)
        if page_token is None:
            break
    return sorted(list(set(pdf_paths)))


def extract_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n".join(parts)
    except Exception:
        return ""


def infer_section_from_filename(fname: str) -> str:
    low = fname.lower()
    if "faculty" in low or "преподав" in low:
        return "Faculty"
    if "resource" in low or "ресурс" in low:
        return "Program Resources"
    if "dh" in low or "цифров" in low or "digital" in low:
        return "Digital Humanities"
    return "PDF"


def chunk_text(text: str, source: str, section: str, chunk_chars: int, overlap: int) -> List[Doc]:
    text = " ".join(text.split())
    docs: List[Doc] = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_chars]
        if not chunk.strip():
            break
        doc_id = hashlib.md5((source + str(i)).encode()).hexdigest()[:16]
        docs.append(Doc(id=doc_id, text=chunk, source=os.path.basename(source), section=section))
        i += max(1, chunk_chars - overlap)
    return docs


def download_pdfs_from_drive() -> List[str]:
    pdf_paths: List[str] = []
    if C.DRIVE_MODE == "public":
        pdf_paths += _list_and_download_public_folder(C.DRIVE_FOLDER_1, os.path.join(C.DATA_DIR, "folder1"))
        pdf_paths += _list_and_download_public_folder(C.DRIVE_FOLDER_2, os.path.join(C.DATA_DIR, "folder2"))
    else:
        pdf_paths += _list_and_download_private_folder(
            C.DRIVE_FOLDER_ID_1, os.path.join(C.DATA_DIR, "folder1"), C.GOOGLE_SERVICE_ACCOUNT_JSON
        )
        pdf_paths += _list_and_download_private_folder(
            C.DRIVE_FOLDER_ID_2, os.path.join(C.DATA_DIR, "folder2"), C.GOOGLE_SERVICE_ACCOUNT_JSON
        )
    return pdf_paths


def build_index_from_pdfs(pdfs: List[str]) -> FaissStore:
    docs: List[Doc] = []
    for pdf in pdfs:
        txt = extract_pdf_text(pdf)
        section = infer_section_from_filename(os.path.basename(pdf))
        docs += chunk_text(txt, source=pdf, section=section, chunk_chars=C.CHUNK_CHARS, overlap=C.CHUNK_OVERLAP)

    if not docs:
        placeholder = "Корпус пуст. Добавьте PDF в указанные папки Google Drive и перезапустите."
        docs = [Doc(id="placeholder", text=placeholder, source="placeholder", section="Notice")]

    texts = [d.text for d in docs]
    embs = normalize(embed_texts(texts))
    if embs.size == 0:
        # fallback размерности — text-embedding-3-large = 3,072
        embs = np.zeros((1, 3072), dtype="float32")
    store = FaissStore(d=embs.shape[1])
    metas = [{"id": d.id, "text": d.text, "source": d.source, "section": d.section} for d in docs]
    store.add(embs, metas)
    store.save()
    return store


def load_or_rebuild_index() -> FaissStore:
    # Текущая "сигнатура" локального кэша PDF: размер и mtime
    current: Dict[str, Dict[str, Any]] = {}
    for root, _, files in os.walk(C.DATA_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                stat = os.stat(path)
                current[path] = {"size": stat.st_size, "mtime": int(stat.st_mtime)}
    prev: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(C.PDF_CACHE):
        with open(C.PDF_CACHE, "r", encoding="utf-8") as fp:
            prev = json.load(fp)
    unchanged = (current == prev)

    if os.path.exists(C.INDEX_PATH) and os.path.exists(C.META_PATH) and unchanged:
        store = FaissStore.load()
        if store:
            return store

    # Иначе — пробуем скачать (с обработкой лимита в публичном режиме) и пересобираем
    pdfs = download_pdfs_from_drive()
    store = build_index_from_pdfs(pdfs)
    with open(C.PDF_CACHE, "w", encoding="utf-8") as fp:
        json.dump(current, fp, ensure_ascii=False, indent=2)
    return store


def retrieve(store: FaissStore, query: str, k: int = None) -> List[Dict[str, Any]]:
    oai = _openai_client()
    emb = oai.embeddings.create(model=C.EMBED_MODEL, input=[query])
    q = normalize(np.array([emb.data[0].embedding], dtype="float32"))
    k = k or C.RETRIEVAL_K
    results = store.search(q, k=k)
    ctxs: List[Dict[str, Any]] = []
    for score, meta in results:
        m = dict(meta)
        m["score"] = round(score, 4)
        ctxs.append(m)
    return ctxs
