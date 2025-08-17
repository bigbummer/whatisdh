import os
import io
import json
import glob
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from pypdf import PdfReader

from openai import OpenAI
import config as C

# Директории для данных и индекса
os.makedirs(C.DATA_DIR, exist_ok=True)
os.makedirs(C.STORAGE_DIR, exist_ok=True)


@dataclass
class Doc:
    id: str
    text: str
    source: str
    section: str


# ---------------------------
# OpenAI helpers
# ---------------------------
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


# ---------------------------
# FAISS store (robust search)
# ---------------------------
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

    def search(self, q: np.ndarray, k: int = None) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index.ntotal == 0:
            return []
        k = int(k or C.RETRIEVAL_K)
        k = max(1, min(k, max(1, self.index.ntotal)))
        D, I = self.index.search(q, k)

        D = np.asarray(D)
        I = np.asarray(I)
        if D.ndim == 1:
            D = D.reshape(1, -1)
        if I.ndim == 1:
            I = I.reshape(1, -1)

        out: List[Tuple[float, Dict[str, Any]]] = []
        # Берём первую строку результатов
        for score, idx in zip(D[0].tolist(), I.tolist()):
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if idx_int < 0:
                continue
            if idx_int >= len(self.meta):
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


# ---------------------------
# Google Drive loaders (public/private)
# ---------------------------
def _list_and_download_public_folder(folder_url: str, out_dir: str) -> List[str]:
    """
    Публичная папка: пытаемся скачать через gdown.
    Любые ошибки не валят процесс — работаем с тем, что есть локально.
    """
    if not folder_url:
        return []
    print(f"[Drive public] Fetch: {folder_url} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    try:
        import gdown
        gdown.download_folder(url=folder_url, output=out_dir, quiet=True, use_cookies=False)
    except Exception as e:
        print(f"[Drive public][WARN] Failed: {e}")
        print("Проверьте формат ссылки (/drive/folders/<ID>) и права (Anyone with the link: Viewer).")

    pdfs: List[str] = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    return sorted(list(set(pdfs)))


def _list_and_download_private_folder(folder_id: str, out_dir: str, service_json: str) -> List[str]:
    print(f"[Drive private] List: {folder_id} -> {out_dir} (key exists: {os.path.exists(service_json)})")
    if not folder_id or not os.path.exists(service_json):
        print("[Drive private][WARN] Missing folder_id or service account key; skipping.")
        return []

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except Exception as e:
        print(f"[Drive private][WARN] Google API libs import failed: {e}")
        return []

    try:
        SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
        creds = service_account.Credentials.from_service_account_file(service_json, scopes=SCOPES)
        drive = build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"[Drive private][WARN] Failed to init Drive client: {e}")
        return []

    os.makedirs(out_dir, exist_ok=True)
    pdf_paths: List[str] = []
    page_token = None
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    try:
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
                try:
                    if not os.path.exists(local_path):
                        request = drive.files().get_media(fileId=file_id)
                        fh = io.FileIO(local_path, "wb")
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                    pdf_paths.append(local_path)
                except Exception as e_file:
                    print(f"[Drive private][WARN] Skip file '{name}' ({file_id}): {e_file}")
                    try:
                        if os.path.exists(local_path):
                            os.remove(local_path)
                    except Exception:
                        pass
                    continue
            page_token = resp.get("nextPageToken", None)
            if page_token is None:
                break
    except Exception as e:
        print(f"[Drive private][WARN] Listing/downloading failed: {e}")
        return []

    return sorted(list(set(pdf_paths)))


# ---------------------------
# PDF parsing
# ---------------------------
def extract_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception as e_page:
                print(f"[PDF][WARN] Failed page in '{path}': {e_page}")
                text = ""
            parts.append(text)
        text = "\n".join(parts).strip()
        if not text:
            print(f"[PDF][WARN] Empty text: {path}")
        return text
    except Exception as e:
        print(f"[PDF][WARN] Failed to read '{path}': {e}")
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
    if not text:
        return docs
    i = 0
    step = max(1, chunk_chars - overlap)
    while i < len(text):
        chunk = text[i:i+chunk_chars]
        if not chunk.strip():
            break
        doc_id = hashlib.md5((source + str(i)).encode()).hexdigest()[:16]
        docs.append(Doc(id=doc_id, text=chunk, source=os.path.basename(source), section=section))
        i += step
    return docs


# ---------------------------
# JSON ingestion (supervisors + council)
# ---------------------------
def _safe_read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[JSON][WARN] Failed to read {path}: {e}")
        return None


def _docs_from_supervisors_json(data: dict) -> List[Doc]:
    """
    Поддерживает форматы:
    - vkr_json_export_all.json:
      { "nauchnye_rukovoditeli_2025_26": { "Sheet1": [ {...}, ... ] }, ... }
    - Nauchnye_rukovoditeli_2025_26__all_sheets.json:
      { "Sheet1": [ {...}, ... ] }
    """
    docs: List[Doc] = []
    sheet = None
    if isinstance(data, dict):
        if "nauchnye_rukovoditeli_2025_26" in data and isinstance(data["nauchnye_rukovoditeli_2025_26"], dict):
            inner = data["nauchnye_rukovoditeli_2025_26"]
            sheet = inner.get("Sheet1")
        elif "Sheet1" in data:
            sheet = data.get("Sheet1")

    if not sheet or not isinstance(sheet, list):
        return docs

    for row in sheet:
        try:
            fio = (row.get("ФИО") or "").strip()
            if not fio:
                continue
            interests = (row.get("Мои интересы:") or "").strip()
            format_vkr = (row.get("Формат ВКР: проект/исследование") or "").strip()
            helpful = (row.get("Чем я могу быть полезен как научный руководитель:") or "").strip()
            ideas = (row.get("Готовые идеи или темы, над которыми я хочу работать вместе с вами:") or "").strip()
            about = (row.get("Хочу рассказать о себе ещё кое-что важное:") or "").strip()

            text = (
                f"Научный руководитель: {fio}\n"
                f"Интересы: {interests}\n"
                f"Формат ВКР: {format_vkr}\n"
                f"Чем полезен: {helpful}\n"
                f"Темы/идеи: {ideas}\n"
                f"Пояснение: {about}\n"
            )
            doc_id = hashlib.md5((fio + interests + format_vkr).encode()).hexdigest()[:16]
            docs.append(Doc(id=doc_id, text=text, source="supervisors_json", section="Faculty"))
        except Exception as e:
            print(f"[JSON][WARN] Skip supervisor row: {e}")
            continue
    return docs


def _docs_from_nps_json(data: dict) -> List[Doc]:
    """
    Научно-производственный совет:
    - vkr_json_export_all.json: { "nauchno_proizvodstvenny_sovet": { "члены научно-производственного ": [ {...}, ... ] } }
    - nauchno-proizvodstvennyi_sovet__all_sheets.json: { "члены научно-производственного ": [ {...}, ... ] }
    """
    docs: List[Doc] = []
    entries = None
    if isinstance(data, dict):
        if "nauchno_proizvodstvenny_sovet" in data and isinstance(data["nauchno_proizvodstvenny_sovet"], dict):
            block = data["nauchno_proizvodstvenny_sovet"]
            entries = block.get("члены научно-производственного ") or block.get("члены научно-производственного")
        elif "члены научно-производственного " in data:
            entries = data["члены научно-производственного "]
        elif "члены научно-производственного" in data:
            entries = data["члены научно-производственного"]

    if not entries or not isinstance(entries, list):
        return docs

    for row in entries:
        try:
            name = (row.get("Фамилия Имя") or "").strip()
            if not name:
                continue
            tema = (row.get("Тема") or "").strip()
            inst = (row.get("Институция") or "").strip()
            fac = (row.get("Факультет/кафедра/лаборатория") or "").strip()
            tg = (row.get("Телеграм") or "").strip()
            email = (row.get("Email для коммуникаций") or row.get("Email для доступов") or "").strip()
            city = (row.get("Город") or "").strip()
            group = (row.get("Группа") or "").strip()
            titr = (row.get("Титр") or "").strip()

            text = (
                f"Эксперт/руководитель: {name}\n"
                f"Статус/титр: {titr}\n"
                f"Институция: {inst}\n"
                f"Подразделение: {fac}\n"
                f"Темы/области: {tema}\n"
                f"Контакты: Telegram={tg} Email={email}\n"
                f"Город/группа: {city} / {group}\n"
            )
            doc_id = hashlib.md5((name + inst + tema).encode()).hexdigest()[:16]
            docs.append(Doc(id=doc_id, text=text, source="nps_json", section="Faculty"))
        except Exception as e:
            print(f"[JSON][WARN] Skip NPS row: {e}")
            continue

    return docs


def _collect_json_docs() -> List[Doc]:
    """
    Ищем JSON в data_json/: vkr_json_export_all.json и любые *_all_sheets.json
    """
    docs: List[Doc] = []
    json_dir = "data_json"
    if not os.path.isdir(json_dir):
        return docs

    # Слитый файл, если есть
    merged_path = os.path.join(json_dir, "vkr_json_export_all.json")
    if os.path.exists(merged_path):
        data = _safe_read_json(merged_path)
        if data:
            docs += _docs_from_supervisors_json(data)
            docs += _docs_from_nps_json(data)

    # Отдельные файлы-экспорты
    for path in glob.glob(os.path.join(json_dir, "*__all_sheets.json")):
        data = _safe_read_json(path)
        if not data:
            continue
        before = len(docs)
        docs += _docs_from_supervisors_json(data)
        docs += _docs_from_nps_json(data)
        if len(docs) == before:
            print(f"[JSON][INFO] Unknown schema in {os.path.basename(path)} — skipped")

    if docs:
        print(f"[JSON][OK] Collected {len(docs)} docs from JSON tables.")
    else:
        print("[JSON][INFO] No JSON docs found or parsed.")
    return docs


# ---------------------------
# Build/download pipeline
# ---------------------------
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
    # фильтруем реально существующие
    pdf_paths = [p for p in sorted(set(pdf_paths)) if os.path.exists(p)]
    if not pdf_paths:
        print("[Drive][WARN] No PDFs found. Proceeding with JSON only (if any).")
    return pdf_paths


def build_index_from_pdfs(pdfs: List[str]) -> FaissStore:
    docs: List[Doc] = []

    # PDF -> chunks
    for pdf in pdfs:
        txt = extract_pdf_text(pdf)
        if not txt:
            print(f"[RAG][INFO] Skip empty/extraction-failed PDF: {pdf}")
            continue
        section = infer_section_from_filename(os.path.basename(pdf))
        docs += chunk_text(txt, source=pdf, section=section, chunk_chars=C.CHUNK_CHARS, overlap=C.CHUNK_OVERLAP)

    # JSON -> faculty docs
    docs += _collect_json_docs()

    if not docs:
        placeholder = "Корпус пуст или недоступен. Добавьте PDF/JSON и перезапустите."
        docs = [Doc(id="placeholder", text=placeholder, source="placeholder", section="Notice")]

    texts = [d.text for d in docs]
    embs = normalize(embed_texts(texts))
    if embs.size == 0:
        # fallback на размерность text-embedding-3-large
        embs = np.zeros((1, 3072), dtype="float32")
        docs = [Doc(id="placeholder2", text="(Нет данных для индекса)", source="placeholder", section="Notice")]

    store = FaissStore(d=embs.shape[1])
    metas = [{"id": d.id, "text": d.text, "source": d.source, "section": d.section} for d in docs]
    store.add(embs, metas)
    store.save()
    print(f"[RAG][OK] Index built. Docs: {len(metas)}")
    return store


def load_or_rebuild_index() -> FaissStore:
    # Локальная подпись PDF-кэша
    current: Dict[str, Dict[str, Any]] = {}
    for root, _, files in os.walk(C.DATA_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                try:
                    stat = os.stat(path)
                    current[path] = {"size": stat.st_size, "mtime": int(stat.st_mtime)}
                except Exception:
                    continue

    prev: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(C.PDF_CACHE):
        try:
            with open(C.PDF_CACHE, "r", encoding="utf-8") as fp:
                prev = json.load(fp)
        except Exception:
            prev = {}
    unchanged = (current == prev)

    if os.path.exists(C.INDEX_PATH) and os.path.exists(C.META_PATH) and unchanged:
        store = FaissStore.load()
        if store:
            print("[RAG] Using cached FAISS index.")
            return store

    pdfs = download_pdfs_from_drive()
    store = build_index_from_pdfs(pdfs)
    try:
        with open(C.PDF_CACHE, "w", encoding="utf-8") as fp:
            json.dump(current, fp, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[RAG][WARN] Failed to write PDF cache: {e}")
    return store


def retrieve(store: FaissStore, query: str, k: int = None) -> List[Dict[str, Any]]:
    oai = _openai_client()
    try:
        emb = oai.embeddings.create(model=C.EMBED_MODEL, input=[query])
        q = normalize(np.array([emb.data[0].embedding], dtype="float32"))
    except Exception as e:
        print(f"[RAG][WARN] Embedding failed: {e}")
        return []
    k = k or C.RETRIEVAL_K
    results = store.search(q, k=k)
    ctxs: List[Dict[str, Any]] = []
    for score, meta in results:
        m = dict(meta)
        m["score"] = round(score, 4)
        ctxs.append(m)
    return ctxs
