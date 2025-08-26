"""
Complete RAG-based Drug Information Chatbot
Integrates all agents: Ingestion, Retrieval, Reasoning, and Answer Generation
Uses Gemini API instead of OpenAI, with confidence-based filtering
Improved PDF extraction: tables, figures, OCR, and heading detection
"""
import streamlit as st
from crewai import Crew, Agent, Task
import os
import re
import hashlib
import json
import sqlite3
import time
import redis
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from functools import wraps
import io
# PDF processing imports
import fitz  # PyMuPDF
from pdf2image import convert_from_path  # noqa: F401 (import kept for parity/optional use)
import pytesseract
import camelot
import pandas as pd
from PIL import Image
# ML/AI imports
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from personalized_phrases import PERSONALIZED_PHRASES

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================
# CONFIGURATION
# =====================================
DATA_DIR = Path(".")
PDF_DIR = DATA_DIR / "pdfs"
ASSETS_DIR = DATA_DIR / "pdf_assets"
LOG_DB = DATA_DIR / "ingest_logs.db"
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_store")

# Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.6
USE_CAMELOT = True

# Configure Gemini with multiple API keys
GOOGLE_API_KEYS = []
for i in range(1, 10):  # Support up to 9 API keys (GOOGLE_API_KEY_1 to GOOGLE_API_KEY_9)
    key = os.getenv(f"GOOGLE_API_KEY_{i}")
    if key:
        GOOGLE_API_KEYS.append(key)

# Also check for the main key
main_key = os.getenv("GOOGLE_API_KEY")
if main_key and main_key not in GOOGLE_API_KEYS:
    GOOGLE_API_KEYS.insert(0, main_key)

if not GOOGLE_API_KEYS:
    raise ValueError("At least one GOOGLE_API_KEY environment variable is required (GOOGLE_API_KEY, GOOGLE_API_KEY_1, etc.)")

# Initialize with first key
current_key_index = 0
genai.configure(api_key=GOOGLE_API_KEYS[current_key_index])

def rotate_api_key():
    """Rotate to the next available API key"""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(GOOGLE_API_KEYS)
    genai.configure(api_key=GOOGLE_API_KEYS[current_key_index])
    logger.info(f"Rotated to API key #{current_key_index + 1}")
    return GOOGLE_API_KEYS[current_key_index]

def safe_gemini_call(model, prompt, max_retries=None):
    """Make a Gemini API call with automatic key rotation on quota exceeded"""
    if max_retries is None:
        max_retries = len(GOOGLE_API_KEYS)
    
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            error_msg = str(e).lower()
            if any(quota_error in error_msg for quota_error in 
                   ['quota', 'rate limit', 'resource_exhausted', 'too many requests']):
                logger.warning(f"Quota exceeded on API key #{current_key_index + 1}, rotating...")
                if attempt < max_retries - 1:  # Don't rotate on last attempt
                    rotate_api_key()
                    # Recreate model with new key
                    model = genai.GenerativeModel(GEMINI_MODEL)
                    continue
            # If not a quota error or last attempt, re-raise
            raise e
    
    raise Exception("All API keys exhausted")

# =====================================
# UTILITY FUNCTIONS
# =====================================
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def retry(max_tries=3, backoff=1.0, exceptions=(Exception,)):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = backoff
            for attempt in range(1, max_tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_tries:
                        raise
                    logger.warning(f"{func.__name__} failed (attempt {attempt}/{max_tries}): {e}")
                    time.sleep(delay)
                    delay *= 2
        return wrapper
    return deco

# =====================================
# AGENT 1: DATA INGESTION
# =====================================
class DataIngestionAgent:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client, self.collection = self._setup_chroma()
        self._init_log_db()
        self.dedupe_hashes = self._load_dedupe_hashes()
        ensure_dir(ASSETS_DIR)

    def _setup_chroma(self):
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection(
            name="drug_pdfs",
            metadata={"hnsw:space": "cosine"}
        )
        return client, collection

    def _get_db_connection(self):
        """Get a new database connection for the current thread"""
        return sqlite3.connect(LOG_DB, check_same_thread=False)

    def _init_log_db(self):
        conn = self._get_db_connection()
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ingests (
            id TEXT PRIMARY KEY,
            file TEXT,
            file_hash TEXT,
            pages INTEGER,
            chunks INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            meta JSON
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dedupe_hashes (
            hash TEXT PRIMARY KEY,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def _load_dedupe_hashes(self) -> set:
        conn = self._get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT hash FROM dedupe_hashes")
        result = {row[0] for row in cur.fetchall()}
        conn.close()
        return result

    def _fix_hyphenation_and_linebreaks(self, text: str) -> str:
        text = re.sub(r"(\w+)-\n(\w+)", lambda m: m.group(1) + m.group(2), text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _detect_headings_with_fonts(self, page: fitz.Page) -> List[Tuple[int, str]]:
        """Return list of (char_index, heading_text) using font size heuristics."""
        try:
            d = page.get_text("dict")
            blocks = d.get("blocks", [])
            candidates = []
            char_cursor = 0
            for b in blocks:
                for line in b.get("lines", []):
                    line_text = "".join([s.get("text", "") for s in line.get("spans", [])])
                    max_size = 0
                    for s in line.get("spans", []):
                        max_size = max(max_size, s.get("size", 0))
                    stripped = line_text.strip()
                    if not stripped:
                        char_cursor += len(line_text) + 1
                        continue
                    if max_size >= 11.5 and (stripped.upper() == stripped or max_size >= 13):
                        candidates.append((char_cursor, stripped))
                    char_cursor += len(line_text) + 1
            if not candidates:
                return self._detect_headings_from_page_text(page.get_text("text"))
            return candidates
        except Exception:
            return self._detect_headings_from_page_text(page.get_text("text"))

    def _detect_headings_from_page_text(self, page_text: str) -> List[Tuple[int, str]]:
        headings = []
        lines = page_text.splitlines()
        idx = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                idx += len(line) + 1
                continue
            if (
                re.match(r'^\d+(\.\d+)*\s+[A-Z][A-Z0-9 \-\,\(\)\/]+$', stripped)
                or (
                    len(stripped) >= 10
                    and stripped.upper() == stripped
                    and sum(c.isalpha() for c in stripped) > 4
                )
            ):
                headings.append((idx, stripped))
            idx += len(line) + 1
        return headings

    def _detect_headings(self, page: fitz.Page) -> List[Tuple[int, str]]:
        """Detect headings using font size heuristics, fall back to text patterns."""
        return self._detect_headings_with_fonts(page)

    def _assign_section(self, page_text: str, chunk_start_idx: int, headings: List[Tuple[int, str]]) -> str:
        section = "Unknown"
        for idx, h in headings:
            if idx <= chunk_start_idx:
                section = h
            else:
                break
        return section

    def _extract_pdf_assets(self, pdf_path: Path, temp_image_dir: Path) -> Dict[str, Any]:
        ensure_dir(temp_image_dir)
        doc = fitz.open(str(pdf_path))
        page_texts, tables, figures = [], [], []
        if USE_CAMELOT:
            try:
                camelot_tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
                logger.info(f"[Camelot] Found {len(camelot_tables)} table(s)")
            except Exception as e:
                logger.warning(f"[Camelot] Error: {e}")
                camelot_tables = []
        else:
            camelot_tables = []
        for pno in range(len(doc)):
            page = doc[pno]
            page_num = pno + 1
            raw_text = page.get_text("text")
            text = raw_text.strip()
            if not text or len(text) < 30:
                try:
                    pix = page.get_pixmap(dpi=200)
                    mode = "RGBA" if pix.alpha else "RGB"
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    text = ocr_text
                    img_path = temp_image_dir / f"{pdf_path.stem}_page_{page_num}.png"
                    img.save(img_path)
                    figures.append({
                        "page": page_num,
                        "figure_id": f"{pdf_path.stem}_page_{page_num}",
                        "image_path": str(img_path),
                        "ocr_text": ocr_text.strip()[:5000],
                    })
                except Exception as e:
                    logger.warning(f"OCR fallback failed: {e}")
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                    img_path = temp_image_dir / f"{pdf_path.stem}_p{page_num}_img{img_index}.{ext}"
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(img_obj)
                    figures.append({
                        "page": page_num,
                        "figure_id": f"{pdf_path.stem}_p{page_num}_img{img_index}",
                        "image_path": str(img_path),
                        "ocr_text": ocr_text.strip()[:5000],
                    })
                except Exception:
                    logger.debug(f"Embedded image processing failed on {pdf_path.name} page {page_num} image {img_index}")
                    continue
            page_texts.append({"page": page_num, "text": text})
        for ti, t in enumerate(camelot_tables):
            try:
                df = t.df
                pg = int(str(t.page).split(",")[0])
                table_id = f"{pdf_path.stem}_table_{ti+1}"
                csv_path = temp_image_dir / f"{table_id}.csv"
                html_path = temp_image_dir / f"{table_id}.html"
                df.to_csv(csv_path, index=False)
                df.to_html(html_path, index=False)
                tables.append({
                    "page": pg,
                    "table_id": table_id,
                    "csv_path": str(csv_path),
                    "html_path": str(html_path),
                    "nrows": len(df),
                    "ncols": len(df.columns),
                    "preview": df.head(3).to_dict(orient="records"),
                })
            except Exception as e:
                logger.warning(f"Camelot processing error: {e}")
        return {"page_texts": page_texts, "tables": tables, "figures": figures}

    def _sanitize_metadata(self, metadata_list):
        clean_list = []
        for meta in metadata_list:
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    if k in ("page", "chunk_start"):
                        clean_meta[k] = -1
                    elif k.endswith("_id"):
                        clean_meta[k] = ""
                    else:
                        clean_meta[k] = ""
                else:
                    clean_meta[k] = v
            clean_list.append(clean_meta)
        return clean_list

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[\.?\!])\s+", text)
        chunks = []
        cur_sentences = []
        cur_len = 0
        cur_start = 0
        char_cursor = 0
        def emit_current():
            nonlocal cur_sentences, cur_len, cur_start
            if not cur_sentences:
                return
            chunk_text = " ".join(s.strip() for s in cur_sentences).strip()
            chunks.append({"chunk": chunk_text, "start_char": cur_start})
            if CHUNK_OVERLAP > 0:
                tail = []
                acc = 0
                for s in reversed(cur_sentences):
                    tail.insert(0, s)
                    acc += len(s) + 1
                    if acc >= CHUNK_OVERLAP:
                        break
                cur_sentences = tail
                cur_len = sum(len(s) + 1 for s in cur_sentences)
                cur_start = max(0, char_cursor - cur_len)
            else:
                cur_sentences = []
                cur_len = 0
        for sent in sentences:
            if not sent.strip():
                char_cursor += len(sent) + 1
                continue
            sent_len = len(sent) + 1
            if cur_len + sent_len <= CHUNK_MAX_CHARS or not cur_sentences:
                if not cur_sentences:
                    cur_start = char_cursor
                cur_sentences.append(sent)
                cur_len += sent_len
            else:
                emit_current()
                if cur_len + sent_len <= CHUNK_MAX_CHARS:
                    cur_sentences.append(sent)
                    cur_len += sent_len
            char_cursor += sent_len
        if cur_sentences:
            emit_current()
        return chunks

    def ingest_pdf(self, pdf_path: Path, drug_name: str):
        logger.info(f"Processing {pdf_path} as {drug_name}")
        assets_dir = ASSETS_DIR / pdf_path.stem
        fh = file_sha1(pdf_path)
        conn = self._get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT file_hash FROM ingests WHERE file = ?", (str(pdf_path.name),))
        row = cur.fetchone()
        if row and row[0] == fh:
            logger.info(f"File {pdf_path.name} already ingested — skipping")
            conn.close()
            return
        extracted = self._extract_pdf_assets(pdf_path, assets_dir)
        all_chunks_to_embed, all_metadatas, ids = [], [], []
        total_chunks = 0
        with fitz.open(str(pdf_path)) as doc:
            for p in extracted["page_texts"]:
                page_num = p["page"]
                page_text = self._fix_hyphenation_and_linebreaks(p["text"])
                page = doc[page_num - 1]
                headings = self._detect_headings(page)
                page_chunks = self._chunk_text(page_text)
                for ci, cobj in enumerate(page_chunks):
                    chunk_text_val = cobj["chunk"]
                    start_char = cobj["start_char"]
                    section = self._assign_section(page_text, start_char, headings)
                    h = sha1(chunk_text_val)
                    if h in self.dedupe_hashes:
                        continue
                    self.dedupe_hashes.add(h)
                    try:
                        cur.execute("INSERT OR IGNORE INTO dedupe_hashes (hash) VALUES (?)", (h,))
                        conn.commit()
                    except Exception:
                        logger.debug("Could not persist dedupe hash")
                    doc_id = f"{pdf_path.stem}~{page_num}~{ci}~{uuid4().hex[:8]}"
                    metadata = {
                        "drug": drug_name,
                        "source_file": str(pdf_path.name),
                        "page": page_num,
                        "section": section,
                        "chunk_start": start_char,
                        "chunk_id": doc_id,
                        "type": "text_chunk",
                    }
                    all_chunks_to_embed.append(chunk_text_val)
                    all_metadatas.append(metadata)
                    ids.append(doc_id)
                    total_chunks += 1
            for table in extracted["tables"]:
                try:
                    df = (pd.read_csv(table["csv_path"]) if Path(table["csv_path"]).exists() else None)
                    if df is not None:
                        df = df.dropna(axis=1, how="all")
                        df = df.astype(str).replace({r"\s+": " "}, regex=True)
                        table_text = df.head(10).to_csv(index=False)
                        cols = ", ".join(map(str, df.columns))
                    else:
                        table_text = table.get("preview", str(table))
                        cols = "unknown"
                    table_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", table_text)
                    table_text = re.sub(r"\s+", " ", table_text).strip()
                except Exception:
                    table_text, cols = (f"[table {table['table_id']} snapshot unavailable]", "unknown")
                text_for_embed = (f"TABLE ({table['table_id']}) on page {table['page']}. Columns: {cols}. Preview:\n{table_text}")
                h = sha1(text_for_embed)
                if h not in self.dedupe_hashes:
                    self.dedupe_hashes.add(h)
                    doc_id = f"{pdf_path.stem}~table~{table['table_id']}"
                    metadata = {
                        "drug": drug_name,
                        "source_file": str(pdf_path.name),
                        "page": table["page"],
                        "section": "Table",
                        "table_id": table["table_id"],
                        "csv_path": table["csv_path"],
                        "html_path": table["html_path"],
                        "chunk_id": doc_id,
                        "type": "table",
                    }
                    all_chunks_to_embed.append(text_for_embed)
                    all_metadatas.append(metadata)
                    ids.append(doc_id)
                    total_chunks += 1
            for fig in extracted["figures"]:
                txt = fig.get("ocr_text", "")
                text_for_embed = (f"FIGURE ({fig['figure_id']}) on page {fig['page']}. OCR_text_preview: {txt[:1000]}")
                h = sha1(text_for_embed)
                if h not in self.dedupe_hashes:
                    self.dedupe_hashes.add(h)
                    doc_id = f"{pdf_path.stem}~fig~{fig['figure_id']}"
                    metadata = {
                        "drug": drug_name,
                        "source_file": str(pdf_path.name),
                        "page": fig["page"],
                        "section": "Figure",
                        "figure_id": fig["figure_id"],
                        "image_path": fig.get("image_path"),
                        "chunk_id": doc_id,
                        "type": "figure",
                    }
                    all_chunks_to_embed.append(text_for_embed)
                    all_metadatas.append(metadata)
                    ids.append(doc_id)
                    total_chunks += 1
        for i, m in enumerate(all_metadatas):
            prev_id = all_metadatas[i - 1]["chunk_id"] if i > 0 else None
            next_id = (all_metadatas[i + 1]["chunk_id"] if i < len(all_metadatas) - 1 else None)
            m["prev_chunk_id"] = prev_id
            m["next_chunk_id"] = next_id
        for i in range(0, len(all_chunks_to_embed), BATCH_SIZE):
            batch_texts = all_chunks_to_embed[i : i + BATCH_SIZE]
            batch_ids = ids[i : i + BATCH_SIZE]
            batch_metas = all_metadatas[i : i + BATCH_SIZE]
            embeddings = self.embedder.encode(batch_texts, show_progress_bar=False)
            embeddings_list = [emb.tolist() for emb in embeddings]
            batch_metadatas = self._sanitize_metadata(batch_metas)
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=embeddings_list,
            )
        ingest_id = sha1(str(pdf_path.resolve()) + fh)
        cur.execute(
            """
            INSERT OR REPLACE INTO ingests (id, file, file_hash, pages, chunks, meta)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ingest_id,
                str(pdf_path.name),
                fh,
                len(extracted["page_texts"]),
                total_chunks,
                json.dumps({"drug": drug_name}),
            ),
        )
        conn.commit()
        conn.close()
        logger.info(f"Stored {total_chunks} docs from {pdf_path.name}")

    def vector_search(self, query: str, top_k: int = 8, filter_metadata: Optional[dict] = None) -> List[Dict]:
        """Perform vector similarity search"""
        query_embedding = self.embedder.encode([query])[0].tolist()
        where = filter_metadata if filter_metadata else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        retrievals = []
        for doc, meta, dist in zip(docs, metas, distances):
            similarity = max(0.0, 1.0 - dist) if dist is not None else 0.0
            citation = f"{meta.get('source_file', '')} (Page: {meta.get('page', '')}, Section: {meta.get('section', '')}, Type: {meta.get('type', '')})"
            retrievals.append({
                "text": doc,
                "metadata": meta,
                "citation": citation,
                "similarity": similarity,
                "distance": dist
            })
        return retrievals

# =====================================
# AGENT 2: RETRIEVAL QUERY ROUTING
# =====================================
class RetrievalQueryRoutingAgent:
    def __init__(self, collection, embedder, session_agent=None):
        self.collection = collection
        self.embedder = embedder
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.session_agent = session_agent

    def extract_entities(self, query: str, session_id: str = None) -> Dict[str, List[str]]:
        """Extract drug names and intent from query using Gemini"""
        history_context = ""
        if session_id and self.session_agent:
            try:
                history = self.session_agent.get_history(session_id)
                recent_history = history[-3:] if len(history) > 3 else history
                history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history if msg['role'] != 'system'])
                if history_context:
                    history_context = f"\nConversation context:\n{history_context}\n"
            except Exception as e:
                logger.warning(f"Failed to fetch history: {e}")
                history_context = ""
        prompt = f"""
        Analyze this medical query and extract:
        1. Drug names mentioned (Humira, Rinvoq, Skyrizi, etc.)
        2. Intent/topic (dosage, side effects, interactions, contraindications, etc.)
        {history_context}
        Current Query: "{query}"
        Respond ONLY in JSON format, no explanation, no markdown, no code block:
        {{
            "drugs": ["drug1", "drug2"],
            "intent": "primary_topic"
        }}
        """
        try:
            response = safe_gemini_call(self.model, prompt)
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            if not response_text.startswith("{"):
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            result = json.loads(response_text)
            if "drugs" not in result:
                result["drugs"] = []
            if "intent" not in result:
                result["intent"] = "general"
            return result
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e} | Using fallback")
            return {"drugs": [], "intent": "general"}

    def vector_search(self, query: str, top_k: int = 8, filter_metadata: Optional[dict] = None) -> List[Dict]:
        """Perform vector similarity search"""
        query_embedding = self.embedder.encode([query])[0].tolist()
        where = filter_metadata if filter_metadata else None
        logger.info("Where Clause:", where)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        retrievals = []
        for doc, meta, dist in zip(docs, metas, distances):
            similarity = max(0.0, 1.0 - dist) if dist is not None else 0.0
            citation = f"{meta.get('source_file', '')} (Page: {meta.get('page', '')}, Section: {meta.get('section', '')}, Type: {meta.get('type', '')})"
            retrievals.append({
                "text": doc,
                "metadata": meta,
                "citation": citation,
                "similarity": similarity,
                "distance": dist
            })
        return retrievals

# =====================================
# AGENT 3: REASONING AND DOMAIN AGENT
# =====================================
class ReasoningDomainAgent:
    def __init__(self):
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def assess_chunk_relevance(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Assess relevance of chunks to query and filter by confidence threshold"""
        filtered_chunks = []
        for chunk in chunks:
            relevance_score = self._calculate_relevance(query, chunk["text"])
            chunk["relevance_score"] = relevance_score
            if relevance_score >= CONFIDENCE_THRESHOLD:
                filtered_chunks.append(chunk)
        filtered_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return filtered_chunks

    def _calculate_relevance(self, query: str, chunk_text: str) -> float:
        """Calculate relevance score between query and chunk"""
        prompt = f"""
        Rate the relevance of this text chunk to the user query on a scale of 0.0 to 1.0.
        Query: "{query}"
        Text Chunk: "{chunk_text[:1000]}..."
        Consider:
        - Direct topical relevance
        - Presence of specific drug names mentioned in query
        - Relevance to medical context
        Respond with only a decimal number between 0.0 and 1.0:
        """
        try:
            response = safe_gemini_call(self.model, prompt)
            score_text = response.text.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            return self._simple_relevance_score(query, chunk_text)

    def _simple_relevance_score(self, query: str, chunk_text: str) -> float:
        """Simple fallback relevance scoring"""
        query_words = set(query.lower().split())
        chunk_words = set(chunk_text.lower().split())
        intersection = query_words & chunk_words
        union = query_words | chunk_words
        if not union:
            return 0.0
        jaccard_score = len(intersection) / len(union)
        return min(1.0, jaccard_score * 2)

    def check_relationships_exist(self, retrievals: List[Dict], query: str) -> bool:
        """Check if retrievals contain sufficient information to answer query"""
        if not retrievals:
            return False
        avg_relevance = sum(r.get("relevance_score", 0) for r in retrievals) / len(retrievals)
        return avg_relevance >= CONFIDENCE_THRESHOLD

# =====================================
# AGENT 4: ANSWER GENERATION
# =====================================
class AnswerGenerationAgent:
    def __init__(self, session_agent=None):
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.session_agent = session_agent

    def generate_final_response(self, query: str, session_id: str, retrievals: List[Dict], user_context: str = "") -> Dict[str, Any]:
        """Generate final structured response using Gemini"""
        if not retrievals:
            return {
                "short_answer": "I don't have sufficient information to answer your query.",
                "confidence_score": 0.0,
                "citations": [],
                "reasoning": "No relevant information found in the knowledge base.",
            }
        history = []
        if session_id and self.session_agent:
            try:
                history = self.session_agent.get_history(session_id)
            except Exception as e:
                logger.warning(f"Failed to fetch history for answer generation: {e}")
        context_parts = []
        citations = []
        for i, retrieval in enumerate(retrievals):
            context_parts.append(f"Source {i+1}: {retrieval['text']}")
            citations.append({
                "source": retrieval["metadata"].get("source_file", ""),
                "page_reference": str(retrieval["metadata"].get("page", "")),
                "section_id": retrieval["metadata"].get("section", ""),
                "relevance_score": retrieval.get("relevance_score", 0.0)
            })
        context = "\n\n".join(context_parts)
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
        user_context_text = f"\nUser Context: {user_context}\n" if user_context else ""
        prompt = f"""
        You are a medical information assistant. Provide a structured response based ONLY on the provided context.
        Conversation History:
        {history_text}
        {user_context_text}
        Retrieved Context:
        {context}
        User Query: {query}
        Your response must be in JSON format with these fields:
        - "short_answer": Direct, concise answer (string)
        - "confidence_score": Your confidence in the answer from 0.0 to 1.0 (float)
        - "reasoning": Brief explanation of your reasoning (string)
        Guidelines:
        - if user_context is provided, incorporate it into your answer based on context
        - Use ONLY information from the provided context
        - Be concise and medically accurate
        - If information is insufficient, state so clearly
        - Include specific details like dosages, side effects when available
        - Maintain professional medical tone
        JSON Response:
        """
        try:
            response = safe_gemini_call(self.model, prompt)
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            final_response = json.loads(response_text)
            final_response["citations"] = citations
            return final_response
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "short_answer": "I encountered an error while processing your query. Please try rephrasing your question.",
                "confidence_score": 0.0,
                "citations": citations,
                "reasoning": f"Error in answer generation: {str(e)}"
            }

# =====================================
# AGENT 5: SESSION MANAGEMENT
# =====================================
class SessionManagementAgent:
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
        except:
            logger.warning("Redis not available, using session state fallback")
            self.use_redis = False

    def get_history(self, session_id: str) -> List[Dict]:
        if self.use_redis:
            try:
                history_json = self.redis_client.get(session_id)
                return json.loads(history_json) if history_json else []
            except:
                return []
        else:
            return st.session_state.get(f"history_{session_id}", [])

    def update_history(self, session_id: str, role: str, content: str):
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        if self.use_redis:
            try:
                self.redis_client.set(session_id, json.dumps(history))
            except:
                pass
        else:
            st.session_state[f"history_{session_id}"] = history

# =====================================
# MAIN ORCHESTRATOR
# =====================================
class DrugChatbotOrchestrator:
    def __init__(self):
        st.cache_resource.clear()
        with st.spinner("Initializing chatbot components..."):
            self.ingestion_agent = DataIngestionAgent()
            self.session_agent = SessionManagementAgent()
            self.retrieval_agent = RetrievalQueryRoutingAgent(
                self.ingestion_agent.collection,
                self.ingestion_agent.embedder,
                self.session_agent
            )
            self.reasoning_agent = ReasoningDomainAgent()
            self.answer_agent = AnswerGenerationAgent(self.session_agent)
            self.crew_agents = {
                "ingestion": Agent(
                    role="Ingestion Agent",
                    goal="Extract and chunk data from PDFs, including images and tables.",
                    backstory="Handles all PDF ingestion and asset extraction."
                ),
                "retrieval": Agent(
                    role="Retrieval Agent",
                    goal="Perform vector search and entity extraction for queries.",
                    backstory="Routes queries and retrieves relevant chunks."
                ),
                "reasoning": Agent(
                    role="Reasoning Agent",
                    goal="Assess relevance and filter retrieved chunks.",
                    backstory="Scores and filters chunks for answer generation."
                ),
                "answer": Agent(
                    role="Answer Agent",
                    goal="Generate final structured medical answers.",
                    backstory="Uses Gemini to generate answers from context."
                ),
                "session": Agent(
                    role="Session Agent",
                    goal="Manage user session and chat history.",
                    backstory="Handles session state and history."
                )
            }
            self.crew = Crew(
                agents=list(self.crew_agents.values()),
                tasks=[]
            )

    def ingest_pdfs(self):
        """Ingest all PDF files in the pdfs directory."""
        pdf_files = list(PDF_DIR.glob("*.pdf"))  # Dynamically fetch all PDFs in the folder
        if not pdf_files:
            st.warning(f"⚠️ No PDF files found in {PDF_DIR}")
            return

        for pdf_path in pdf_files:
            drug_name = pdf_path.stem  # Use the file name (without extension) as the drug name
            try:
                task = Task(
                    description=f"Ingest PDF {pdf_path.name} for drug {drug_name}",
                    agent=self.crew_agents["ingestion"],
                    expected_output="PDF ingested and assets extracted."
                )
                self.crew.tasks.append(task)
                logger.info(f"Calling agent: {self.crew_agents['ingestion'].role}")
                self.ingestion_agent.ingest_pdf(pdf_path, drug_name)
                st.success(f"✅ Ingested {pdf_path.name}")
            except Exception as e:
                st.error(f"❌ Failed to ingest {pdf_path.name}: {e}")

    def process_query(self, query: str, session_id: str) -> Dict[str, Any]:
        user_info = st.session_state.get("user_info", {})
        is_personalized = any(
            phrase in query.lower()
            for phrase in ["can i use", "should i take", "is it safe for me", "based on my"]
        )
        user_context = ""
        enhanced_query = query
        if is_personalized and user_info:
            user_context = (
                f"Age={user_info.get('age', 'N/A')}, "
                f"Weight={user_info.get('weight', 'N/A')}, "
                f"Symptoms={', '.join(user_info.get('symptoms', []))}"
            )
            enhanced_query = f"{query} [User context: {user_context}]"
        
        print(f"Processing query: {query} | Enhanced: {enhanced_query}")
        logger.info(f"Processing query: {query} | Enhanced: {enhanced_query}")
        
        entities_task = Task(
            description="Extract entities from user query.",
            agent=self.crew_agents["retrieval"],
            expected_output="Entities extracted."
        )
        retrieval_task = Task(
            description="Perform vector search for query.",
            agent=self.crew_agents["retrieval"],
            expected_output="Relevant chunks retrieved."
        )
        reasoning_task = Task(
            description="Assess relevance of retrieved chunks.",
            agent=self.crew_agents["reasoning"],
            expected_output="Chunks filtered by relevance."
        )
        answer_task = Task(
            description="Generate final answer from filtered chunks.",
            agent=self.crew_agents["answer"],
            expected_output="Structured answer generated."
        )
        session_task = Task(
            description="Update and retrieve session history.",
            agent=self.crew_agents["session"],
            expected_output="Session history managed."
        )
        self.crew.tasks.extend([
            entities_task, retrieval_task, reasoning_task, answer_task, session_task
        ])
        # Extract entities using original query
        logger.info(f"Calling agent: {self.crew_agents['retrieval'].role}")
        entities = self.retrieval_agent.extract_entities(query, session_id)

        # Construct filter_metadata
        filter_metadata = None
        if entities.get("drugs"):
            # Use proper $in operator for ChromaDB
            filter_metadata = {"drug": {"$in": entities["drugs"]}}
            print("Filter Metadata:", filter_metadata)  # Debug: Print the filter_metadata

        # Perform vector search using enhanced query for personalized questions
        logger.info(f"Calling agent: {self.crew_agents['retrieval'].role}")
        retrievals = self.retrieval_agent.vector_search(
            enhanced_query, top_k=8, filter_metadata=filter_metadata
        )
        
        logger.info(f"Calling agent: {self.crew_agents['reasoning'].role}")
        filtered_retrievals = self.reasoning_agent.assess_chunk_relevance(query, retrievals)
        logger.info(f"Calling agent: {self.crew_agents['reasoning'].role}")
        has_sufficient_info = self.reasoning_agent.check_relationships_exist(filtered_retrievals, query)
        if not has_sufficient_info:
            return {
                "short_answer": "I don't have sufficient relevant information to answer your query based on the available documents.",
                "confidence_score": 0.0,
                "citations": [],
                "reasoning": "Insufficient relevant information in knowledge base.",
                "entities_found": entities
            }
        logger.info(f"Calling agent: {self.crew_agents['answer'].role}")
        response = self.answer_agent.generate_final_response(query, session_id, filtered_retrievals[:4], user_context)
        response["entities_found"] = entities
        return response

# =====================================
# STREAMLIT UI
# =====================================
def main():
    st.set_page_config(page_title="Drug Information Chatbot", page_icon="💊", layout="wide")
    st.title("💊 Drug Information Chatbot")
    st.caption("RAG-powered medical information assistant using Agentic AI")

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}_{uuid4().hex[:8]}"
    if "orchestrator" not in st.session_state:
        try:
            st.session_state.orchestrator = DrugChatbotOrchestrator()
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {e}")
            return
    orchestrator = st.session_state.orchestrator

    # Sidebar for user information form
    with st.sidebar:
        st.header("User Information")
        with st.form("user_info_form"):
            age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
            weight = st.number_input("Weight (kg)", min_value=1, max_value=200, value=70, step=1)
            symptoms = st.text_area("Symptoms (comma-separated)", value="")
            uploaded_file = st.file_uploader("Upload Document (PDF)", type=["pdf"])
            submit_button = st.form_submit_button("Save")
        if submit_button:
            st.session_state.user_info = {
                "age": age,
                "weight": weight,
                "symptoms": [s.strip() for s in symptoms.split(",") if s.strip()],
                "uploaded_file": uploaded_file.name if uploaded_file else None
            }
            if uploaded_file:
                ensure_dir(PDF_DIR)
                file_path = PDF_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved {uploaded_file.name} to {PDF_DIR}")
            st.success("User information saved!")
        st.header("System Controls")
        if st.button("🔄 Ingest PDFs"):
            with st.spinner("Ingesting PDF documents..."):
                orchestrator.ingest_pdfs()
        st.header("Configuration")
        st.info(f"**Model:** {GEMINI_MODEL}")
        st.info(f"**API Keys Available:** {len(GOOGLE_API_KEYS)}")
        st.info(f"**Current Key:** #{current_key_index + 1}")
        st.info(f"**Confidence Threshold:** {CONFIDENCE_THRESHOLD}")
        st.info(f"**Session ID:** {st.session_state.session_id}")

        if st.button("🗑️ Clear History"):
            orchestrator.session_agent.update_history(st.session_state.session_id, "system", "History cleared")
            st.rerun()
        if "user_info" in st.session_state:
            st.subheader("Your Information")
            st.markdown(
                f"- **Age:** {st.session_state.user_info.get('age', 'N/A')}\n"
                f"- **Weight:** {st.session_state.user_info.get('weight', 'N/A')} kg\n"
                f"- **Symptoms:** {', '.join(st.session_state.user_info.get('symptoms', []))}\n"
                f"- **Uploaded File:** {st.session_state.user_info.get('uploaded_file', 'None')}"
            )

    # Main chat interface
    history = orchestrator.session_agent.get_history(st.session_state.session_id)
    for message in history:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if query := st.chat_input("Ask about drug information (e.g., 'What are the side effects of Humira?')"):
        with st.chat_message("user"):
            st.markdown(query)
        orchestrator.session_agent.update_history(st.session_state.session_id, "user", query)
        with st.chat_message("assistant"):
            agent_status = st.empty()
            with st.spinner("Processing your query..."):
                # Check if this is a personalized question and enhance query (define variables first)
                user_info = st.session_state.get("user_info", {})
                is_personalized = any(
                    phrase in query.lower()
                    for phrase in PERSONALIZED_PHRASES
                )
                user_context = ""
                enhanced_query = query
                if is_personalized and user_info:
                    user_context = (
                        f"Age={user_info.get('age', 'N/A')}, "
                        f"Weight={user_info.get('weight', 'N/A')}, "
                        f"Symptoms={', '.join(user_info.get('symptoms', []))}"
                    )
                    enhanced_query = f"{query} [User context: {user_context}]"
                
                # Use the orchestrator's process_query method for proper logging and flow
                try:
                    # Show initial processing status
                    agent_status.info("🤖 **Initializing Multi-Agent System...**")
                    time.sleep(0.5)
                    
                    agent_status.info("🔍 **Retrieval Agent:** Extracting entities from query...")
                    entities = orchestrator.retrieval_agent.extract_entities(query, st.session_state.session_id)
                    time.sleep(0.3)
                    
                    # Construct filter metadata
                    filter_metadata = None
                    if entities.get("drugs"):
                        filter_metadata = {"drug": {"$in": entities["drugs"]}}
                        agent_status.info(f"🎯 **Retrieval Agent:** Found drugs: {', '.join(entities['drugs'])}")
                        time.sleep(0.3)
                    
                    if is_personalized and user_info:
                        agent_status.info("👤 **Retrieval Agent:** Enhancing query with user context...")
                        time.sleep(0.3)
                    
                    agent_status.info("🔍 **Retrieval Agent:** Performing vector similarity search...")
                    retrievals = orchestrator.retrieval_agent.vector_search(
                        enhanced_query, top_k=8, filter_metadata=filter_metadata
                    )
                    time.sleep(0.5)
                    
                    agent_status.info(f"📊 **Reasoning Agent:** Analyzing {len(retrievals)} retrieved chunks...")
                    filtered_retrievals = orchestrator.reasoning_agent.assess_chunk_relevance(query, retrievals)
                    time.sleep(0.5)
                    
                    agent_status.info("🧠 **Reasoning Agent:** Checking information sufficiency...")
                    has_sufficient_info = orchestrator.reasoning_agent.check_relationships_exist(filtered_retrievals, query)
                    time.sleep(0.3)
                    
                    if not has_sufficient_info:
                        agent_status.warning("⚠️ **Reasoning Agent:** Insufficient relevant information found")
                        response = {
                            "short_answer": "I don't have sufficient relevant information to answer your query based on the available documents.",
                            "confidence_score": 0.0,
                            "citations": [],
                            "reasoning": "Insufficient relevant information in knowledge base.",
                            "entities_found": entities
                        }
                    else:
                        agent_status.info(f"✅ **Reasoning Agent:** Found {len(filtered_retrievals)} relevant chunks")
                        time.sleep(0.3)
                        
                        agent_status.info("💬 **Answer Agent:** Generating structured response...")
                        response = orchestrator.answer_agent.generate_final_response(
                            query, st.session_state.session_id, filtered_retrievals[:4], user_context
                        )
                        response["entities_found"] = entities
                        time.sleep(0.5)
                        
                        agent_status.success("🎉 **Answer Agent:** Response generated successfully!")
                        time.sleep(0.3)
                        
                except Exception as e:
                    logger.error(f"Error in process_query: {e}")
                    agent_status.error(f"❌ **System Error:** {str(e)}")
                    time.sleep(0.5)
                    
                    # Fallback to manual processing with status updates
                    agent_status.info("🔄 **System:** Switching to fallback processing...")
                    entities = orchestrator.retrieval_agent.extract_entities(query, st.session_state.session_id)
                    
                    if is_personalized and user_info:
                        agent_status.info("👤 **Fallback:** Applying user context...")
                        print(f"Enhanced query for personalized question: {enhanced_query}")
                    
                    agent_status.info("🔍 **Fallback:** Performing vector search...")
                    filter_metadata = None
                    if entities.get("drugs"):
                        filter_metadata = {"drug": {"$in": entities["drugs"]}}
                    retrievals = orchestrator.retrieval_agent.vector_search(
                        enhanced_query, top_k=8, filter_metadata=filter_metadata
                    )
                    
                    agent_status.info("🧠 **Fallback:** Assessing chunk relevance...")
                    filtered_retrievals = orchestrator.reasoning_agent.assess_chunk_relevance(query, retrievals)
                    has_sufficient_info = orchestrator.reasoning_agent.check_relationships_exist(filtered_retrievals, query)
                    
                    if not has_sufficient_info:
                        agent_status.warning("⚠️ **Fallback:** Insufficient information")
                        response = {
                            "short_answer": "I don't have sufficient relevant information to answer your query based on the available documents.",
                            "confidence_score": 0.0,
                            "citations": [],
                            "reasoning": "Insufficient relevant information in knowledge base.",
                            "entities_found": entities
                        }
                    else:
                        agent_status.info("💬 **Fallback:** Generating answer...")
                        response = orchestrator.answer_agent.generate_final_response(
                            query, st.session_state.session_id, filtered_retrievals[:4]
                        )
                        response["entities_found"] = entities
                        agent_status.success("✅ **Fallback:** Processing completed")
                        
                # Extract data for debugging display
                entities = response.get("entities_found", {})
                
                # Final status update
                confidence = response.get('confidence_score', 0.0)
                if confidence >= 0.8:
                    agent_status.success(f"🎯 **Complete:** High confidence response ({confidence:.2f})")
                elif confidence >= 0.6:
                    agent_status.info(f"✅ **Complete:** Medium confidence response ({confidence:.2f})")
                else:
                    agent_status.warning(f"⚠️ **Complete:** Low confidence response ({confidence:.2f})")
                    
            # Clear the agent status after a brief display
            time.sleep(2)
            agent_status.empty()
            
            # Display debug information
            with st.expander("🔍 **Full Analysis Process**", expanded=False):
                if is_personalized and user_info:
                    st.info(f"🎯 **Personalized Query Detected**")
                    st.markdown(f"**Original Query:** {query}")
                    st.markdown(f"**Enhanced Query:** {enhanced_query}")
                    st.markdown(f"**User Context:** {user_context}")
                    
                st.subheader("1. Entity Extraction (Agent: Retrieval)")
                st.info("Agent called: Retrieval Agent")
                st.json(entities)
                
                # Only show retrievals if we have them (from successful processing)
                if 'retrievals' in locals():
                    st.subheader("2. Vector Search Results (Agent: Retrieval)")
                    st.info("Agent called: Retrieval Agent")
                    for i, retrieval in enumerate(retrievals):
                        st.markdown(f"**Retrieval {i+1}**")
                        st.markdown(f"- **Text:** {retrieval['text'][:200]}...")
                        st.markdown(f"- **Source:** {retrieval['metadata'].get('source_file', 'Unknown')}")
                        st.markdown(f"- **Page:** {retrieval['metadata'].get('page', 'N/A')}")
                        st.markdown(f"- **Section:** {retrieval['metadata'].get('section', 'N/A')}")
                        st.markdown(f"- **Similarity:** {retrieval['similarity']:.2f}")
                        st.markdown("---")
                        
                if 'filtered_retrievals' in locals():
                    st.subheader("3. Filtered Retrievals (Relevance Scored) (Agent: Reasoning)")
                    st.info("Agent called: Reasoning Agent")
                    for i, retrieval in enumerate(filtered_retrievals):
                        st.markdown(f"**Filtered Retrieval {i+1}**")
                        st.markdown(f"- **Text:** {retrieval['text'][:200]}...")
                        st.markdown(f"- **Relevance Score:** {retrieval.get('relevance_score', 0.0):.2f}")
                        st.markdown(f"- **Source:** {retrieval['metadata'].get('source_file', 'Unknown')}")
                        st.markdown(f"- **Page:** {retrieval['metadata'].get('page', 'N/A')}")
                        st.markdown(f"- **Section:** {retrieval['metadata'].get('section', 'N/A')}")
                        st.markdown("---")
                        
                st.subheader("4. Final Answer Generation (Agent: Answer)")
                st.info("Agent called: Answer Agent")
                st.markdown(f"- **Confidence:** {response.get('confidence_score', 0.0):.2f}")
                st.markdown(f"- **Reasoning:** {response.get('reasoning', 'No reasoning provided')}")
            formatted_response = f"""
            **Answer:** {response.get('short_answer', 'No answer available')}
            **Confidence:** {response.get('confidence_score', 0.0):.2f}
            **Reasoning:** {response.get('reasoning', 'No reasoning provided')}
            """
            if response.get('citations'):
                formatted_response += "\n\n**Sources:**\n"
                for i, citation in enumerate(response['citations'], 1):
                    formatted_response += f"{i}. {citation.get('source', 'Unknown')} "
                    formatted_response += f"(Page: {citation.get('page_reference', 'N/A')}, "
                    formatted_response += f"Section: {citation.get('section_id', 'N/A')}, "
                    formatted_response += f"Relevance: {citation.get('relevance_score', 0.0):.2f})\n"
            if response.get('entities_found'):
                with st.expander("🔍 Entity Analysis"):
                    st.json(response['entities_found'])
            st.markdown(formatted_response)
        orchestrator.session_agent.update_history(
            st.session_state.session_id,
            "assistant",
            formatted_response
        )


if __name__ == "__main__":
    main()

