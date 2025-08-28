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
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
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
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

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
        # Use proper formatting to avoid logging errors
        logger.info(f"Where Clause: {where}")
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
# RAGAS EVALUATION UTILITIES
# =====================================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=2, base_delay=10.0)  # Longer delays for free tier
def process_single_question_with_retry(orchestrator, question: str, ground_truths: List[str], top_k: int = 4):
    """Process a single question with retry logic and very conservative rate limiting for free tier."""
    # Much longer delay to respect free tier limits (10 req/min = 1 req every 6 seconds minimum)
    time.sleep(random.uniform(8.0, 12.0))  # 8-12 second delay between requests
    
    try:
        # Retrieve and score
        entities = orchestrator.retrieval_agent.extract_entities(question, None)
        filt = {"drug": {"$in": entities["drugs"]}} if entities.get("drugs") else None
        rets = orchestrator.retrieval_agent.vector_search(question, top_k=top_k, filter_metadata=filt)
        
        # Long delay between retrieval and reasoning
        time.sleep(8.0)
        
        scored = orchestrator.reasoning_agent.assess_chunk_relevance(question, rets)
        
        # Long delay between reasoning and answer generation  
        time.sleep(8.0)
        
        # Generate answer
        resp = orchestrator.answer_agent.generate_final_response(
            question, session_id="", retrievals=scored[:top_k], user_context=""
        )
        
        return {
            "question": question,
            "contexts": [r["text"] for r in scored[:top_k]] if scored else [""],
            "answer": resp.get("short_answer", ""),
            "ground_truths": ground_truths,
            "reference": ground_truths[0] if ground_truths else ""
        }
    except Exception as e:
        logger.error(f"Error processing question '{question}': {e}")
        return {
            "question": question,
            "contexts": [""],
            "answer": f"Error: {str(e)}",
            "ground_truths": ground_truths,
            "reference": ground_truths[0] if ground_truths else ""
        }

def load_test_dataset() -> List[Dict[str, Any]]:
    """Load the comprehensive test dataset from JSON file."""
    try:
        dataset_path = Path("test_dataset.json")
        if dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("test_dataset", [])
        else:
            st.warning("Test dataset file not found. Using fallback sample questions.")
            return []
    except Exception as e:
        st.error(f"Error loading test dataset: {e}")
        return []

def save_ragas_results(results: Dict[str, Any], filename: str = None):
    """Save RAGAS evaluation results to JSON file with timestamp."""
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ragas_results_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return filename
    except Exception as e:
        st.error(f"Error saving results: {e}")
        return None

def generate_research_report(metrics_result, categories: Dict[str, int], results_data: Dict[str, Any]):
    """Generate a comprehensive research report from RAGAS evaluation results."""
    st.header("📋 **Research Evaluation Report**")
    
    # Executive Summary
    st.subheader("📊 **Executive Summary**")
    total_questions = results_data.get('total_questions', 0)
    processing_time = results_data.get('total_time_hours', 0)
    
    summary_text = f"""
    **Large-Scale RAG System Evaluation**
    
    - **Evaluation Date**: {results_data.get('timestamp', 'N/A')}
    - **Total Questions Processed**: {total_questions}
    - **Processing Duration**: {processing_time:.1f} hours
    - **Evaluation Mode**: Large-scale research with {results_data.get('delay_minutes', 10)}-minute delays
    - **API Strategy**: Multi-key rotation across {len(GOOGLE_API_KEYS)} keys
    """
    st.markdown(summary_text)
    
    # Question Categories Analysis
    st.subheader("📈 **Question Categories Distribution**")
    if categories:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Category Breakdown:**")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_questions) * 100
                st.write(f"• **{category}**: {count} questions ({percentage:.1f}%)")
        
        with col2:
            # Create a simple bar chart representation
            st.write("**Visual Distribution:**")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                bar_length = int((count / max(categories.values())) * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                st.write(f"{category[:15]:<15} {bar} {count}")
    
    # RAGAS Metrics Analysis
    if metrics_result:
        st.subheader("🔬 **RAGAS Metrics Analysis**")
        
        try:
            df = metrics_result.to_pandas()
            summary = {k: float(v) for k, v in metrics_result.summary().items()}
            
            # Overall Performance Assessment
            overall_score = sum(summary.values()) / len(summary)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏆 **Overall Score**", f"{overall_score:.3f}")
                if overall_score >= 0.8:
                    st.success("🥇 Excellent Performance")
                elif overall_score >= 0.7:
                    st.info("🥈 Good Performance")
                elif overall_score >= 0.6:
                    st.warning("🥉 Fair Performance")
                else:
                    st.error("⚠️ Needs Improvement")
            
            with col2:
                st.metric("📊 **Questions Evaluated**", total_questions)
                st.metric("⏱️ **Processing Time**", f"{processing_time:.1f}h")
            
            with col3:
                best_metric = max(summary.items(), key=lambda x: x[1])
                worst_metric = min(summary.items(), key=lambda x: x[1])
                st.metric("🔝 **Best Metric**", best_metric[0].replace('_', ' ').title())
                st.metric("📉 **Lowest Metric**", worst_metric[0].replace('_', ' ').title())
            
            # Detailed Metrics Table
            st.subheader("📋 **Detailed Metrics Breakdown**")
            
            metrics_data = []
            for metric, score in summary.items():
                performance_level = "Excellent" if score >= 0.8 else "Good" if score >= 0.7 else "Fair" if score >= 0.6 else "Poor"
                metrics_data.append({
                    "Metric": metric.replace('_', ' ').title(),
                    "Score": f"{score:.3f}",
                    "Performance": performance_level,
                    "Recommendation": get_metric_recommendation(metric, score)
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Statistical Analysis
            st.subheader("📊 **Statistical Analysis**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Score Distribution:**")
                score_ranges = {
                    "Excellent (≥0.8)": sum(1 for s in summary.values() if s >= 0.8),
                    "Good (0.7-0.8)": sum(1 for s in summary.values() if 0.7 <= s < 0.8),
                    "Fair (0.6-0.7)": sum(1 for s in summary.values() if 0.6 <= s < 0.7),
                    "Poor (<0.6)": sum(1 for s in summary.values() if s < 0.6)
                }
                
                for range_name, count in score_ranges.items():
                    if count > 0:
                        st.write(f"• {range_name}: {count} metrics")
            
            with col2:
                st.write("**Variance Analysis:**")
                scores = list(summary.values())
                std_dev = pd.Series(scores).std()
                variance = pd.Series(scores).var()
                
                st.write(f"• **Mean Score**: {overall_score:.3f}")
                st.write(f"• **Standard Deviation**: {std_dev:.3f}")
                st.write(f"• **Variance**: {variance:.3f}")
                st.write(f"• **Score Range**: {min(scores):.3f} - {max(scores):.3f}")
        
        except Exception as e:
            st.error(f"Error analyzing metrics: {e}")
    
    # Recommendations Section
    st.subheader("🎯 **Recommendations & Next Steps**")
    
    recommendations = []
    
    if metrics_result:
        try:
            summary = {k: float(v) for k, v in metrics_result.summary().items()}
            
            # Context-based recommendations
            if 'context_precision' in summary and summary['context_precision'] < 0.7:
                recommendations.append("🔍 **Improve Context Precision**: Enhance retrieval algorithms to reduce irrelevant context")
            
            if 'context_recall' in summary and summary['context_recall'] < 0.7:
                recommendations.append("📚 **Enhance Context Recall**: Increase context window or improve chunk overlap")
            
            if 'answer_relevancy' in summary and summary['answer_relevancy'] < 0.7:
                recommendations.append("🎯 **Boost Answer Relevancy**: Fine-tune prompt engineering and response generation")
            
            if 'faithfulness' in summary and summary['faithfulness'] < 0.7:
                recommendations.append("✅ **Improve Faithfulness**: Strengthen fact-checking and grounding mechanisms")
            
            if 'answer_correctness' in summary and summary['answer_correctness'] < 0.7:
                recommendations.append("🎓 **Enhance Answer Correctness**: Improve knowledge base quality and model training")
        
        except Exception as e:
            st.warning(f"Could not generate specific recommendations: {e}")
    
    # General recommendations
    recommendations.extend([
        f"📈 **Scale Testing**: Current evaluation used {total_questions} questions - consider expanding to 100+ for production",
        "🔄 **Continuous Monitoring**: Implement regular RAGAS evaluations for system maintenance",
        "📊 **A/B Testing**: Compare different retrieval strategies using this evaluation framework",
        "🎯 **Domain-Specific Tuning**: Focus on categories with lower performance scores"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Export Section
    st.subheader("📁 **Research Data Export**")
    
    # Create comprehensive export data
    export_data = {
        "research_metadata": {
            "evaluation_type": "large_scale_research",
            "timestamp": results_data.get('timestamp'),
            "total_questions": total_questions,
            "processing_hours": processing_time,
            "delay_minutes": results_data.get('delay_minutes'),
            "api_keys_used": len(GOOGLE_API_KEYS)
        },
        "category_distribution": categories,
        "ragas_metrics": summary if metrics_result else {},
        "statistical_analysis": {
            "overall_score": overall_score if metrics_result else 0,
            "score_distribution": score_ranges if metrics_result else {},
            "recommendations": recommendations
        }
    }
    
    # JSON Export
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 **Download Full Research Data (JSON)**",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name=f"research_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        if metrics_result:
            csv_data = metrics_result.to_pandas().to_csv(index=False)
            st.download_button(
                label="📊 **Download Metrics CSV**",
                data=csv_data,
                file_name=f"ragas_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Research Summary Report
    st.subheader("📝 **Research Summary for Publication**")
    
    research_summary = f"""
# RAG System Evaluation Report

## Executive Summary
This report presents the results of a comprehensive evaluation of a Retrieval-Augmented Generation (RAG) system for drug information queries using the RAGAS evaluation framework.

## Methodology
- **Evaluation Framework**: RAGAS v0.3.2
- **Dataset Size**: {total_questions} questions across {len(categories)} categories
- **Evaluation Duration**: {processing_time:.1f} hours with {results_data.get('delay_minutes', 10)}-minute delays
- **Rate Limiting Strategy**: Multi-API key rotation across {len(GOOGLE_API_KEYS)} keys

## Key Findings
- **Overall Performance Score**: {overall_score:.3f} ({get_performance_level(overall_score)})
- **Question Categories**: {', '.join([f"{cat} ({count})" for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]])}
- **Processing Efficiency**: {total_questions/processing_time:.1f} questions per hour

## Performance Metrics
{chr(10).join([f"- **{metric.replace('_', ' ').title()}**: {score:.3f}" for metric, score in (summary.items() if metrics_result else [])])}

## Recommendations
{chr(10).join([f"{i}. {rec.replace('🔍 ', '').replace('📚 ', '').replace('🎯 ', '').replace('✅ ', '').replace('🎓 ', '').replace('📈 ', '').replace('🔄 ', '').replace('📊 ', '')}" for i, rec in enumerate(recommendations[:5], 1)])}

## Conclusion
{"This evaluation demonstrates excellent RAG system performance with robust accuracy and retrieval capabilities." if overall_score >= 0.8 else "This evaluation shows good RAG system performance with opportunities for targeted improvements." if overall_score >= 0.7 else "This evaluation indicates fair RAG system performance requiring systematic optimization." if overall_score >= 0.6 else "This evaluation reveals significant opportunities for RAG system improvement across multiple metrics."}

*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')} using automated RAGAS evaluation pipeline.*
"""
    
    st.text_area("📋 **Copy for Research Publication**", research_summary, height=400)

def get_metric_recommendation(metric: str, score: float) -> str:
    """Get specific recommendations based on metric performance."""
    recommendations = {
        'context_precision': {
            'low': "Improve retrieval filtering and ranking algorithms",
            'medium': "Fine-tune similarity thresholds and context selection",
            'high': "Maintain current retrieval strategy"
        },
        'context_recall': {
            'low': "Increase retrieval window and improve chunking strategy", 
            'medium': "Optimize chunk size and overlap parameters",
            'high': "Current context coverage is excellent"
        },
        'answer_relevancy': {
            'low': "Enhance prompt engineering and response filtering",
            'medium': "Fine-tune answer generation parameters", 
            'high': "Answer relevancy is optimal"
        },
        'faithfulness': {
            'low': "Implement stronger fact-checking and source grounding",
            'medium': "Improve citation accuracy and content verification",
            'high': "Faithfulness to sources is excellent"
        },
        'answer_correctness': {
            'low': "Enhance knowledge base quality and model training",
            'medium': "Improve factual accuracy checking mechanisms",
            'high': "Answer correctness is highly reliable"
        }
    }
    
    if score >= 0.8:
        level = 'high'
    elif score >= 0.6:
        level = 'medium'
    else:
        level = 'low'
    
    return recommendations.get(metric, {}).get(level, "Monitor and optimize as needed")

def get_performance_level(score: float) -> str:
    """Get performance level description."""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.6:
        return "Fair"
    else:
        return "Needs Improvement"

def build_ragas_dataset_batch(
    orchestrator,
    samples: List[Dict[str, Any]],
    top_k: int = 4,
) -> Dataset:
    """Build a HuggingFace Dataset for multiple questions with intelligent rate limiting and aggregation.

    Expected sample schema per item:
    {"question": str, "ground_truths": Optional[List[str]]}
    """
    questions: List[str] = []
    contexts: List[List[str]] = []
    answers: List[str] = []
    gts: List[List[str]] = []
    references: List[str] = []

    total_questions = len(samples)
    st.info(f"🚀 **Batch Processing**: {total_questions} questions with {len(GOOGLE_API_KEYS)} API keys")
    
    # Calculate optimal delays based on number of API keys
    base_delay = max(8.0, 60.0 / len(GOOGLE_API_KEYS))  # Ensure we don't exceed rate limits
    st.info(f"⏱️ **Processing Strategy**: {base_delay:.1f}s delay between operations")
    
    # Create a progress container
    progress_container = st.container()
    
    with progress_container:
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # Process questions sequentially with intelligent delays and key rotation
        for i, s in enumerate(samples):
            q = s.get("question", "").strip()
            if not q:
                continue
                
            gt_list = s.get("ground_truths", [])
            
            # Update progress
            progress_percent = int((i / total_questions) * 100)
            overall_progress.progress(progress_percent)
            status_text.text(f"🔄 Processing question {i+1}/{total_questions}: {q[:60]}...")
            
            # Show current API key being used
            st.info(f"🔑 Using API Key #{current_key_index + 1} for question {i+1}")
            
            try:
                result = process_single_question_with_retry(orchestrator, q, gt_list, top_k)
                
                questions.append(result["question"])
                contexts.append(result["contexts"])
                answers.append(result["answer"])
                gts.append(result["ground_truths"])
                references.append(result["reference"])
                
                # Rotate API key for next question to distribute load
                if i < total_questions - 1 and len(GOOGLE_API_KEYS) > 1:
                    rotate_api_key()
                    st.success(f"✅ Question {i+1} completed. Rotated to API Key #{current_key_index + 1}")
                else:
                    st.success(f"✅ Question {i+1} completed.")
                
                # Add delay between questions (shorter with more keys)
                if i < total_questions - 1:  # Don't delay after last question
                    delay_time = base_delay + random.uniform(2.0, 4.0)
                    status_text.text(f"⏳ Waiting {delay_time:.1f}s before next question...")
                    time.sleep(delay_time)
                
            except Exception as e:
                st.error(f"❌ Failed to process question {i+1}: {e}")
                # Add empty entries to maintain dataset consistency
                questions.append(q)
                contexts.append([""])
                answers.append(f"Error: {str(e)}")
                gts.append(gt_list)
                references.append(gt_list[0] if gt_list else "")
                
                # Try rotating key on error
                if len(GOOGLE_API_KEYS) > 1:
                    rotate_api_key()
                    st.warning(f"🔄 Rotated to API Key #{current_key_index + 1} after error")
        
        overall_progress.progress(100)
        status_text.text("✅ All questions processed!")

    return Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truths": gts,
            "reference": references,
        }
    )

def build_ragas_dataset_batch(
    orchestrator,
    samples: List[Dict[str, Any]],
    top_k: int = 4,
) -> Dataset:
    """Build a HuggingFace Dataset for multiple questions with intelligent rate limiting and aggregation.

    Expected sample schema per item:
    {"question": str, "ground_truths": Optional[List[str]]}
    """
    questions: List[str] = []
    contexts: List[List[str]] = []
    answers: List[str] = []
    gts: List[List[str]] = []
    references: List[str] = []

    total_questions = len(samples)
    st.info(f"🚀 **Batch Processing**: {total_questions} questions with {len(GOOGLE_API_KEYS)} API keys")
    
    # Calculate optimal delays based on number of API keys
    base_delay = max(8.0, 60.0 / len(GOOGLE_API_KEYS))  # Ensure we don't exceed rate limits
    st.info(f"⏱️ **Processing Strategy**: {base_delay:.1f}s delay between operations")
    
    # Create a progress container
    progress_container = st.container()
    
    with progress_container:
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # Process questions sequentially with intelligent delays and key rotation
        for i, s in enumerate(samples):
            q = s.get("question", "").strip()
            if not q:
                continue
                
            gt_list = s.get("ground_truths", [])
            
            # Update progress
            progress_percent = int((i / total_questions) * 100)
            overall_progress.progress(progress_percent)
            status_text.text(f"🔄 Processing question {i+1}/{total_questions}: {q[:60]}...")
            
            # Show current API key being used
            st.info(f"🔑 Using API Key #{current_key_index + 1} for question {i+1}")
            
            try:
                result = process_single_question_with_retry(orchestrator, q, gt_list, top_k)
                
                questions.append(result["question"])
                contexts.append(result["contexts"])
                answers.append(result["answer"])
                gts.append(result["ground_truths"])
                references.append(result["reference"])
                
                # Rotate API key for next question to distribute load
                if i < total_questions - 1 and len(GOOGLE_API_KEYS) > 1:
                    rotate_api_key()
                    st.success(f"✅ Question {i+1} completed. Rotated to API Key #{current_key_index + 1}")
                else:
                    st.success(f"✅ Question {i+1} completed.")
                
                # Add delay between questions (shorter with more keys)
                if i < total_questions - 1:  # Don't delay after last question
                    delay_time = base_delay + random.uniform(2.0, 4.0)
                    status_text.text(f"⏳ Waiting {delay_time:.1f}s before next question...")
                    time.sleep(delay_time)
                
            except Exception as e:
                st.error(f"❌ Failed to process question {i+1}: {e}")
                # Add empty entries to maintain dataset consistency
                questions.append(q)
                contexts.append([""])
                answers.append(f"Error: {str(e)}")
                gts.append(gt_list)
                references.append(gt_list[0] if gt_list else "")
                
                # Try rotating key on error
                if len(GOOGLE_API_KEYS) > 1:
                    rotate_api_key()
                    st.warning(f"🔄 Rotated to API Key #{current_key_index + 1} after error")
        
        overall_progress.progress(100)
        status_text.text("✅ All questions processed!")

    return Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truths": gts,
            "reference": references,
        }
    )

def process_large_batch_with_delays(
    orchestrator,
    samples: List[Dict[str, Any]],
    delay_minutes: int = 10,
    top_k: int = 4,
) -> Dataset:
    """Process large batch of questions with configurable delays between each question.
    
    Args:
        orchestrator: The RAG orchestrator
        samples: List of questions with ground truths
        delay_minutes: Minutes to wait between each question (default 10)
        top_k: Number of top contexts to retrieve
    """
    questions: List[str] = []
    contexts: List[List[str]] = []
    answers: List[str] = []
    gts: List[List[str]] = []
    references: List[str] = []

    total_questions = len(samples)
    delay_seconds = delay_minutes * 60
    
    st.success(f"🎯 **Large Scale Evaluation**: {total_questions} questions with {delay_minutes}-minute delays")
    st.info(f"⏱️ **Total Estimated Time**: {(total_questions * delay_minutes):.0f} minutes ({(total_questions * delay_minutes / 60):.1f} hours)")
    st.info(f"🔑 **API Strategy**: {len(GOOGLE_API_KEYS)} keys with auto-rotation every question")
    
    # Create a detailed progress tracking system
    progress_container = st.container()
    
    with progress_container:
        # Overall progress
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # Detailed status container
        details_container = st.container()
        
        # Results tracking
        successful_questions = 0
        failed_questions = 0
        start_time = time.time()
        
        for i, s in enumerate(samples):
            q = s.get("question", "").strip()
            category = s.get("category", "Unknown")
            
            if not q:
                continue
                
            gt_list = s.get("ground_truths", [])
            if isinstance(s.get("ground_truth"), str):
                gt_list = [s.get("ground_truth")]
            
            # Update overall progress
            progress_percent = int((i / total_questions) * 100)
            overall_progress.progress(progress_percent)
            
            # Current question info
            with details_container:
                st.subheader(f"🔄 Processing Question {i+1}/{total_questions}")
                st.info(f"**Category**: {category}")
                st.info(f"**Question**: {q}")
                st.info(f"**API Key**: #{current_key_index + 1}")
                
                # Time estimates
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_question = elapsed_time / i
                    remaining_questions = total_questions - i
                    estimated_remaining = (remaining_questions * avg_time_per_question) / 60
                    st.info(f"⏱️ **Progress**: {elapsed_time/60:.1f}min elapsed, ~{estimated_remaining:.0f}min remaining")
                
                # Statistics
                st.info(f"📊 **Status**: ✅ {successful_questions} successful, ❌ {failed_questions} failed")
            
            try:
                # Process the question
                status_text.text(f"Processing question {i+1}: {q[:60]}...")
                
                result = process_single_question_with_retry(orchestrator, q, gt_list, top_k)
                
                questions.append(result["question"])
                contexts.append(result["contexts"])
                answers.append(result["answer"])
                gts.append(result["ground_truths"])
                references.append(result["reference"])
                
                successful_questions += 1
                
                # Rotate API key for next question
                if len(GOOGLE_API_KEYS) > 1:
                    rotate_api_key()
                
                st.success(f"✅ Question {i+1} completed successfully!")
                
                # Save intermediate results every 10 questions
                if (i + 1) % 10 == 0:
                    intermediate_data = {
                        "partial_results": {
                            "questions": questions,
                            "contexts": contexts,
                            "answers": answers,
                            "ground_truths": gts,
                            "references": references
                        },
                        "progress": {
                            "completed": i + 1,
                            "total": total_questions,
                            "successful": successful_questions,
                            "failed": failed_questions,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    backup_file = save_ragas_results(intermediate_data, f"backup_results_{i+1}_questions.json")
                    if backup_file:
                        st.info(f"💾 Backup saved: {backup_file}")
                
                # Delay before next question (except for the last one)
                if i < total_questions - 1:
                    st.info(f"⏳ **Waiting {delay_minutes} minutes before next question...**")
                    
                    # Countdown timer
                    countdown_placeholder = st.empty()
                    for remaining in range(delay_seconds, 0, -1):
                        mins, secs = divmod(remaining, 60)
                        countdown_placeholder.info(f"⏱️ Next question in: {mins:02d}:{secs:02d}")
                        time.sleep(1)
                    countdown_placeholder.empty()
                
            except Exception as e:
                failed_questions += 1
                st.error(f"❌ Failed to process question {i+1}: {e}")
                
                # Add empty entries to maintain dataset consistency
                questions.append(q)
                contexts.append([""])
                answers.append(f"Error: {str(e)}")
                gts.append(gt_list)
                references.append(gt_list[0] if gt_list else "")
                
                # Try rotating key on error
                if len(GOOGLE_API_KEYS) > 1:
                    rotate_api_key()
                    st.warning(f"🔄 Rotated to API Key #{current_key_index + 1} after error")
                
                # Still wait before next question even on error
                if i < total_questions - 1:
                    st.info(f"⏳ Waiting {delay_minutes} minutes before next question (even after error)...")
                    time.sleep(delay_seconds)
        
        # Final summary
        total_time = time.time() - start_time
        overall_progress.progress(100)
        status_text.text("✅ All questions processed!")
        
        st.balloons()
        st.success(f"🎉 **Large Scale Evaluation Complete!**")
        st.info(f"📊 **Final Statistics**: ✅ {successful_questions} successful, ❌ {failed_questions} failed")
        st.info(f"⏱️ **Total Time**: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")

    return Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truths": gts,
            "reference": references,
        }
    )


def run_ragas_evaluation(dataset: Dataset, include_correctness: bool = False, use_all_metrics: bool = False):
    """Run RAGAS metrics using Gemini (via LangChain) and HF embeddings with intelligent rate limiting."""
    
    # Display API key status
    st.info(f"🔑 **API Keys Available**: {len(GOOGLE_API_KEYS)} | **Currently Using**: Key #{current_key_index + 1}")
    
    try:
        # Use current active API key
        active_key = GOOGLE_API_KEYS[current_key_index] if GOOGLE_API_KEYS else os.getenv("GOOGLE_API_KEY", "")
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, 
            google_api_key=active_key,
            temperature=0.1,
            max_retries=2,
            request_timeout=60  # Longer timeout
        )
    except Exception:
        # Fallback to default construction
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, 
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            temperature=0.1,
            max_retries=2,
            request_timeout=60
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if use_all_metrics:
        # All available RAGAS metrics for comprehensive evaluation
        metrics = [context_precision, context_recall, answer_relevancy, faithfulness]
        if include_correctness:
            metrics.append(answer_correctness)
        
        st.success("🔥 **Full RAGAS Evaluation Mode** - Using all metrics")
        st.info("📊 **Metrics**: Context Precision, Context Recall, Answer Relevancy, Faithfulness" + 
                (" + Answer Correctness" if include_correctness else ""))
        
        if len(GOOGLE_API_KEYS) > 1:
            st.info(f"⚡ **Multi-Key Advantage**: {len(GOOGLE_API_KEYS)} API keys available for automatic fallback")
            st.info("⏱️ **Estimated Time**: 4-8 minutes (faster with multiple keys)")
        else:
            st.warning("⏱️ **Single Key**: 8-12 minutes (consider adding more API keys for faster evaluation)")
        
        # Shorter delay with multiple keys
        delay_time = 30.0 if len(GOOGLE_API_KEYS) > 1 else 60.0
        st.info(f"Waiting {int(delay_time)} seconds before full evaluation...")
        time.sleep(delay_time)
        
    else:
        # Minimal metrics for quick evaluation
        metrics = [answer_relevancy]  # Only 1 metric to minimize API usage
        st.info("📊 **Quick Mode**: Answer Relevancy only")
        delay_time = 10.0 if len(GOOGLE_API_KEYS) > 1 else 15.0
        time.sleep(delay_time)
    
    try:
        result = ragas_evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        return result
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check if it's a quota error and we have more keys
        if any(quota_error in error_msg for quota_error in 
               ['quota', 'rate limit', 'resource_exhausted', 'too many requests']) and len(GOOGLE_API_KEYS) > 1:
            
            st.warning(f"🔄 **Auto-Rotating**: Key #{current_key_index + 1} quota exceeded, trying next key...")
            rotate_api_key()
            
            # Try with the new key
            try:
                new_active_key = GOOGLE_API_KEYS[current_key_index]
                llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL, 
                    google_api_key=new_active_key,
                    temperature=0.1,
                    max_retries=2,
                    request_timeout=60
                )
                
                st.info(f"🔑 **Now Using**: Key #{current_key_index + 1}")
                time.sleep(15.0)  # Brief delay before retry
                
                result = ragas_evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
                return result
                
            except Exception as e2:
                st.error(f"Retry with new key also failed: {e2}")
        
        st.error(f"RAGAS evaluation failed: {e}")
        
        # Provide specific guidance based on number of keys
        if len(GOOGLE_API_KEYS) == 1:
            st.info("💡 **Single API Key Limitations**:")
            st.info("1. Only 10 requests per minute on free tier")
            st.info("2. Consider adding more API keys to .env file")
            st.info("3. Wait 2-3 minutes between evaluations")
        else:
            st.info("💡 **Multi-Key Setup Issues**:")
            st.info("1. All keys may have reached quota")
            st.info("2. Wait 2-3 minutes for quota reset")
            st.info("3. Verify all keys in .env file are valid")
        
        # If full metrics failed, offer fallback
        if use_all_metrics and len(metrics) > 1:
            st.warning("� **Fallback Strategy**: Try Quick Mode (Answer Relevancy only)")
        
        raise e

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
        # RAGAS Evaluation UI
        st.subheader("RAGAS Evaluation")
        with st.expander("Run quick RAGAS eval"):
            st.caption("Enter 1-5 test questions; optionally add ground truths (one per line).")
            
            # Pre-populated sample questions for easy testing
            if st.button("📋 Load 5 Sample Questions for Batch Evaluation", key="load_samples"):
                # Question 1 - Dosage Information
                st.session_state.ragas_q_0 = "What is the recommended dosage of Humira for rheumatoid arthritis?"
                st.session_state.ragas_gt_0 = "The recommended dosage of Humira for rheumatoid arthritis in adults is 40 mg administered subcutaneously every other week. Some patients may benefit from increasing the frequency to 40 mg every week if they are not taking methotrexate concomitantly."
                
                # Question 2 - Safety Information
                st.session_state.ragas_q_1 = "What are the contraindications for Wegovy?"
                st.session_state.ragas_gt_1 = "Wegovy is contraindicated in patients with a personal or family history of medullary thyroid carcinoma (MTC), patients with Multiple Endocrine Neoplasia syndrome type 2 (MEN 2), and patients with known hypersensitivity to semaglutide or any of the excipients in Wegovy."
                
                # Question 3 - Black Box Warnings
                st.session_state.ragas_q_2 = "What are the black box warnings for Rinvoq?"
                st.session_state.ragas_gt_2 = "Rinvoq carries black box warnings for increased risk of serious infections that may lead to hospitalization or death, lymphoma and other malignancies, and major adverse cardiovascular events including cardiovascular death, myocardial infarction, and stroke."
                
                # Question 4 - Mechanism of Action
                st.session_state.ragas_q_3 = "What is the mechanism of action of Stelara?"
                st.session_state.ragas_gt_3 = "Stelara is a human monoclonal antibody that binds to the p40 protein subunit shared by both interleukin (IL)-12 and IL-23 cytokines. By binding to this subunit, Stelara prevents IL-12 and IL-23 from binding to their receptor protein (IL-12Rβ1) on the surface of immune cells, thereby inhibiting inflammatory responses."
                
                # Question 5 - Drug Interactions
                st.session_state.ragas_q_4 = "Can Humira be used with live vaccines?"
                st.session_state.ragas_gt_4 = "No, live vaccines should not be administered to patients receiving Humira. Treatment with Humira may result in immunosuppression and patients may have a decreased ability to fight infections. Live vaccines could potentially cause serious infections in immunocompromised patients."
                
                st.success("✅ Loaded 5 comprehensive sample questions covering dosage, safety, warnings, mechanism of action, and drug interactions!")
                st.rerun()
            
            n = st.number_input("# Samples", min_value=1, max_value=5, value=5, step=1, key="ragas_n")
            
            st.info(f"📊 **Batch Evaluation**: Process {n} questions for comprehensive analysis")
            if n >= 3:
                st.success(f"✅ **Report Ready**: {n} questions provide robust evaluation metrics")
            
            samples: List[Dict[str, Any]] = []
            for i in range(int(n)):
                with st.expander(f"Question {i+1}", expanded=(i < 2)):  # Keep first 2 expanded
                    q = st.text_input(f"Q{i+1}", key=f"ragas_q_{i}")
                    gt_raw = st.text_area(f"Ground truth for Q{i+1} (optional)", key=f"ragas_gt_{i}")
                    gt_list = [g.strip() for g in gt_raw.splitlines() if g.strip()]
                    if q:
                        samples.append({"question": q, "ground_truths": gt_list})
            
            # Add option for full metrics
            st.subheader("🎯 Evaluation Configuration")
            use_all_metrics = st.checkbox(
                "🔥 **Use All RAGAS Metrics** (for comprehensive report)", 
                value=True,  # Default to True for batch processing
                key="use_all_metrics",
                help="Enable this for complete evaluation with all metrics: Context Precision, Context Recall, Answer Relevancy, Faithfulness"
            )
            
            if use_all_metrics:
                st.success("🎉 **Full Evaluation Mode** - Perfect for Research Reports")
                st.info("📊 **Metrics**: Context Precision, Context Recall, Answer Relevancy, Faithfulness")
                
                # Calculate time estimate based on number of API keys and questions
                estimated_minutes = (len(samples) * 3) // len(GOOGLE_API_KEYS) + 2
                st.info(f"⏱️ **Estimated Time**: {estimated_minutes}-{estimated_minutes+2} minutes with {len(GOOGLE_API_KEYS)} API keys")
                st.info(f"� **Multi-Key Advantage**: Using {len(GOOGLE_API_KEYS)} keys for parallel processing")
                
                include_correctness = st.checkbox("Include Answer Correctness (needs ground truths)", value=True, key="ragas_correctness")
                
            else:
                st.info("⚡ **Quick Mode**: Answer Relevancy only")
                estimated_minutes = len(samples) // len(GOOGLE_API_KEYS) + 1
                st.info(f"⏱️ **Estimated Time**: {estimated_minutes}-{estimated_minutes+1} minutes")
                include_correctness = False
            if st.button("🚀 Run Batch RAGAS Evaluation", key="btn_ragas"):
                if not samples:
                    st.warning("Add at least one question.")
                elif len(samples) > 5:
                    st.error("🚫 **Maximum 5 questions** allowed for batch processing")
                else:
                    evaluation_type = "Full RAGAS Batch Evaluation" if use_all_metrics else "Quick RAGAS Batch Evaluation"
                    st.info(f"🎯 Starting {evaluation_type} for {len(samples)} questions...")
                    
                    if use_all_metrics:
                        st.success(f"🔥 **Comprehensive Analysis**: All RAGAS metrics for {len(samples)} questions")
                        st.info("📊 **Metrics**: Context Precision, Context Recall, Answer Relevancy, Faithfulness" + 
                               (" + Answer Correctness" if include_correctness else ""))
                        estimated_time = (len(samples) * 3) // len(GOOGLE_API_KEYS) + 2
                        st.warning(f"⏱️ **Processing Time**: {estimated_time}-{estimated_time+2} minutes with {len(GOOGLE_API_KEYS)} API keys")
                    else:
                        st.info(f"⚡ **Quick Batch**: Answer Relevancy for {len(samples)} questions")
                        estimated_time = len(samples) // len(GOOGLE_API_KEYS) + 1
                        st.info(f"⏱️ **Processing Time**: {estimated_time}-{estimated_time+1} minutes")
                    
                    # Start batch processing
                    start_time = time.time()
                    
                    try:
                        st.info("🔄 **Phase 1**: Building dataset with intelligent rate limiting...")
                        ds = build_ragas_dataset_batch(orchestrator, samples)
                        
                        processing_time = time.time() - start_time
                        st.success(f"✅ **Dataset Built**: {len(samples)} questions processed in {processing_time/60:.1f} minutes")
                        
                        st.info("🔄 **Phase 2**: Running RAGAS evaluation...")
                        eval_start = time.time()
                        res = run_ragas_evaluation(ds, include_correctness=include_correctness, use_all_metrics=use_all_metrics)
                        eval_time = time.time() - eval_start
                        
                        total_time = time.time() - start_time
                        
                        # Display comprehensive results
                        st.balloons()  # Celebration for completing batch evaluation!
                        st.success(f"🎉 **Batch Evaluation Complete!** Total time: {total_time/60:.1f} minutes")
                        
                        # Show detailed results
                        st.header("📊 **Comprehensive RAGAS Results**")
                        
                        # Display the dataframe with all question results
                        df = res.to_pandas()
                        st.subheader("📋 **Individual Question Results**")
                        st.dataframe(df, use_container_width=True)
                        
                        # Calculate and display aggregated metrics
                        try:
                            summary = {k: float(v) for k, v in res.summary().items()}
                            
                            st.subheader("📈 **Aggregated Performance Metrics**")
                            st.info(f"📊 **Based on {len(samples)} questions** - Perfect for research reports and analysis")
                            
                            # Display metrics in a professional layout
                            metric_cols = st.columns(len(summary))
                            for i, (metric, score) in enumerate(summary.items()):
                                with metric_cols[i]:
                                    # Color code the metrics
                                    color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                                    st.metric(
                                        label=f"{color} {metric.replace('_', ' ').title()}",
                                        value=f"{score:.3f}",
                                        help=f"Average score across {len(samples)} questions: {score:.6f}"
                                    )
                            
                            # Overall score calculation
                            overall_score = sum(summary.values()) / len(summary)
                            st.subheader("🏆 **Overall RAGAS Score**")
                            
                            if overall_score >= 0.8:
                                st.success(f"🥇 **Excellent**: {overall_score:.3f} - Your RAG system performs exceptionally well!")
                            elif overall_score >= 0.7:
                                st.success(f"🥈 **Good**: {overall_score:.3f} - Your RAG system performs well with room for improvement")
                            elif overall_score >= 0.6:
                                st.warning(f"🥉 **Fair**: {overall_score:.3f} - Your RAG system needs optimization")
                            else:
                                st.error(f"⚠️ **Needs Improvement**: {overall_score:.3f} - Consider enhancing your RAG pipeline")
                            
                            # Export options for reports
                            with st.expander("📁 **Export Results for Report**"):
                                # JSON format
                                st.subheader("🔢 **Raw Metrics (JSON)**")
                                export_data = {
                                    "evaluation_summary": {
                                        "total_questions": len(samples),
                                        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "processing_time_minutes": round(total_time/60, 2),
                                        "overall_score": round(overall_score, 4),
                                        "metrics": summary
                                    },
                                    "individual_results": df.to_dict('records')
                                }
                                st.json(export_data)
                                
                                # CSV format
                                st.subheader("� **CSV Export**")
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name=f"ragas_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                # Report summary text
                                st.subheader("📝 **Report Summary Text**")
                                report_text = f"""
RAGAS Evaluation Summary
========================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Questions Evaluated: {len(samples)}
Processing Time: {total_time/60:.1f} minutes
Overall Score: {overall_score:.3f}

Detailed Metrics:
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v:.3f}" for k, v in summary.items()])}

System Performance: {'Excellent' if overall_score >= 0.8 else 'Good' if overall_score >= 0.7 else 'Fair' if overall_score >= 0.6 else 'Needs Improvement'}
"""
                                st.text_area("Copy this for your report:", report_text, height=200)
                                
                        except Exception as json_error:
                            st.warning(f"Could not format summary: {json_error}")
                            
                    except Exception as e:
                        st.error(f"Batch RAGAS evaluation failed: {e}")
                        st.info("💡 **Batch Processing Tips**:")
                        st.info("1. Ensure stable internet connection")
                        st.info("2. All API keys are valid and have quota available")
                        st.info("3. Try reducing number of questions if issues persist")
                        st.info("4. Check that questions are well-formed")
        st.header("Configuration")
        st.info(f"**Model:** {GEMINI_MODEL}")
        st.info(f"**API Keys Available:** {len(GOOGLE_API_KEYS)}")
        st.info(f"**Current Key:** #{current_key_index + 1}")
        st.info(f"**Confidence Threshold:** {CONFIDENCE_THRESHOLD}")
        st.info(f"**Session ID:** {st.session_state.session_id}")

        # API Key Management
        if len(GOOGLE_API_KEYS) > 1:
            st.subheader("🔑 API Key Management")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Rotate API Key", help="Switch to next API key"):
                    rotate_api_key()
                    st.success(f"Switched to API Key #{current_key_index + 1}")
                    st.rerun()
            with col2:
                if st.button("📊 Test Current Key", help="Test if current API key is working"):
                    try:
                        model = genai.GenerativeModel(GEMINI_MODEL)
                        response = model.generate_content("Hello")
                        st.success(f"✅ API Key #{current_key_index + 1} is working!")
                    except Exception as e:
                        st.error(f"❌ API Key #{current_key_index + 1} failed: {str(e)[:100]}...")
            
            # Show all key status
            with st.expander("📋 All API Keys Status"):
                for i, key in enumerate(GOOGLE_API_KEYS):
                    status = "🟢 Current" if i == current_key_index else "⚪ Available"
                    masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else key[:8] + "..."
                    st.text(f"Key #{i+1}: {status} - {masked_key}")

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

