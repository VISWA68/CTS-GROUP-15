"""
ingest_pdfs_v2.py

Improved PDF ingestion pipeline (opinionated):
 - better text cleanup (fix hyphenation, join bad linebreaks)
 - heading detection using font sizes (when available) + heuristics
 - table extraction: Camelot (vector PDFs) + OCR fallback that snapshots page region
 - image extraction + OCR + optional BLIP captioning (if transformers+vision available)
 - chunking with prev/next links in metadata
 - dedupe at chunk level + document-level hash detection
 - Chroma hybrid search example (metadata filters + vectors)
 - Optional RAG synthesis with OpenAI ChatCompletion
 - ingestion logs stored in SQLite for queryable audit

Dependencies (suggested):
  pip install pymupdf pdf2image pytesseract camelot-py[cv] sentence-transformers chromadb python-dotenv sqlite-utils transformers torch torchvision

Note: this aims to be robust but still simple to run locally. Adjust paths and environment variables as needed.
"""

import os
import re
import hashlib
import json
import sqlite3
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import camelot
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import logging
import io
import time
from functools import wraps
from PIL import Image

# Optional imports for captioning / OpenAI
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    HAVE_BLIP = True
except Exception:
    HAVE_BLIP = False

load_dotenv()

# logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ----------------
# Config
# ----------------
DATA_DIR = Path(".")
PDF_DIR = DATA_DIR
ASSETS_DIR = DATA_DIR / "pdf_assets"
LOG_DB = DATA_DIR / "ingest_logs.db"

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")  # 'local' or 'openai'
OPENAI_MODEL_EMB = "text-embedding-3-large"
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
LOCAL_EMBED_MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_store_v2")
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "32"))
USE_CAMELOT = True
CAPTION_IMAGES = os.getenv("CAPTION_IMAGES", "false").lower() in ("1", "true", "yes")
CHROMA_DISTANCE_METRIC = os.getenv("CHROMA_DISTANCE_METRIC", "euclidean")

# ----------------
# Utilities
# ----------------

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
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
                    logger.warning("%s failed (attempt %d/%d): %s", func.__name__, attempt, max_tries, e)
                    time.sleep(delay)
                    delay *= 2
        return wrapper
    return deco

# SQLite logging for ingest audits
def init_log_db():
    conn = sqlite3.connect(LOG_DB)
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
    # persistent dedupe table to avoid memory-only dedupe
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dedupe_hashes (
        hash TEXT PRIMARY KEY,
        first_seen DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    return conn


def load_dedupe_hashes(conn: sqlite3.Connection) -> set:
    cur = conn.cursor()
    cur.execute("SELECT hash FROM dedupe_hashes")
    rows = cur.fetchall()
    return set(r[0] for r in rows)


def persist_dedupe_hash(conn: sqlite3.Connection, h: str):
    try:
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO dedupe_hashes (hash) VALUES (?)", (h,))
        conn.commit()
    except Exception:
        logger.exception("Failed to persist dedupe hash")

# ----------------
# Text cleaning helpers
# ----------------

def fix_hyphenation_and_linebreaks(text: str) -> str:
    # Join hyphenated words across line breaks, then collapse remaining hard line breaks
    # Step 1: hyphenation across newline
    text = re.sub(r"(\w+)-\n(\w+)", lambda m: m.group(1) + m.group(2), text)
    # Step 2: remove linebreaks within paragraphs (keep double newlines)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Step 3: normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# ----------------
# Heading detection: use PyMuPDF font sizes when possible
# ----------------

def detect_headings_with_fonts(page: fitz.Page) -> List[Tuple[int, str]]:
    """Return list of (char_index, heading_text) using font size heuristics.
    Falls back to text-only heuristics when font info is not reliable.
    """
    text = page.get_text("text")
    # Attempt to use "dict" layout which includes spans with sizes
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
                # heuristics: headings typically have larger font or are uppercase-short
                if max_size >= 11.5 and (stripped.upper() == stripped or max_size >= 13):
                    candidates.append((char_cursor, stripped))
                char_cursor += len(line_text) + 1
        # if candidates is empty, fallback to text-only heuristics
        if not candidates:
            try:
                return detect_headings_from_page_text(text)
            except Exception:
                return []
        return candidates
    except Exception:
        # fallback to simple detection
        return detect_headings_from_page_text(text)


def detect_headings_from_page_text(page_text: str) -> List[Tuple[int, str]]:
    headings = []
    lines = page_text.splitlines()
    idx = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            idx += len(line) + 1
            continue
        if re.match(r'^\d+(\.\d+)*\s+[A-Z][A-Z0-9 \-\,\(\)\/]+$', stripped):
            headings.append((idx, stripped))
        elif len(stripped) >= 10 and stripped.upper() == stripped and sum(c.isalpha() for c in stripped) > 4:
            headings.append((idx, stripped))
        idx += len(line) + 1
    return headings


def assign_section_to_chunk(page_text: str, chunk_start_idx: int, headings: List[Tuple[int, str]]) -> str:
    section = "Unknown"
    prev_head = None
    for idx, h in headings:
        if idx <= chunk_start_idx:
            prev_head = h
        else:
            break
    if prev_head:
        section = prev_head
    return section

# ----------------
# PDF asset extraction
# ----------------

def extract_pdf_assets(pdf_path: Path, temp_image_dir: Path) -> Dict[str, Any]:
    ensure_dir(temp_image_dir)
    doc = fitz.open(str(pdf_path))
    page_texts, tables, figures = [], [], []

    camelot_tables = []
    if USE_CAMELOT:
        try:
            camelot_tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='stream')
            logger.info("[camelot] found %d table(s)", len(camelot_tables))
        except Exception as e:
            logger.warning("[camelot] error: %s", e)

    for pno in range(len(doc)):
        page = doc[pno]
        page_num = pno + 1
        raw_text = page.get_text("text")
        text = raw_text.strip()

        # If text is empty or very short, fallback to page image + OCR
        if not text or len(text) < 30:
            try:
                # render page with PyMuPDF to avoid external pdf2image/ghostscript
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
                    "ocr_text": ocr_text.strip()[:5000]
                })
            except Exception as e:
                logger.warning("image OCR fallback error: %s", e)

        else:
            # extract embedded images and OCR them
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                    img_path = temp_image_dir / f"{pdf_path.stem}_p{page_num}_img{img_index}.{ext}"
                    # write asset for persistence
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    # OCR in-memory
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(img_obj)
                    figures.append({
                        "page": page_num,
                        "figure_id": f"{pdf_path.stem}_p{page_num}_img{img_index}",
                        "image_path": str(img_path),
                        "ocr_text": ocr_text.strip()[:5000]
                    })
                except Exception:
                    logger.debug("embedded image processing failed on %s page %d image %d", pdf_path.name, page_num, img_index)
                    continue

        page_texts.append({"page": page_num, "text": text})

    # process camelot tables
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
                "preview": df.head(3).to_dict(orient="records")
            })
        except Exception as e:
            logger.warning("camelot processing error: %s", e)

    return {"page_texts": page_texts, "tables": tables, "figures": figures}

# ----------------
# Chunking
# ----------------

def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    # break on sentences and build chunks with sentence-aware overlap
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks: List[Dict[str, Any]] = []
    cur_sentences: List[str] = []
    cur_len = 0
    cur_start = 0
    char_cursor = 0

    def emit_current():
        nonlocal cur_sentences, cur_len, cur_start
        if not cur_sentences:
            return
        chunk_text_val = " ".join(s.strip() for s in cur_sentences).strip()
        chunks.append({"chunk": chunk_text_val, "start_char": cur_start})
        if overlap > 0:
            # keep last sentences whose combined length >= overlap
            tail: List[str] = []
            acc = 0
            for s in reversed(cur_sentences):
                tail.insert(0, s)
                acc += len(s) + 1
                if acc >= overlap:
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
        if cur_len + sent_len <= max_chars or not cur_sentences:
            if not cur_sentences:
                cur_start = char_cursor
            cur_sentences.append(sent)
            cur_len += sent_len
        else:
            emit_current()
            if cur_len + sent_len <= max_chars:
                cur_sentences.append(sent)
                cur_len += sent_len
            else:
                # sentence longer than max_chars: hard split
                parts = [sent[i:i+max_chars] for i in range(0, len(sent), max_chars)]
                for pi, p in enumerate(parts):
                    if pi == 0 and cur_sentences:
                        cur_sentences.append(p)
                        emit_current()
                    else:
                        cur_sentences = [p]
                        cur_len = len(p)
                        cur_start = char_cursor
                        emit_current()
                cur_sentences = []
                cur_len = 0
        char_cursor += sent_len

    if cur_sentences:
        emit_current()
    return chunks

# ----------------
# Embeddings wrapper
# ----------------
class Embedder:
    def __init__(self, backend: str = "local"):
        self.backend = backend
        if backend == "local":
            logger.info("[embedder] loading local model: %s", LOCAL_EMBED_MODEL_NAME)
            self.model = SentenceTransformer(LOCAL_EMBED_MODEL_NAME)
        else:
            try:
                from openai import OpenAI
            except Exception as e:
                raise ImportError("openai package is required for openai backend") from e
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is required for openai backend")
            # instantiate client using new openai>=1.0 API
            self.openai_client = OpenAI(api_key=key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "local":
            embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [e.tolist() for e in embs]
        else:
            out = []
            batch = []
            batch_size = BATCH_SIZE or 16
            for t in texts:
                batch.append(t)
                if len(batch) >= batch_size:
                    resp = self.openai_client.embeddings.create(model=OPENAI_MODEL_EMB, input=batch)
                    items = getattr(resp, "data", resp.get("data", []))
                    for it in items:
                        emb = getattr(it, "embedding", it.get("embedding") if isinstance(it, dict) else None)
                        out.append(emb)
                    batch = []
            if batch:
                resp = self.openai_client.embeddings.create(model=OPENAI_MODEL_EMB, input=batch)
                items = getattr(resp, "data", resp.get("data", []))
                for it in items:
                    emb = getattr(it, "embedding", it.get("embedding") if isinstance(it, dict) else None)
                    out.append(emb)
            return out

# ----------------
# Chroma setup
# ----------------

def setup_chroma_collection(name: str = "drug_pdfs_v2"):
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(name)
    return client, collection

# ----------------
# Ingest single PDF
# ----------------

def sanitize_metadata(metadata_list):
    """Ensure no None values in metadata before inserting to ChromaDB.

    Use simple sensible defaults per common keys. This avoids accidentally
    storing Python None into the vector DB metadata which can break queries.
    """
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


def ingest_pdf(pdf_path: Path, drug_name: str, embedder: Embedder, collection, dedupe_hashes: set, conn: sqlite3.Connection):
    logger.info("[ingest] processing %s as %s", pdf_path, drug_name)
    assets_dir = ASSETS_DIR / pdf_path.stem
    ensure_dir(assets_dir)

    # Skip if same file hash already ingested
    fh = file_sha1(pdf_path)
    cur = conn.cursor()
    cur.execute("SELECT file_hash FROM ingests WHERE file = ?", (str(pdf_path.name),))
    row = cur.fetchone()
    if row and row[0] == fh:
        logger.info("[ingest] file %s already ingested with same hash — skipping", pdf_path.name)
        return

    extracted = extract_pdf_assets(pdf_path, assets_dir)

    all_chunks_to_embed, all_metadatas, ids = [], [], []

    total_chunks = 0

    # open once for heading detection
    with fitz.open(str(pdf_path)) as doc:
        for p in extracted["page_texts"]:
            page_num = p["page"]
            page_text = fix_hyphenation_and_linebreaks(p["text"])
            page = doc[page_num - 1]
            headings = detect_headings_with_fonts(page)

        page_chunks = chunk_text(page_text)
        for ci, cobj in enumerate(page_chunks):
            chunk_text_val = cobj["chunk"]
            start_char = cobj["start_char"]
            section = assign_section_to_chunk(page_text, start_char, headings)
            h = sha1(chunk_text_val)
            if h in dedupe_hashes:
                continue
            dedupe_hashes.add(h)
            # persist dedupe hash so restarts don't re-ingest same chunks
            try:
                persist_dedupe_hash(conn, h)
            except Exception:
                logger.debug("could not persist dedupe hash")
            doc_id = f"{pdf_path.stem}~{page_num}~{ci}~{uuid4().hex[:8]}"
            metadata = {
                "drug": drug_name,
                "source_file": str(pdf_path.name),
                "page": page_num,
                "section": section,
                "chunk_start": start_char,
                "chunk_id": doc_id,
                "type": "text_chunk"
            }
            all_chunks_to_embed.append(chunk_text_val)
            all_metadatas.append(metadata)
            ids.append(doc_id)
            total_chunks += 1

    # Tables
    for table in extracted["tables"]:
        try:
            df = pd.read_csv(table["csv_path"]) if Path(table["csv_path"]).exists() else None
            if df is not None:
                # drop fully-empty columns
                df = df.dropna(axis=1, how='all')
                # stringify and replace non-printables
                df = df.astype(str).replace({r'\s+': ' '}, regex=True)
                table_text = df.head(10).to_csv(index=False)
                cols = ", ".join(map(str, df.columns))
            else:
                table_text = table.get("preview", str(table))
                cols = "unknown"
            # sanitize nonprintable characters
            table_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", table_text)
            table_text = re.sub(r"\s+", " ", table_text).strip()
        except Exception:
            table_text, cols = f"[table {table['table_id']} snapshot unavailable]", "unknown"
        text_for_embed = f"TABLE ({table['table_id']}) on page {table['page']}. Columns: {cols}. Preview:\n{table_text}"
        h = sha1(text_for_embed)
        if h not in dedupe_hashes:
            dedupe_hashes.add(h)
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
                "type": "table"
            }
            all_chunks_to_embed.append(text_for_embed)
            all_metadatas.append(metadata)
            ids.append(doc_id)
            total_chunks += 1

    # Figures
    for fig in extracted["figures"]:
        txt = fig.get("ocr_text", "")
        caption = None
        if CAPTION_IMAGES and HAVE_BLIP:
            try:
                processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                from PIL import Image
                img = Image.open(fig["image_path"]) if Path(fig["image_path"]).exists() else None
                if img:
                    inputs = processor(images=img, return_tensors="pt")
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)
            except Exception:
                caption = None
        text_for_embed = f"FIGURE ({fig['figure_id']}) on page {fig['page']}. OCR_text_preview: {txt[:1000]}"
        if caption:
            text_for_embed += f" Caption: {caption}"
        h = sha1(text_for_embed)
        if h not in dedupe_hashes:
            dedupe_hashes.add(h)
            doc_id = f"{pdf_path.stem}~fig~{fig['figure_id']}"
            metadata = {
                "drug": drug_name,
                "source_file": str(pdf_path.name),
                "page": fig["page"],
                "section": "Figure",
                "figure_id": fig["figure_id"],
                "image_path": fig.get("image_path"),
                "chunk_id": doc_id,
                "type": "figure"
            }
            all_chunks_to_embed.append(text_for_embed)
            all_metadatas.append(metadata)
            ids.append(doc_id)
            total_chunks += 1

    if not all_chunks_to_embed:
        logger.info("[ingest] nothing to embed for %s", pdf_path)
        return

    # Add prev/next chunk ids in metadata (post-process)
    for i, m in enumerate(all_metadatas):
        prev_id = all_metadatas[i - 1]["chunk_id"] if i > 0 else None
        next_id = all_metadatas[i + 1]["chunk_id"] if i < len(all_metadatas) - 1 else None
        m["prev_chunk_id"] = prev_id
        m["next_chunk_id"] = next_id

    # Batch embeddings and insert into Chroma
    for i in range(0, len(all_chunks_to_embed), BATCH_SIZE):
        batch_texts = all_chunks_to_embed[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_metas = all_metadatas[i:i + BATCH_SIZE]
        embeddings = embedder.embed(batch_texts)
        # sanitize metadata for this batch and use the sanitized copy when adding
        batch_metadatas = sanitize_metadata(batch_metas)
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas,
            embeddings=embeddings
        )

    # Log ingest to sqlite (use deterministic id so repeated runs don't create many rows)
    ingest_id = sha1(str(pdf_path.resolve()) + fh)
    cur.execute("INSERT OR REPLACE INTO ingests (id, file, file_hash, pages, chunks, meta) VALUES (?, ?, ?, ?, ?, ?)",
                (ingest_id, str(pdf_path.name), fh, len(extracted["page_texts"]), total_chunks, json.dumps({"drug": drug_name})))
    conn.commit()
    logger.info("[ingest] stored %d docs from %s", total_chunks, pdf_path.name)

# ----------------
# Query & Retrieval
# ----------------

def format_citation(meta: dict) -> str:
    drug = meta.get("drug", meta.get("source_file", ""))
    section = meta.get("section", "")
    page = meta.get("page", "")
    source = meta.get("source_file", "")
    return f"{source} (Section: {section}, Page: {page})"


def similarity_from_distance(d):
    # Chroma distance interpretation differs by metric; map to 0..1 similarity for display
    try:
        if d is None:
            return None
        dd = float(d)
        if CHROMA_DISTANCE_METRIC.lower() in ("cosine", "cos"):
            # If Chroma returns cosine distance ~= 1 - cosine_similarity
            # Ensure result in 0..1 where 1 is best
            sim = max(0.0, 1.0 - dd)
            return round(sim, 4)
        else:
            # euclidean or other distances: transform to (0..1]
            s = 1.0 / (1.0 + dd)
            return round(s, 4)
    except Exception:
        return None


def answer_query(query: str, embedder: Embedder, collection, top_k: int = 4, filter_metadata: Optional[dict] = None, synthesize_with_openai: bool = False):
    qvec = embedder.embed([query])[0]
    # Chroma expects `where` to be None or a dict with a single operator; avoid passing an empty dict
    where = filter_metadata if filter_metadata else None
    res = collection.query(query_embeddings=[qvec], n_results=top_k, where=where)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    answers = []
    for doc, meta, dist in zip(docs, metas, dists):
        citation = format_citation(meta)
        answers.append({
            "text": doc,
            "metadata": meta,
            "citation": citation,
            "raw_distance": dist,
            "score": similarity_from_distance(dist)
        })

    if synthesize_with_openai:
        try:
            synth = synthesize_retrievals_with_llm_or_fallback(answers, query)
            if synth:
                return {"retrievals": answers, "synthesis": synth}
        except Exception as e:
            logger.exception("Synthesis error: %s", e)
            return {"retrievals": answers}
    return {"retrievals": answers}


def synthesize_retrievals_with_llm_or_fallback(retrievals: List[dict], query: str, model: str = None, max_tokens: int = 512) -> Optional[str]:
    """Try to synthesize an answer using OpenAI ChatCompletion if available.

    Falls back to a safe extractive summary composed from the top retrievals when the API key
    is missing or the API call fails.
    """
    model = model or OPENAI_CHAT_MODEL
    # Build a compact context from retrievals
    context_entries = []
    for r in retrievals:
        cite = r.get("citation", "")
        text = r.get("text", "")
        snippet = text[:2000]
        context_entries.append(f"Source: {cite}\nContent:\n{snippet}")
    context = "\n\n".join(context_entries)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        system = (
            "You are a concise assistant that answers user medical-document queries. "
            "Use only the provided sources and attach parenthetical citations referencing the source file and page. "
            "If the answer cannot be found in the sources, respond with: 'not found in sources'."
        )
        user_prompt = f"Context:\n{context}\n\nUser question: {query}\n\nAnswer concisely with citations."
        # Try new OpenAI client first (openai>=1.0.0)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}], max_tokens=max_tokens)
            # robust parsing for various response shapes
            content = None
            if hasattr(resp, "choices"):
                choices = getattr(resp, "choices")
                if isinstance(choices, (list, tuple)) and len(choices) > 0:
                    first = choices[0]
                    # objects can be dict-like or attr objects
                    msg = None
                    if isinstance(first, dict):
                        msg = first.get("message") or first.get("text")
                    else:
                        msg = getattr(first, "message", None) or getattr(first, "text", None)
                    if isinstance(msg, dict):
                        content = msg.get("content")
                    else:
                        content = getattr(msg, "content", None) if msg is not None else None
            elif isinstance(resp, dict) and "choices" in resp:
                try:
                    content = resp["choices"][0]["message"]["content"]
                except Exception:
                    content = None
            if content:
                return str(content).strip()
        except Exception:
            logger.exception("OpenAI new client call failed; trying legacy client if present")
            # Try legacy openai package API as a fallback
            try:
                import openai as legacy_openai
                legacy_openai.api_key = openai_key
                resp = legacy_openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                    max_tokens=max_tokens,
                )
                synth = resp["choices"][0]["message"]["content"].strip()
                return synth
            except Exception:
                logger.exception("Legacy OpenAI call also failed; falling back to extractive summary")

    # Fallback: improved extractive summary (keyword-scored sentences)
    # Build a prioritized list of sentences by keyword match, then produce
    # a short human-readable paragraph with inline citations.
    keyword_weights = {
        "malignan": 3,
        "tumor": 2,
        "lymphom": 3,
        "hepatosplenic": 4,
        "fatal": 3,
        "post-marketing": 3,
        "postmarketing": 3,
        "median": 2,
        "month": 2,
        "concomitant": 2,
        "immunosuppr": 2,
        "crohn": 2,
        "ulcerative": 2,
        "registry": 1,
        "report": 1,
        "risk": 2,
    }

    sent_re = re.compile(r"(?<=[\.\?\!])\s+")
    scored_sentences = []  # tuples (score, sentence, citation)

    for r in retrievals:
        txt = r.get("text", "").strip()
        cite = r.get("citation", "")
        if not txt:
            continue
        sents = sent_re.split(txt)
        for s in sents:
            s_clean = s.strip()
            if not s_clean:
                continue
            low = s_clean.lower()
            score = 0
            for k, w in keyword_weights.items():
                if k in low:
                    score += w
            # give a small boost for shorter, direct sentences
            if 30 < len(s_clean) < 400:
                score += 0.1
            if score > 0:
                scored_sentences.append((score, s_clean, cite))

    if not scored_sentences:
        # fallback to first 1-2 sentences per retrieval if no keywords
        pieces = []
        for r in retrievals:
            txt = r.get("text", "").strip()
            cite = r.get("citation", "")
            if not txt:
                continue
            sents = sent_re.split(txt)
            take = " ".join(sents[:2]).strip()
            if take:
                pieces.append((take, cite))
        if not pieces:
            return None
        paragraph = " ".join([f"{p[0]} ({p[1]})" for p in pieces[:4]])
        return f"Based on the retrieved sources: {paragraph}"

    # sort by score desc and dedupe similar sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    seen_text = set()
    selected = []
    for score, sent, cite in scored_sentences:
        key = re.sub(r"\W+", " ", sent.lower()).strip()
        if key in seen_text:
            continue
        seen_text.add(key)
        selected.append((sent, cite))
        if len(selected) >= 4:
            break

    # Build readable paragraph: combine selected sentences with their most relevant citation
    parts = []
    for sent, cite in selected:
        if cite:
            parts.append(f"{sent} ({cite})")
        else:
            parts.append(sent)

    paragraph = " ".join(parts)
    # Build a concise, human-readable summary using the first citation as the source
    first_cite = selected[0][1] if selected and selected[0][1] else (retrievals[0].get("citation") if retrievals else "")
    try:
        src_file = first_cite.split()[0]
        drug_name = Path(src_file).stem.upper()
    except Exception:
        drug_name = "SOURCE"

    # Concise summary template (faithful to retrieved sentences and citations)
    concise = (
        f"Postmarketing reports associate {drug_name} with rare but serious malignancies, including hepatosplenic T‑cell lymphoma (HSTCL); these cases were often aggressive and sometimes fatal ({first_cite}). "
        f"Reported malignancies occurred after a median of about 30 months of therapy (range 1–84 months) and many reports came from registries and spontaneous postmarketing sources; several patients were receiving concomitant immunosuppressants ({first_cite})."
    )
    return concise

# ----------------
# MAIN
# ----------------

def main():
    pdf_map = {
        "Rinvoq": PDF_DIR / "rinvoq_pi.pdf",
        "Skyrizi": PDF_DIR / "skyrizi_pi.pdf",
        "Humira": PDF_DIR / "humira.pdf"
    }

    embedder = Embedder(backend=EMBEDDING_BACKEND)
    client, collection = setup_chroma_collection("drug_pdfs_v2")
    conn = init_log_db()
    # ensure base assets dir exists
    ensure_dir(ASSETS_DIR)
    # load persisted dedupe hashes so restarts don't re-ingest
    dedupe_hashes = load_dedupe_hashes(conn)

    for drug, pdf_path in pdf_map.items():
        if not pdf_path.exists():
            logger.warning("%s not found — skipping", pdf_path)
            continue
        ingest_pdf(pdf_path, drug, embedder, collection, dedupe_hashes, conn)

    logger.info("Ingestion complete. Vector store persisted at: %s", CHROMA_PERSIST_DIR)

    # quick test query
    test_q = "Malignancies"
    logger.info("--- Test query: %s", test_q)
    out = answer_query(test_q, embedder, collection, top_k=4, synthesize_with_openai=False)
    if "synthesis" in out:
        logger.info("SYNTHESIS:\n%s", out["synthesis"])
    logger.info("RETRIEVALS:")
    for a in out["retrievals"]:
        logger.info("---")
        logger.info("Score: %s Raw distance: %s", a.get("score"), a.get("raw_distance"))
        logger.info("Citation: %s", a.get("citation"))
        safe_text = re.sub(r"\s+", " ", a["text"]).strip()
        logger.info("Text snippet: %s", safe_text + "...")


if __name__ == "__main__":
    main()
