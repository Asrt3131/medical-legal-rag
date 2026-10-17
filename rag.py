import os
import re
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K


# ============================================================
# مدل امبدینگ — یک‌بار لود، کش بین اجراها
# ============================================================
@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# BM25 سبک (بدون نیاز به پکیج اضافی)
# ============================================================
def _tokenize(text: str) -> list[str]:
    """توکنایزر ساده برای فارسی/انگلیسی."""
    text = text.lower()
    # حذف کاراکترهای غیرحرفی، حفظ فاصله
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text)
    # حذف کلمات خیلی کوتاه و stopwordهای ساده
    stops = {"و", "از", "به", "که", "در", "را", "با", "این", "آن", "است", "the", "a", "an", "of", "to", "and", "in", "is"}
    return [t for t in tokens if len(t) > 1 and t not in stops]


def _bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """امتیاز BM25 ساده برای یک query روی چند سند."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return np.zeros(len(docs))

    doc_tokens = [_tokenize(d) for d in docs]
    N = len(docs)
    avg_dl = sum(len(d) for d in doc_tokens) / max(N, 1)

    # document frequency
    df = {}
    for d in doc_tokens:
        for t in set(d):
            df[t] = df.get(t, 0) + 1

    scores = np.zeros(N)
    for i, d in enumerate(doc_tokens):
        dl = len(d)
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        for t in q_tokens:
            if t not in tf:
                continue
            idf = np.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            num = tf[t] * (k1 + 1)
            denom = tf[t] + k1 * (1 - b + b * dl / max(avg_dl, 1e-9))
            scores[i] += idf * num / denom
    return scores


def _normalize(scores: np.ndarray) -> np.ndarray:
    """نرمال‌سازی min-max به بازه [0,1]."""
    if len(scores) == 0:
        return scores
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


# ============================================================
# ساخت ایندکس
# ============================================================
def build_index(pdf_path: str, file_name: str | None = None):
    """
    PDF را می‌خواند، chunk می‌کند، embed می‌کند و FAISS index می‌سازد.

    Returns:
        (faiss.IndexFlatIP, list[Document])
    """
    docs = PyPDFLoader(pdf_path).load()
    if not docs:
        raise ValueError("PDF خالی است یا قابل خواندن نیست")

    # نام فایل واقعی را در metadata بگذار (برای نمایش در منابع)
    src_name = file_name or os.path.basename(pdf_path)
    for d in docs:
        d.metadata["source"] = src_name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError("هیچ متنی از PDF استخراج نشد")

    texts = [c.page_content for c in chunks]
    model = get_model()
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    embs = np.asarray(embs, dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    return index, chunks


# ============================================================
# جستجو
# ============================================================
def search(
    index,
    chunks: list,
    query: str,
    top_k: int = TOP_K,
    min_score: float = 0.0,
    hybrid: bool = True,
):
    """
    بهترین chunkهای مرتبط با query را برمی‌گرداند.

    Args:
        index: FAISS index
        chunks: لیست Documentها
        query: سوال کاربر
        top_k: تعداد نتایج نهایی
        min_score: حداقل امتیاز (بعد از نرمال‌سازی)
        hybrid: اگر True، ترکیب امتیاز برداری و BM25

    Returns:
        لیست dict با کلیدهای text, page, file, score
    """
    if index is None or index.ntotal == 0 or not chunks:
        return []

    top_k = int(top_k)
    if top_k < 1:
        return []

    # --- جستجوی برداری ---
    model = get_model()
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # کمی بیشتر از top_k بگیر تا در حالت hybrid فضای انتخاب داشته باشیم
    fetch_k = min(max(top_k * 3, top_k), index.ntotal)
    D, I = index.search(q_vec, fetch_k)
    D, I = D[0], I[0]

    # --- جستجوی BM25 (اختیاری) ---
    if hybrid:
        candidate_texts = [chunks[i].page_content for i in I]
        bm25 = _bm25_scores(query, candidate_texts)
        bm25_norm = _normalize(bm25)
        vec_norm = _normalize(D)
        # ترکیب 60/40 (قابل تنظیم)
        combined = 0.6 * vec_norm + 0.4 * bm25_norm
        order = np.argsort(-combined)
    else:
        combined = D
        order = np.argsort(-D)

    # --- ساخت نتایج ---
    results = []
    for pos in order:
        idx = int(I[pos])
        score = float(combined[pos])
        if score < min_score:
            continue
        chunk = chunks[idx]
        # مسیر فایل را فقط نام نگه می‌داریم
        src = chunk.metadata.get("source", "?")
        src_name = os.path.basename(str(src).replace("\\", "/"))
        results.append({
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", "?"),
            "file": src_name,
            "score": score,
        })
        if len(results) >= top_k:
            break

    return results