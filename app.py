import os
import time
import json
import hashlib
import tempfile
import traceback
from pathlib import Path

import streamlit as st
import faiss
import numpy as np

from rag import build_index, search, get_model
from llm import generate_answer, is_llm_available
from config import TOP_K, CHUNK_SIZE

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="MediLegal RAG Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Styles
# ============================================================
st.markdown(
    """
<style>
    .stChatMessage { direction: rtl; text-align: right; }
    .stChatMessage .stMarkdown { direction: rtl; text-align: right; }

    .source-card {
        background: #f8f9fa; border-radius: 8px; padding: 12px;
        border-left: 3px solid #4a90e2; margin: 8px 0;
        direction: rtl; text-align: right;
    }
    .metric-badge {
        display: inline-block; padding: 2px 8px; border-radius: 12px;
        background: #e8f4fd; color: #1a5490; font-size: 0.8em;
    }
    .footer {
        text-align: center; padding: 20px 0; color: #6c757d;
        font-size: 0.85em; line-height: 1.7em;
    }
    .footer .name {
        color: #4a90e2; font-weight: 600;
    }
    .footer a {
        color: #4a90e2; text-decoration: none;
    }
    .footer a:hover { text-decoration: underline; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Domain configuration
# ============================================================
DOMAINS = {
    "🏥 Medical": {
        "system": (
            "You are a medical documents assistant. Answer in English, "
            "cite page numbers, and add a disclaimer that this is not medical advice."
        ),
        "emoji": "🏥",
        "color": "#4a90e2",
    },
    "⚖️ Legal": {
        "system": (
            "You are a legal documents assistant. Answer in English, "
            "cite articles/pages, and add a disclaimer that this is not legal advice."
        ),
        "emoji": "⚖️",
        "color": "#e2a04a",
    },
}

# ============================================================
# Session state
# ============================================================
def init_state():
    defaults = {
        "index": None,
        "chunks": [],
        "files_meta": [],
        "messages": [],
        "indexed_files": set(),
        "model_loaded": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()

LLM_AVAILABLE = is_llm_available()


# ============================================================
# Helpers
# ============================================================
def file_key(f):
    """Stable identity for a file: name + size + content hash."""
    data = f.getvalue()
    digest = hashlib.md5(data).hexdigest()
    return f"{f.name}|{len(data)}|{digest}"


def reset_index():
    st.session_state.index = None
    st.session_state.chunks = []


def rebuild_index():
    """
    Rebuild the FAISS index from files in session_state.files_meta.
    Uses st.session_state._file_bytes to reconstruct each uploaded file.
    If a file's raw bytes are gone, its entry is dropped.
    """
    reset_index()
    kept_meta = []
    all_chunks = []
    all_vectors = []

    for meta in st.session_state.files_meta:
        data = st.session_state.get("_file_bytes", {}).get(meta["hash"])
        if data is None:
            continue
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            idx, chunks = build_index(tmp_path, file_name=meta["name"])
            if idx is not None and chunks:
                all_vectors.append(idx.reconstruct_n(0, idx.ntotal))
                all_chunks.extend(chunks)
                kept_meta.append(meta)
        except Exception as e:
            st.warning(f"Could not re-index {meta['name']}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if all_vectors:
        merged = np.vstack(all_vectors).astype("float32")
        new_index = faiss.IndexFlatIP(merged.shape[1])
        new_index.add(merged)
        st.session_state.index = new_index
        st.session_state.chunks = all_chunks

    st.session_state.files_meta = kept_meta
    st.session_state.indexed_files = {
        f"{m['name']}|{m['size']}|{m['hash']}" for m in kept_meta
    }


def render_sources(sources, with_latency=False, latency=None):
    """Unified source rendering for both live answers and chat history."""
    header = f"📚 Sources ({len(sources)})"
    if with_latency and latency is not None:
        header += f" — ⏱️ {latency}s"
    with st.expander(header):
        for i, s in enumerate(sources, 1):
            score_pct = max(0, min(100, int(s["score"] * 100)))
            filled = score_pct // 10
            bar = "█" * filled + "░" * (10 - filled)
            file_name = s.get("file", "?")
            page = s.get("page", "?")
            st.markdown(
                f"**Source {i}** — 📄 `{file_name}` — page {page} "
                f"| `{bar}` {s['score']:.2f}"
            )
            text = s.get("text", "")
            st.caption(text[:400] + ("..." if len(text) > 400 else ""))
            st.divider()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("🩺📄 MediLegal RAG Pro")

    # --- LLM status ---
    if LLM_AVAILABLE:
        st.success("✅ LLM ready")
    else:
        st.warning("⚠️ API key not set — document browsing only")

    # --- Upload ---
    st.header("📂 Documents")
    uploaded = st.file_uploader(
        "Upload PDFs (multiple files allowed)",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploader",
    )

    if uploaded:
        st.session_state.setdefault("_file_bytes", {})
        for f in uploaded:
            key = file_key(f)
            h = key.split("|")[2]
            if h not in st.session_state["_file_bytes"]:
                st.session_state["_file_bytes"][h] = f.getvalue()

        new_files = [
            (f, file_key(f))
            for f in uploaded
            if file_key(f) not in st.session_state.indexed_files
        ]

        if new_files and st.button(
            f"🚀 Index {len(new_files)} new file(s)",
            type="primary",
            use_container_width=True,
        ):
            progress = st.progress(0, text="Preparing...")
            new_pairs, new_meta, errors = [], [], []
            total = len(new_files)

            for i, (f, key) in enumerate(new_files):
                progress.progress(
                    i / total,
                    text=f"[{i+1}/{total}] Indexing {f.name}...",
                )
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    idx, chunks = build_index(tmp_path, file_name=f.name)
                    if idx is None or not chunks:
                        raise ValueError("No text could be extracted from the PDF")
                    new_pairs.append((idx, chunks))
                    new_meta.append({
                        "name": f.name,
                        "chunks": len(chunks),
                        "pages": len({c.metadata.get("page", 0) for c in chunks}),
                        "size": len(f.getvalue()),
                        "hash": key.split("|")[2],
                    })
                    st.session_state.indexed_files.add(key)
                except Exception as e:
                    errors.append(f"{f.name}: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            if new_pairs:
                if st.session_state.index is not None and st.session_state.chunks:
                    existing_meta = st.session_state.files_meta
                    st.session_state.files_meta = existing_meta + new_meta
                    rebuild_index()
                else:
                    all_vectors = [idx.reconstruct_n(0, idx.ntotal) for idx, _ in new_pairs]
                    merged = np.vstack(all_vectors).astype("float32")
                    new_index = faiss.IndexFlatIP(merged.shape[1])
                    new_index.add(merged)
                    st.session_state.index = new_index
                    st.session_state.chunks = [
                        c for _, chunks in new_pairs for c in chunks
                    ]
                    st.session_state.files_meta.extend(new_meta)

            progress.progress(1.0, text="Done")
            if errors:
                st.error("Errors in some files:\n" + "\n".join(errors))
            if new_pairs:
                st.toast(f"✅ {len(new_pairs)} file(s) indexed", icon="🎉")
            time.sleep(0.3)
            st.rerun()

    # --- Document list ---
    if st.session_state.files_meta:
        st.divider()
        st.subheader("📋 Indexed documents")
        to_delete = None
        for i, meta in enumerate(st.session_state.files_meta):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(
                    f"📄 **{meta['name']}** — {meta['chunks']} chunk(s) "
                    f"— {meta['pages']} page(s)"
                )
            with col2:
                if st.button("✕", key=f"del_{i}", help="Delete this document"):
                    to_delete = i
        if to_delete is not None:
            removed = st.session_state.files_meta.pop(to_delete)
            st.session_state["_file_bytes"].pop(removed["hash"], None)
            rebuild_index()
            st.rerun()

        st.metric("Total chunks", len(st.session_state.chunks))
        if st.button("🗑️ Delete all documents", use_container_width=True):
            st.session_state.files_meta = []
            st.session_state.indexed_files = set()
            st.session_state["_file_bytes"] = {}
            reset_index()
            st.rerun()

    # --- Settings ---
    st.divider()
    st.header("⚙️ Search settings")
    domain = st.selectbox("Specialty mode", list(DOMAINS.keys()))
    top_k = st.slider("Number of sources (top-k)", 1, 10, TOP_K)
    min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.35, 0.05)
    use_hybrid = st.toggle("Hybrid search (BioBERT + BM25)", value=True)

    # --- Chat ---
    st.divider()
    st.subheader("💬 Chat")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        transcript_txt = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}"
            for m in st.session_state.messages
        )
        transcript_json = json.dumps(
            st.session_state.messages, ensure_ascii=False, indent=2
        )
        transcript_md = "\n\n".join(
            f"### {m['role'].capitalize()}\n\n{m['content']}"
            for m in st.session_state.messages
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ TXT", transcript_txt, "chat.txt",
                use_container_width=True, key="dl_txt",
            )
        with c2:
            st.download_button(
                "⬇️ JSON", transcript_json, "chat.json",
                use_container_width=True, key="dl_json",
            )
        with c3:
            st.download_button(
                "⬇️ MD", transcript_md, "chat.md",
                use_container_width=True, key="dl_md",
            )

# ============================================================
# Main
# ============================================================
st.title(f"{DOMAINS[domain]['emoji']} Medical/Legal Documents Assistant")
st.caption(
    f"Specialized RAG with BioBERT — "
    f"{len(st.session_state.chunks)} chunk(s) from "
    f"{len(st.session_state.files_meta)} document(s)"
)

# --- Render chat history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            render_sources(
                m["sources"],
                with_latency=True,
                latency=m.get("latency"),
            )

# ============================================================
# User input
# ============================================================
q_raw = st.chat_input("Write your question...")
if q_raw is not None:
    q = q_raw.strip()
    if not q:
        st.warning("⚠️ Please enter a non-empty question.")
        st.stop()

    if st.session_state.index is None:
        st.warning("⚠️ First upload and index a document.")
        st.stop()

    if not LLM_AVAILABLE:
        st.warning(
            "⚠️ LLM is not configured. Set OPENAI_API_KEY in Streamlit Secrets "
            "to enable question answering."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    t0 = time.time()
    with st.chat_message("assistant"):
        try:
            with st.status("🔍 Searching the documents...", expanded=True) as status:
                st.write(f"Searching in {len(st.session_state.chunks)} chunk(s)...")
                t_retrieval = time.time()
                sources = search(
                    st.session_state.index,
                    st.session_state.chunks,
                    q,
                    top_k=top_k,
                    min_score=min_score,
                    hybrid=use_hybrid,
                )
                sources = [s for s in sources if s["score"] >= min_score]
                retrieval_time = round(time.time() - t_retrieval, 2)

                if not sources:
                    status.update(
                        label="❌ No relevant source found",
                        state="error",
                    )
                    msg = {
                        "role": "assistant",
                        "content": (
                            "I couldn't find an answer in the current documents. "
                            "Try rephrasing your question or uploading more documents."
                        ),
                        "sources": [],
                        "latency": round(time.time() - t0, 1),
                    }
                    st.session_state.messages.append(msg)
                    st.markdown(msg["content"])
                    st.stop()

                st.write(
                    f"✅ {len(sources)} relevant source(s) — generating answer..."
                )
                answer = generate_answer(
                    q,
                    sources,
                    system_prompt=DOMAINS[domain]["system"],
                )
                latency = round(time.time() - t0, 1)
                status.update(
                    label=(
                        f"✅ Answer ready in {latency}s "
                        f"(retrieval: {retrieval_time}s)"
                    ),
                    state="complete",
                )

            st.markdown(answer)
            render_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "latency": latency,
            })

        except Exception as e:
            st.error(f"❌ Error: {type(e).__name__}: {e}")
            if (
                st.session_state.messages
                and st.session_state.messages[-1]["role"] == "user"
            ):
                st.session_state.messages.pop()
            with st.expander("🔍 Error details"):
                st.code(traceback.format_exc(), language="python")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Built by <span class="name">A_srt343</span><br>
        Instagram: <a href="https://instagram.com/project_srt343" target="_blank">@project_srt343</a>
    </div>
    """,
    unsafe_allow_html=True,
)
