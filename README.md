# 🩺📄 Medical/Legal Document RAG Assistant

A domain-specific Retrieval-Augmented Generation (RAG) system for medical and legal documents, powered by specialized embeddings (**BioBERT** for medical text, **Legal-BERT** for legal text).

![Main UI](docs/screenshots/main-ui.png)
*Streamlit interface — document upload and Q&A*

> ⚠️ **Status:** The external API (OpenAI) is **not configured** in the current version.
> The application runs fully and without errors. The following features work as expected:
> - ✅ PDF upload and document indexing
> - ✅ Semantic search using BioBERT
> - ✅ Source retrieval with page numbers
> - ✅ Complete and interactive user interface
>
> To enable LLM-generated answers, simply add your API key to the `.env` file.

---

## 🚀 Live Demo

👉 **Try the app:** [medical-legal-rag.streamlit.app](https://medical-legal-rag-f8whusetlraqtgxt8lwiam.streamlit.app/)

---

## 📖 Overview

Medical and legal documents are dense, terminology-heavy, and unforgiving to generic search. A general-purpose embedding model will often confuse acronyms like "MI" (Myocardial Infarction vs. Michigan) or miss subtle legal phrasing. This project solves that by building a **domain-specific RAG pipeline** that:

- Loads and chunks PDF documents intelligently
- Uses **BioBERT** (or Legal-BERT) for domain-aware embeddings
- Stores vectors in **FAISS** for fast similarity search
- Retrieves the top-K most relevant chunks for any query
- Returns answers with **page-level source attribution**

---

## ✨ Features

- ✅ **Domain-specific embeddings** — BioBERT for medical, Legal-BERT for legal
- ✅ **PDF parsing** with `PyPDFLoader`
- ✅ **Smart chunking** — 500-char chunks with 50-char overlap
- ✅ **FAISS vector search** for sub-second retrieval
- ✅ **Source attribution** — every answer cites page numbers
- ✅ **Interactive Streamlit UI** — upload, ask, get answers
- ✅ **Configurable** — swap embedding models via `.env`

---

## 🧠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Embeddings** | BioBERT (Sentence-Transformers) |
| **Vector Search** | FAISS |
| **PDF Parsing** | LangChain (`PyPDFLoader`) |
| **Interface** | Streamlit |
| **LLM (optional)** | OpenAI GPT |
| **Config** | python-dotenv |

---

## 📂 Dataset

No fixed dataset — this is a **document-agnostic** system. You can upload:

- Medical: clinical guidelines, research papers, drug references
- Legal: contracts, case law, statutes, compliance documents

---

## 📊 Results

The system is evaluated on:

- **Retrieval quality** — top-K relevance (by embedding similarity score)
- **Answer faithfulness** — citations matched to the correct page
- **Latency** — typically <1s retrieval + 2-4s LLM generation

Without an LLM key, retrieval still works fully — you'll just see retrieved sources instead of generated text.

---

## 📁 Project Structure

```
medical-legal-rag/
│
├── app.py                  # Streamlit web application
├── rag.py                  # RAG logic (indexing + retrieval)
├── llm.py                  # LLM integration (OpenAI)
├── config.py               # Configuration and .env loader
├── requirements.txt        # Dependencies
├── README.md               # This file
├── .env.example            # Environment variables template
├── .gitignore
│
├── docs/
│   └── screenshots/
│       ├── main-ui.png
│       └── qa-result.png
│
├── data/
│   └── sample.pdf          # Sample test PDF
│
└── tests/
    └── test_rag.py         # Unit tests
```

---

## ⚙️ How to Run Locally

### Prerequisites

- Python 3.9 or higher
- Git
- ~2 GB free disk space (for the BioBERT model)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Asrt3131/medical-legal-rag.git
cd medical-legal-rag

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy the environment template
# Windows:
copy .env.example .env
# Linux / macOS:
cp .env.example .env

# 6. Run the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🔬 How It Works

### 1. Indexing
When you upload a PDF and click **Index**:
- `PyPDFLoader` reads the PDF
- `RecursiveCharacterTextSplitter` splits it into ~500-char chunks
- Each chunk is embedded with BioBERT
- Embeddings are stored in a FAISS index

### 2. Retrieval
When you ask a question:
- The query is embedded with the same model
- FAISS returns the top-3 most similar chunks
- Page numbers and similarity scores are extracted

### 3. Generation (optional)
If `OPENAI_API_KEY` is set:
- Retrieved chunks are inserted into the prompt
- GPT generates an answer with source citations
- Without a key, only the retrieved sources are shown

---

## 🧪 Testing

```bash
python tests/test_rag.py
```

Expected output:

```
✅ All tests passed
```

---

## 🛠️ Troubleshooting

| Error | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'config'` | Ensure `config.py` is in the same folder as `rag.py` |
| `No module named 'sentence_transformers'` | Run `pip install -r requirements.txt` again |
| `ImportError: cannot import name 'build_index'` | Rename `rag.py` → `my_rag.py` if a package conflict exists |
| `Could not resolve host: github.com` | Network issue — run `ping github.com` |
| Model fails to download | Check connection to `huggingface.co`; use a VPN if needed |
| `FileNotFoundError: temp.pdf` | Ensure the upload succeeded before clicking Index |

---

## 🚀 Deployment (Streamlit Cloud)

1. Push the project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repository
4. Set **Main file path** to `app.py`
5. Add your `OPENAI_API_KEY` under **Secrets**
6. Click **Deploy**

---

## 🔐 Security Notes

- The `.env` file is listed in `.gitignore` — **never commit API keys**
- All processing runs locally; only if you enable the OpenAI key will document text be sent externally
- No user data is logged or stored by the app

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is **not a substitute** for professional medical diagnosis, legal advice, or any specialized decision-making. Always consult a licensed professional before acting on information retrieved by this system. The author is not responsible for any consequences of using this tool in real-world decisions.

---

## 📜 License

This project is released under the **MIT License**. Free to use, modify, and distribute.

---

## 🔗 Links

- **GitHub:** [github.com/Asrt3131/medical-legal-rag](https://github.com/Asrt3131/medical-legal-rag)
- **Live Demo:** [medical-legal-rag.streamlit.app](https://medical-legal-rag-f8whusetlraqtgxt8lwiam.streamlit.app/)
- **Instagram:** [@project_srt343](https://instagram.com/project_srt343)

---

<p align="center">
  <b>Built with ❤️ by A_srt343</b><br>
  <a href="https://instagram.com/project_srt343">Instagram: @project_srt343</a>
</p>
