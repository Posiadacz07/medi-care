# ---------------------------------------------------------------------------
# MediCare Local — on-device clinical intake prototype (Apple Silicon)
#
# Patient audio, transcripts, and retrieval stay on this machine.
# The first launch needs internet once, to download Hugging Face knowledge
# and the Whisper weights. After that, inference is local via Ollama.
#
# Required pip packages (including datasets):
#
#   pip install streamlit chromadb langchain langchain-community \
#       langchain-core langchain-text-splitters pypdf faster-whisper datasets
#
# System tools (not pip):
#   - Ollama: https://ollama.com
#       ollama pull medgemma
#       ollama pull nomic-embed-text
#   - ffmpeg is recommended so faster-whisper can decode microphone audio
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.config import Settings
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bars
from faster_whisper import WhisperModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
disable_progress_bars()

logger = logging.getLogger("medicare")

# --- Local models and paths -------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "medgemma"
EMBED_MODEL = "nomic-embed-text"
WHISPER_MODEL_SIZE = "base"  # "small" is more accurate and still fine on 36GB RAM
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

KNOWLEDGE_BASE_DIR = Path("./knowledge_base")
PERSIST_DIR = Path("./chroma_db")
STATS_PATH = Path("./ingestion_stats.json")
REBUILD_FLAG = Path("./.rebuild_requested")
COLLECTION_NAME = "medi_care_knowledge"

# Caps the rows embedded from each Hugging Face repo so the first local
# index build finishes on a laptop. Raise these once Ollama embeddings
# are warmed up; the loader still iterates row by row.
MAX_ROWS_PER_HF_SOURCE = 200
PDF_CHUNK_SIZE = 1000
PDF_CHUNK_OVERLAP = 150
RETRIEVAL_K = 4
EMBED_BATCH_SIZE = 32

CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

HF_SOURCES = (
    {
        "repo": "proadhikary/MENST",
        "config": None,
        "data_files": "training2K.csv",
        "description": "Menstrual health question-answer pairs",
    },
    {
        "repo": "parissharpe/naos-nutrition-training-pairs",
        "config": None,
        "data_files": "naos-training-pairs-v2-1-flat.jsonl",
        "description": "Evidence-based nutrition training pairs",
    },
    {
        "repo": "lavita/medical-qa-datasets",
        "config": "medical_meadow_health_advice",
        "data_files": None,
        "description": "Sample of patient health-advice QA",
    },
)

CLINICAL_PROMPT = PromptTemplate(
    input_variables=["context", "symptoms"],
    template=(
        "You are MediCare Local, an on-device clinical information assistant "
        "for a healthcare prototype. You are not a physician. You do not "
        "diagnose, prescribe, or invent evidence.\n\n"
        "Use the retrieved context as your evidence. It was retrieved from a "
        "local ChromaDB collection that merges clinical guideline PDFs and "
        "curated medical question-answer pairs. If the context does not "
        "support a claim, say that the local knowledge base does not cover it. "
        "Do not invent citations, doses, or study results.\n\n"
        "If the symptoms suggest an emergency (trouble breathing, chest pain, "
        "fainting, one-sided weakness, severe bleeding, confusion, a rapidly "
        "worsening allergic reaction, or thoughts of self-harm), tell the "
        "person to contact emergency services now, before any other advice.\n\n"
        "Write in plain English with these sections:\n"
        "1. What to do right now\n"
        "2. What the retrieved sources support\n"
        "3. What a clinician should evaluate\n"
        "4. Limits of this prototype\n\n"
        "Retrieved context:\n{context}\n\n"
        "Reviewed symptom description:\n{symptoms}\n\n"
        "Response:"
    ),
)


def _clean_text(value: object) -> str:
    """Flatten a dataset cell into a single line of text."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple)):
        parts = [_clean_text(item) for item in value]
        text = " ".join(part for part in parts if part)
    elif isinstance(value, Mapping):
        # Chat messages are handled separately; other structs become text.
        text = " ".join(
            _clean_text(item) for item in value.values() if _clean_text(item)
        )
    else:
        text = str(value)
    return " ".join(text.replace("\x00", " ").split())


def _first_text(row: Mapping, keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _clean_text(row.get(key))
        if text:
            return text
    return ""


def _looks_like_task_instruction(text: str) -> bool:
    lowered = text.lower()
    return (
        lowered.startswith("if you are a doctor")
        or lowered.startswith("you are a")
        or lowered.startswith("you are an")
        or "please answer" in lowered[:180]
    )


def _qa_from_messages(messages: object) -> tuple[str, str]:
    """Pull the user turn and the assistant turn out of a chat-formatted row."""
    if isinstance(messages, tuple):
        messages = list(messages)
    elif not isinstance(messages, list):
        return "", ""
    question = ""
    answer = ""
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = _clean_text(message.get("role")).lower()
        content = _clean_text(message.get("content"))
        if not content:
            continue
        if role == "user":
            question = content
        elif role == "assistant":
            answer = content
    return question, answer


def extract_qa_pair(row: Mapping) -> tuple[str, str]:
    """Normalize one dataset row into a question and a recommendation."""
    data = {str(key).strip().lower(): value for key, value in dict(row).items()}
    question, answer = _qa_from_messages(data.get("messages"))
    if question and answer:
        return question, answer

    patient_input = _clean_text(data.get("input"))
    instruction = _clean_text(data.get("instruction"))
    question = _first_text(data, ("question", "questions", "prompt"))
    answer = _first_text(
        data,
        (
            "answer",
            "human",
            "completion",
            "best_answer",
            "answer_chatdoctor",
            "output",
        ),
    )

    if patient_input and (
        not question or question == instruction or _looks_like_task_instruction(question)
    ):
        question = patient_input
    if not question and instruction and not _looks_like_task_instruction(instruction):
        question = instruction
    if not answer:
        answer = _clean_text(data.get("output"))
    return question, answer


def format_qa_text(question: str, answer: str) -> str:
    """Format required by the hybrid knowledge base."""
    question = question.strip().rstrip(".")
    answer = answer.strip().rstrip(".")
    return f"Patient Question: {question}. Medical Recommendation: {answer}."


def qa_rows_to_documents(rows, source_name: str) -> list[Document]:
    """Iterate QA rows and convert each pair into a LangChain Document."""
    documents: list[Document] = []
    for index, row in enumerate(rows):
        if index >= MAX_ROWS_PER_HF_SOURCE:
            break
        if not isinstance(row, Mapping):
            continue
        question, answer = extract_qa_pair(row)
        if not question or not answer:
            continue
        documents.append(
            Document(
                page_content=format_qa_text(question, answer),
                metadata={
                    "source": source_name,
                    "source_type": "huggingface_qa",
                    "row_index": index,
                },
            )
        )
    return documents


def _is_connectivity_error(exc: BaseException) -> bool:
    text = f"{exc.__class__.__name__} {exc}".lower()
    needles = (
        "connection",
        "timeout",
        "timed out",
        "name resolution",
        "temporary failure",
        "network",
        "offline",
        "failed to resolve",
        "max retries",
        "nodename nor servname",
        "unreachable",
    )
    return any(needle in text for needle in needles)


def _format_hf_error(repo: str, exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    if _is_connectivity_error(exc):
        return (
            f"No internet connection while downloading {repo}. "
            "The first index build needs network access for Hugging Face. "
            "Connect, then use Rebuild knowledge base. "
            f"Details: {detail}"
        )
    return f"Could not load {repo}: {detail}"


def _load_hf_split(spec: Mapping):
    repo = spec["repo"]
    config = spec.get("config")
    data_files = spec.get("data_files")

    def _call(split: str):
        kwargs = {"split": split}
        if data_files:
            kwargs["data_files"] = data_files
        if config:
            return load_dataset(repo, config, **kwargs)
        return load_dataset(repo, **kwargs)

    sliced = f"train[:{MAX_ROWS_PER_HF_SOURCE}]"
    try:
        return _call(sliced)
    except Exception as sliced_error:
        if _is_connectivity_error(sliced_error):
            raise
        dataset = _call("train")
        limit = min(MAX_ROWS_PER_HF_SOURCE, len(dataset))
        return dataset.select(range(limit))


def load_huggingface_documents() -> tuple[list[Document], list[str], dict[str, int]]:
    """Download the three QA repos and merge them into Document objects."""
    documents: list[Document] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}

    for spec in HF_SOURCES:
        repo = spec["repo"]
        try:
            dataset = _load_hf_split(spec)
            docs = qa_rows_to_documents(dataset, repo)
            if not docs:
                warnings.append(
                    f"{repo} downloaded, but no question-answer rows could be parsed."
                )
            documents.extend(docs)
            counts[repo] = len(docs)
            logger.info("Loaded %s documents from %s", len(docs), repo)
        except Exception as exc:
            counts[repo] = 0
            warnings.append(_format_hf_error(repo, exc))
            logger.warning("Hugging Face source failed: %s", warnings[-1])

    return documents, warnings, counts


def load_pdf_documents() -> tuple[list[Document], list[str], int]:
    """Ingest clinical guideline PDFs from ./knowledge_base and split them."""
    warnings: list[str] = []
    if not KNOWLEDGE_BASE_DIR.exists() or not KNOWLEDGE_BASE_DIR.is_dir():
        warnings.append(
            f"Local folder {KNOWLEDGE_BASE_DIR.as_posix()} is missing. "
            "PDF guidelines were not ingested. Create that folder, add .pdf files, "
            "and rebuild the knowledge base."
        )
        return [], warnings, 0

    pdf_paths = sorted(KNOWLEDGE_BASE_DIR.glob("*.pdf"))
    if not pdf_paths:
        warnings.append(
            f"{KNOWLEDGE_BASE_DIR.as_posix()} exists, but it has no PDF files. "
            "Add clinical guidelines and rebuild the knowledge base."
        )
        return [], warnings, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pages: list[Document] = []
    for pdf_path in pdf_paths:
        try:
            loaded = PyPDFLoader(str(pdf_path)).load()
        except Exception as exc:
            warnings.append(f"Could not read {pdf_path.name}: {exc}")
            continue
        for page in loaded:
            page.metadata["source"] = pdf_path.name
            page.metadata["source_type"] = "local_pdf"
        pages.extend(loaded)

    if not pages:
        warnings.append("PDF files were found, but none produced readable text.")
        return [], warnings, len(pdf_paths)

    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source_type"] = "local_pdf"
    return chunks, warnings, len(pdf_paths)


def collect_documents() -> tuple[list[Document], dict]:
    """Merge Source A (local PDFs) and Source B (Hugging Face QA) ."""
    pdf_chunks, pdf_warnings, pdf_files = load_pdf_documents()
    qa_documents, hf_warnings, hf_counts = load_huggingface_documents()
    merged = pdf_chunks + qa_documents
    stats = {
        "complete": False,
        "pdf_files": pdf_files,
        "pdf_chunks": len(pdf_chunks),
        "qa_documents": len(qa_documents),
        "chunk_count": len(merged),
        "sources": hf_counts,
        "warnings": pdf_warnings + hf_warnings,
    }
    return merged, stats


def _ollama_tags() -> list[str]:
    url = f"{OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama is not running at "
            f"{OLLAMA_BASE_URL}. Start it with `ollama serve` "
            "(or open the Ollama app), then pull the models below."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Could not reach Ollama at {OLLAMA_BASE_URL}: {exc}") from exc

    models = payload.get("models", []) if isinstance(payload, dict) else []
    return [str(item.get("name", "")) for item in models if isinstance(item, dict)]


def _model_is_pulled(available: list[str], wanted: str) -> bool:
    wanted_lower = wanted.lower()
    for name in available:
        normalized = name.lower()
        base = normalized.split(":", 1)[0]
        if base == wanted_lower or normalized.startswith(wanted_lower):
            return True
    return False


def ensure_ollama_models() -> list[str]:
    """Fail early with install instructions if a required local model is absent."""
    available = _ollama_tags()
    missing = [
        model
        for model in (LLM_MODEL, EMBED_MODEL)
        if not _model_is_pulled(available, model)
    ]
    if missing:
        pulls = " ".join(f"`ollama pull {model}`" for model in missing)
        raise RuntimeError(
            "Ollama is running, but these models are not pulled: "
            + ", ".join(missing)
            + f". Run {pulls} and rebuild the knowledge base."
        )
    return available


def _read_stats() -> dict | None:
    if not STATS_PATH.exists():
        return None
    try:
        payload = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _index_is_ready() -> bool:
    stats = _read_stats()
    sqlite_path = PERSIST_DIR / "chroma.sqlite3"
    return bool(stats and stats.get("complete") and sqlite_path.exists())


def _persistent_client() -> chromadb.PersistentClient:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(PERSIST_DIR),
        settings=CHROMA_SETTINGS,
    )


def _open_vectorstore(
    embeddings: OllamaEmbeddings,
    client: chromadb.PersistentClient,
) -> Chroma:
    # Pass the already-open client so LangChain does not open a second
    # SQLite connection against the same local collection.
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def _reset_collection(client: chromadb.PersistentClient) -> None:
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        logger.info("No existing %s collection to delete.", COLLECTION_NAME)
    if STATS_PATH.exists():
        STATS_PATH.unlink()


def _embed_documents(vectorstore: Chroma, documents: list[Document]) -> None:
    for start in range(0, len(documents), EMBED_BATCH_SIZE):
        batch = documents[start : start + EMBED_BATCH_SIZE]
        vectorstore.add_documents(batch)
        logger.info(
            "Embedded chunks %s-%s of %s",
            start + 1,
            start + len(batch),
            len(documents),
        )


def build_vectorstore(
    embeddings: OllamaEmbeddings,
    client: chromadb.PersistentClient,
) -> tuple[Chroma, dict]:
    """Create the single local collection from PDFs and Hugging Face QA."""
    _reset_collection(client)
    documents, stats = collect_documents()
    if not documents:
        detail = " ".join(stats["warnings"]) or "No PDF or Hugging Face documents were loaded."
        raise RuntimeError(
            "The knowledge base is empty, so there is nothing to index. " + detail
        )

    vectorstore = _open_vectorstore(embeddings, client)
    _embed_documents(vectorstore, documents)
    stats["complete"] = True
    stats["chunk_count"] = len(documents)
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return vectorstore, stats


@st.cache_resource(show_spinner="Preparing the local ChromaDB knowledge base...")
def load_knowledge_base() -> tuple[Chroma, dict]:
    """Reuse a finished index, or build one from both knowledge sources."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    client = _persistent_client()
    rebuild = REBUILD_FLAG.exists()
    if rebuild:
        REBUILD_FLAG.unlink()

    if _index_is_ready() and not rebuild:
        stats = _read_stats() or {}
        return _open_vectorstore(embeddings, client), stats

    ensure_ollama_models()
    return build_vectorstore(embeddings, client)


@st.cache_resource(show_spinner="Loading Whisper on CPU (int8)...")
def load_whisper_model() -> WhisperModel:
    # Apple Silicon has no CUDA. int8 on CPU is the supported configuration.
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )


def _audio_suffix(payload: bytes, filename: str | None) -> str:
    if filename and "." in filename:
        extension = Path(filename).suffix.lower()
        if extension in {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}:
            return extension
    if payload.startswith(b"RIFF"):
        return ".wav"
    if payload.startswith(b"fLaC"):
        return ".flac"
    if payload.startswith(b"OggS"):
        return ".ogg"
    if payload.startswith(b"ID3"):
        return ".mp3"
    return ".wav"


def _audio_token(uploaded_file) -> str:
    file_id = getattr(uploaded_file, "file_id", None)
    if file_id:
        return str(file_id)
    return str(hash(uploaded_file.getvalue()))


def transcribe_audio_file(audio_path: str) -> str:
    """Transcribe a local audio file to English text."""
    model = load_whisper_model()
    try:
        segments, _info = model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            beam_size=1,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception:
        segments, _info = model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            beam_size=1,
            vad_filter=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise RuntimeError(
            "Whisper did not return any speech. Record again, closer to the microphone."
        )
    return text


def transcribe_upload(uploaded_file) -> str:
    """Persist the Streamlit upload so faster-whisper can read a real path."""
    payload = uploaded_file.getvalue()
    if not payload:
        raise RuntimeError("The recording was empty. Please record again.")
    suffix = _audio_suffix(payload, getattr(uploaded_file, "name", None))
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary.write(payload)
        temporary.flush()
        temporary.close()
        return transcribe_audio_file(temporary.name)
    finally:
        temporary.close()
        try:
            os.unlink(temporary.name)
        except OSError:
            pass


def _context_from_docs(documents: list[Document]) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        kind = document.metadata.get("source_type", "unknown")
        blocks.append(
            f"[{index}] source={source} type={kind}\n{document.page_content}"
        )
    return "\n\n".join(blocks)


def retrieve_context(vectorstore: Chroma, symptoms: str) -> list[Document]:
    """Embed the reviewed symptoms and search the merged local collection."""
    return vectorstore.similarity_search(symptoms, k=RETRIEVAL_K)


def generate_answer(symptoms: str, retrieved: list[Document]) -> str:
    """Pass retrieved chunks and the reviewed symptoms to MedGemma."""
    context = _context_from_docs(retrieved) or (
        "No relevant context was retrieved from the local knowledge base."
    )
    prompt = CLINICAL_PROMPT.format(context=context, symptoms=symptoms)
    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    raw = llm.invoke(prompt)
    if isinstance(raw, str):
        answer = raw.strip()
    else:
        answer = str(getattr(raw, "content", raw)).strip()
    if not answer:
        raise RuntimeError("MedGemma returned an empty response.")
    return answer


def _plain_sources(documents: list[Document]) -> list[dict]:
    plain = []
    for document in documents:
        metadata = {}
        for key, value in (document.metadata or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
            else:
                metadata[key] = str(value)
        plain.append({"content": document.page_content, "metadata": metadata})
    return plain


def _init_session_state() -> None:
    defaults = {
        "transcribed_text": "",
        "transcript_version": 0,
        "audio_token": None,
        "transcription_error": None,
        "analysis": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_sidebar(stats: dict | None, kb_error: str | None) -> None:
    st.sidebar.header("On this Mac")
    st.sidebar.caption(
        "Audio and transcripts are not uploaded. "
        "Retrieval uses a local ChromaDB collection."
    )
    st.sidebar.markdown(f"**LLM:** `{LLM_MODEL}`")
    st.sidebar.markdown(f"**Embeddings:** `{EMBED_MODEL}`")
    st.sidebar.markdown(
        f"**Speech-to-text:** faster-whisper `{WHISPER_MODEL_SIZE}` "
        f"({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})"
    )
    if kb_error:
        st.sidebar.error(kb_error)
    if stats:
        st.sidebar.metric("Indexed chunks", int(stats.get("chunk_count", 0)))
        st.sidebar.metric("PDF chunks", int(stats.get("pdf_chunks", 0)))
        st.sidebar.metric("QA documents", int(stats.get("qa_documents", 0)))
        sources = stats.get("sources") or {}
        if sources:
            st.sidebar.markdown("**Hugging Face rows indexed**")
            for repo, count in sources.items():
                st.sidebar.markdown(f"- `{repo}`: {count}")
        for warning in stats.get("warnings") or []:
            st.sidebar.warning(warning)
    st.sidebar.divider()
    st.sidebar.caption(
        "Rebuild after you add PDFs or change MAX_ROWS_PER_HF_SOURCE. "
        "Rebuilding re-embeds every chunk locally."
    )
    if st.sidebar.button("Rebuild knowledge base", use_container_width=True):
        REBUILD_FLAG.write_text("rebuild", encoding="utf-8")
        if STATS_PATH.exists():
            STATS_PATH.unlink()
        st.cache_resource.clear()
        st.rerun()


def _render_sources(sources: list[dict]) -> None:
    with st.expander("Retrieved source chunks", expanded=False):
        st.caption(
            "Exact chunks returned by similarity search over the single "
            "local collection (PDF guidelines and Hugging Face QA)."
        )
        if not sources:
            st.write("No chunks were retrieved.")
            return
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata") or {}
            label = metadata.get("source", "unknown")
            kind = metadata.get("source_type", "unknown")
            details = []
            if "page" in metadata:
                details.append(f"page {metadata['page']}")
            if "row_index" in metadata:
                details.append(f"row {metadata['row_index']}")
            suffix = f" ({', '.join(details)})" if details else ""
            st.markdown(f"**{index}. {label}** · `{kind}`{suffix}")
            st.text(source.get("content", ""))


def _handle_audio() -> None:
    audio = st.audio_input("Record your symptoms")
    if audio is None:
        return
    st.audio(audio)
    token = _audio_token(audio)
    if st.session_state.audio_token == token:
        return
    with st.spinner("Transcribing on this Mac with Whisper (CPU, int8)..."):
        try:
            transcript = transcribe_upload(audio)
        except Exception as exc:
            st.session_state.audio_token = token
            st.session_state.transcription_error = str(exc)
            return
    st.session_state.audio_token = token
    st.session_state.transcription_error = None
    st.session_state.transcribed_text = transcript
    st.session_state.transcript_version += 1
    st.session_state.analysis = None


def main() -> None:
    st.set_page_config(
        page_title="MediCare Local",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.4rem; max-width: 1120px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _init_session_state()

    st.title("MediCare Local")
    st.caption(
        "Local-first symptom intake for Apple Silicon. "
        "Speech is transcribed on device, you review the text, then MedGemma "
        "answers from a private ChromaDB index."
    )
    st.info(
        "Prototype only. This tool does not diagnose, prescribe, or replace a "
        "clinician. If symptoms are severe, sudden, or worsening, contact "
        "emergency services.",
        icon="ℹ️",
    )

    kb_error = None
    stats = None
    vectorstore = None
    try:
        vectorstore, stats = load_knowledge_base()
    except Exception as exc:
        kb_error = str(exc)

    _render_sidebar(stats, kb_error)

    intake, result = st.columns([1.05, 0.95], gap="large")

    with intake:
        st.subheader("Describe symptoms")
        st.markdown(
            "Record a short description. The transcript is placed in the editor "
            "so you can correct medical words before anything is analyzed."
        )
        _handle_audio()
        if kb_error:
            st.error(kb_error)
        if st.session_state.transcription_error:
            st.error(st.session_state.transcription_error)

        transcribed_text = st.session_state.transcribed_text
        # A new recording bumps transcript_version, which gives the text area
        # a fresh key so value=transcribed_text replaces the previous draft.
        # Edits persist until the next recording. Analysis uses this text only.
        edited_symptoms = st.text_area(
            "Review and edit your symptoms:",
            value=transcribed_text,
            height=220,
            key=f"symptom_draft_{st.session_state.transcript_version}",
            placeholder="Your transcript appears here. You can also type symptoms directly.",
        )

        analyze = st.button(
            "Analyze Symptoms",
            type="primary",
            disabled=vectorstore is None,
        )
        if analyze:
            symptoms = edited_symptoms.strip()
            if not symptoms:
                st.warning("Enter or record symptoms before analysis.")
            elif vectorstore is None:
                st.error(kb_error or "The local knowledge base is not ready.")
            else:
                with st.status("Running local retrieval...", expanded=True) as status:
                    try:
                        status.write(
                            "Embedding the reviewed symptom text with nomic-embed-text "
                            "and searching the local collection."
                        )
                        documents = retrieve_context(vectorstore, symptoms)
                        status.write(
                            f"Retrieved {len(documents)} chunks from PDFs and "
                            "Hugging Face QA data."
                        )
                        status.write(f"Sending the context and symptoms to {LLM_MODEL}.")
                        answer = generate_answer(symptoms, documents)
                    except Exception as exc:
                        status.update(label="Analysis failed", state="error")
                        st.session_state.analysis = None
                        message = str(exc)
                        if "11434" in message or "Ollama" in message or "ollama" in message:
                            message = (
                                "MedGemma could not be reached. Confirm Ollama is "
                                f"running and `{LLM_MODEL}` is pulled. Details: {message}"
                            )
                        st.error(message)
                    else:
                        status.update(label="Analysis complete", state="complete")
                        st.session_state.analysis = {
                            "symptoms": symptoms,
                            "answer": answer,
                            "sources": _plain_sources(documents),
                        }

    with result:
        st.subheader("Guidance")
        analysis = st.session_state.analysis
        if not analysis:
            st.markdown(
                "The response will appear here after you click **Analyze Symptoms**. "
                "Retrieval does not run on the recording itself."
            )
        else:
            st.markdown("**Reviewed symptoms**")
            st.write(analysis["symptoms"])
            st.markdown("**MedGemma**")
            st.markdown(analysis["answer"])
            _render_sources(analysis["sources"])
            st.caption(
                "Text only. This prototype has no text-to-speech. "
                "Confirm any next step with a licensed clinician."
            )


if __name__ == "__main__":
    main()
