# ---------------------------------------------------------------------------
# MediCare Local — on-device clinical intake prototype (Apple Silicon)
#
# Patient audio, transcripts, and retrieval stay on this machine.
# The first launch needs internet once, to download Hugging Face knowledge
# and the Whisper weights. After that, inference is local via Ollama.
#
# Required pip packages (including datasets):
#
#   pip install streamlit chromadb langchain langchain-community langchain-chroma \
#       langchain-core langchain-text-splitters pypdf faster-whisper datasets
#
# System tools (not pip):
#   - Ollama: https://ollama.com
#       ollama pull medgemma
#       ollama pull nomic-embed-text
#   - ffmpeg is recommended so faster-whisper can decode microphone audio
# ---------------------------------------------------------------------------

from __future__ import annotations

import html
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components
from chromadb.config import Settings
from datasets import load_dataset
# from datasets.utils.logging import disable_progress_bars
from faster_whisper import WhisperModel
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
# os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# disable_progress_bars()

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
PROFILE_PATH = Path("./patient_profile.json")
CHECKED_SOURCES_PATH = Path(__file__).resolve().parent / "checked_sources.json"
MAX_CHECKED_SOURCES = 4
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
HF_DESCRIPTIONS = {spec["repo"]: spec["description"] for spec in HF_SOURCES}

CLINICAL_PROMPT = PromptTemplate(
    input_variables=["context", "symptoms", "profile"],
    template=(
        "You are MediCare Local, an on-device clinical information assistant "
        "for a healthcare prototype. You are not a physician. You do not "
        "diagnose, prescribe, or invent evidence.\n\n"
        "Use the retrieved context as your evidence. It was retrieved from a "
        "local ChromaDB collection that merges clinical guideline PDFs and "
        "curated medical question-answer pairs. If the context does not "
        "support a claim, say that the local knowledge base does not cover it. "
        "Do not invent citations, doses, or study results.\n\n"
        "The health profile is information the user entered on this device. "
        "Use age, the date of the last period, chronic conditions, and current "
        "medicines to judge whether a retrieved passage applies. Do not tell "
        "the user to start, stop, or change a medicine or dose. If a listed "
        "condition or medicine may change what is safe, say that a clinician "
        "needs to review it.\n\n"
        "If the symptoms suggest an emergency (trouble breathing, chest pain, "
        "fainting, one-sided weakness, severe bleeding, confusion, a rapidly "
        "worsening allergic reaction, or thoughts of self-harm), tell the "
        "person to contact emergency services now, before any other advice.\n\n"
        "Write in plain English with these sections:\n"
        "1. What to do right now\n"
        "2. What the retrieved sources support for this profile\n"
        "3. What a clinician should evaluate, including conditions and medicines\n"
        "4. Limits of this prototype\n\n"
        "Health profile:\n{profile}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Reviewed symptom description:\n{symptoms}\n\n"
        "Response:"
    ),
)

FOLLOWUP_PROMPT = PromptTemplate(
    input_variables=[
        "context",
        "symptoms",
        "profile",
        "complete_answers",
        "earlier_followups",
        "followup",
    ],
    template=(
        "You are MediCare Local, continuing a conversation on this device. "
        "You are not a physician. You do not diagnose, prescribe, or invent evidence.\n\n"
        "Read every complete answer below from beginning to end before you reply. "
        "Account for the whole answer, not only the last sentence. The user's "
        "follow-up may say that part of it was already discussed with a doctor, "
        "that they want to avoid some options, or that the concern is still open. "
        "Apply those points to the full answer.\n\n"
        "Use the retrieved context as your evidence. If it does not support a "
        "claim, say so. Do not invent citations, doses, or study results. "
        "Do not tell the user to start, stop, or change a medicine or dose. "
        "Treat a doctor's earlier discussion as already settled unless the user "
        "asks to revisit it. Do not recommend an option the user wants to avoid. "
        "If the retrieved context only supports an avoided option, say that and "
        "leave the decision with a clinician.\n\n"
        "If the symptoms suggest an emergency (trouble breathing, chest pain, "
        "fainting, one-sided weakness, severe bleeding, confusion, a rapidly "
        "worsening allergic reaction, or thoughts of self-harm), tell the "
        "person to contact emergency services now.\n\n"
        "Write a short conversational reply that revises the full answer in light "
        "of the follow-up. Stop when the reply is complete. Do not ask whether "
        "the user is satisfied.\n\n"
        "Health profile:\n{profile}\n\n"
        "Reviewed symptoms:\n{symptoms}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Complete answers already given:\n{complete_answers}\n\n"
        "Earlier follow-ups from the user:\n{earlier_followups}\n\n"
        "This follow-up:\n{followup}\n\n"
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


def _open_vectorstore(embeddings: OllamaEmbeddings) -> Chroma:
    """Open the local collection through langchain-chroma.

    persist_directory makes Chroma use a PersistentClient under ./chroma_db.
    """
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
        client_settings=CHROMA_SETTINGS,
    )


def _reset_collection(vectorstore: Chroma) -> None:
    vectorstore.reset_collection()
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


def build_vectorstore(embeddings: OllamaEmbeddings) -> tuple[Chroma, dict]:
    """Create the single local collection from PDFs and Hugging Face QA."""
    vectorstore = _open_vectorstore(embeddings)
    _reset_collection(vectorstore)
    documents, stats = collect_documents()
    if not documents:
        detail = " ".join(stats["warnings"]) or "No PDF or Hugging Face documents were loaded."
        raise RuntimeError(
            "The knowledge base is empty, so there is nothing to index. " + detail
        )

    _embed_documents(vectorstore, documents)
    stats["complete"] = True
    stats["chunk_count"] = len(documents)
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return vectorstore, stats


@st.cache_resource(show_spinner="Getting your private health library ready...")
def load_knowledge_base() -> tuple[Chroma, dict]:
    """Reuse a finished index, or build one from both knowledge sources."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    rebuild = REBUILD_FLAG.exists()
    if rebuild:
        REBUILD_FLAG.unlink()

    if _index_is_ready() and not rebuild:
        stats = _read_stats() or {}
        return _open_vectorstore(embeddings), stats

    ensure_ollama_models()
    return build_vectorstore(embeddings)


@st.cache_resource(show_spinner="Preparing voice recognition...")
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


def retrieve_context(vectorstore: Chroma, symptoms: str, profile_text: str) -> list[Document]:
    """Embed symptoms plus the health profile and search the local collection."""
    query = f"{symptoms}\n\n{profile_text}"
    return vectorstore.similarity_search(query, k=RETRIEVAL_K)


def _invoke_medgemma(prompt: str) -> str:
    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    raw = llm.invoke(prompt)
    if isinstance(raw, str):
        answer = raw.strip()
    else:
        answer = str(getattr(raw, "content", raw)).strip()
    if not answer:
        raise RuntimeError("MedGemma returned an empty response.")
    return answer


def generate_answer(symptoms: str, profile_text: str, retrieved: list[Document]) -> str:
    """Pass retrieved chunks, the health profile, and the reviewed symptoms to MedGemma."""
    context = _context_from_docs(retrieved) or (
        "No relevant context was retrieved from the local knowledge base."
    )
    prompt = CLINICAL_PROMPT.format(
        context=context,
        symptoms=symptoms,
        profile=profile_text,
    )
    return _invoke_medgemma(prompt)


def _complete_answers(analysis: dict) -> str:
    """Every model reply in full, so a follow-up is applied to the whole answer."""
    parts = [f"Answer 1:\n{analysis.get('answer', '')}"]
    number = 2
    for turn in analysis.get("thread") or []:
        if turn.get("role") != "assistant":
            continue
        parts.append(f"Answer {number}:\n{turn.get('content', '')}")
        number += 1
    return "\n\n".join(parts)


def _earlier_followups(thread: list[dict]) -> str:
    messages = [
        turn.get("content", "").strip()
        for turn in thread
        if turn.get("role") == "user" and turn.get("content", "").strip()
    ]
    return "\n\n".join(messages) or "None yet."


def _load_checked_sources() -> list[dict]:
    """Checked web pages shipped with the app. The model cannot add its own links."""
    try:
        payload = json.loads(CHECKED_SOURCES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    sources = []
    if not isinstance(payload, list):
        return sources
    for item in payload:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if not url.startswith("https://") or not title:
            continue
        keywords = item.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        sources.append(
            {
                "title": title,
                "publisher": str(item.get("publisher") or "Checked page").strip(),
                "url": url,
                "keywords": [str(keyword).lower() for keyword in keywords if str(keyword).strip()],
                "fallback": bool(item.get("fallback")),
            }
        )
    return sources


CHECKED_SOURCES = _load_checked_sources()


def _match_checked_sources(*texts: str) -> list[dict]:
    """Pick checked pages whose topics appear in the answer or symptoms."""
    haystack = " ".join(text.lower() for text in texts if text)
    scored: list[tuple[int, dict]] = []
    for source in CHECKED_SOURCES:
        score = sum(1 for keyword in source["keywords"] if keyword and keyword in haystack)
        if score:
            scored.append((score, source))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = [source for _, source in scored[:MAX_CHECKED_SOURCES]]
    if chosen:
        return chosen
    return [source for source in CHECKED_SOURCES if source.get("fallback")][:MAX_CHECKED_SOURCES]


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _library_chips(sources: list[dict]) -> list[dict]:
    """One chip per knowledge-base document behind an answer, without duplicates."""
    chips: list[dict] = []
    seen: set[str] = set()
    for source in sources or []:
        metadata = source.get("metadata") or {}
        name = str(metadata.get("source") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        if metadata.get("source_type") == "huggingface_qa":
            url = f"https://huggingface.co/datasets/{name}"
            chips.append(
                {"name": HF_DESCRIPTIONS.get(name, name), "meta": _domain(url), "url": url}
            )
        else:
            label = Path(name).stem.replace("_", " ").replace("-", " ").strip()
            chips.append(
                {
                    "name": label[:1].upper() + label[1:] if label else name,
                    "meta": "Clinical guideline on this device",
                    "url": None,
                }
            )
    return chips


def _chip_html(chip: dict) -> str:
    name = html.escape(chip["name"])
    meta = html.escape(chip["meta"])
    if chip.get("url"):
        return (
            f'<a class="mc-chip" href="{html.escape(chip["url"], quote=True)}" '
            'target="_blank" rel="noopener noreferrer">'
            f'<span class="mc-chip-name">{name}</span>'
            f'<span class="mc-chip-meta">{meta} ↗</span></a>'
        )
    return (
        '<span class="mc-chip mc-chip-static">'
        f'<span class="mc-chip-name">{name}</span>'
        f'<span class="mc-chip-meta">{meta}</span></span>'
    )


def _render_learn_more(sources: list[dict], *texts: str) -> None:
    """Checked web pages and knowledge-base documents, shown as chips under a reply."""
    chips = [
        {"name": page["title"], "meta": f'{page["publisher"]} · {_domain(page["url"])}', "url": page["url"]}
        for page in _match_checked_sources(*texts)
    ]
    chips.extend(_library_chips(sources))
    if not chips:
        return
    st.markdown(
        '<div class="mc-sources">'
        '<div class="mc-sources-title">Want to learn more?</div>'
        '<div class="mc-sources-intro">You may want to check these sources to learn more '
        "about the conditions we discussed.</div>"
        '<div class="mc-chip-row">' + "".join(_chip_html(chip) for chip in chips) + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def continue_conversation(
    vectorstore: Chroma,
    analysis: dict,
    followup: str,
) -> tuple[str, list[Document]]:
    """Retrieve again and reply using the full answers plus this follow-up."""
    query = "\n\n".join(
        part
        for part in (analysis.get("symptoms", ""), analysis.get("profile", ""), followup)
        if part
    )
    retrieved = vectorstore.similarity_search(query, k=RETRIEVAL_K)
    context = _context_from_docs(retrieved) or (
        "No relevant context was retrieved from the local knowledge base."
    )
    prompt = FOLLOWUP_PROMPT.format(
        context=context,
        symptoms=analysis.get("symptoms", ""),
        profile=analysis.get("profile", ""),
        complete_answers=_complete_answers(analysis),
        earlier_followups=_earlier_followups(analysis.get("thread") or []),
        followup=followup,
    )
    return _invoke_medgemma(prompt), retrieved


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


def _shift_years(day: date, years: int) -> date:
    """Move a date by whole years, including 29 February."""
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)


def _age_years(born: date, today: date | None = None) -> int:
    today = today or date.today()
    years = today.year - born.year
    if (today.month, today.day) < (born.month, born.day):
        years -= 1
    return years


def _empty_profile() -> dict:
    return {
        "date_of_birth": None,
        "last_period_date": None,
        "chronic_conditions": "",
        "medications": "",
    }


def _valid_birth_date(born: date) -> bool:
    today = date.today()
    return _shift_years(today, -100) <= born <= _shift_years(today, -10)


def _parse_saved_period(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    if parsed > date.today():
        return None
    return value


def _read_profile_file() -> dict:
    """Load the saved profile, including the last period date."""
    profile = _empty_profile()
    if not PROFILE_PATH.exists():
        return profile
    try:
        payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return profile
    if not isinstance(payload, dict):
        return profile
    birth = payload.get("date_of_birth")
    if isinstance(birth, str) and birth:
        try:
            born = date.fromisoformat(birth)
        except ValueError:
            born = None
        if born is not None and _valid_birth_date(born):
            profile["date_of_birth"] = birth
    profile["last_period_date"] = _parse_saved_period(payload.get("last_period_date"))
    for key in ("chronic_conditions", "medications"):
        value = payload.get(key)
        if isinstance(value, str):
            profile[key] = value.strip()
    return profile


def _write_profile_file(profile: dict) -> None:
    """Persist date of birth, last period, illnesses, and medicines."""
    durable = {
        "date_of_birth": profile.get("date_of_birth"),
        "last_period_date": profile.get("last_period_date"),
        "chronic_conditions": profile.get("chronic_conditions") or "",
        "medications": profile.get("medications") or "",
    }
    PROFILE_PATH.write_text(json.dumps(durable, indent=2), encoding="utf-8")


def _profile_problem(profile: dict) -> str | None:
    if not profile.get("date_of_birth"):
        return "Add your date of birth in your profile so guidance can match your age."
    if not profile.get("last_period_date"):
        return "Add the first day of your last period in your profile."
    return None


def _profile_text(profile: dict) -> str:
    """Turn the profile into the block sent to retrieval and MedGemma."""
    birth = profile.get("date_of_birth")
    if not birth:
        age_lines = ["Date of birth: not provided", "Age: not provided"]
    else:
        born = date.fromisoformat(birth)
        age_lines = [
            f"Date of birth: {birth}",
            f"Age: {_age_years(born)} years, calculated from the date of birth",
        ]
    last_period = profile.get("last_period_date")
    if not last_period:
        period_line = "Date of last period: not provided"
    else:
        period_day = date.fromisoformat(last_period)
        days_ago = (date.today() - period_day).days
        period_line = f"Date of last period: {last_period} ({days_ago} days ago)"
    conditions = profile.get("chronic_conditions") or "none recorded"
    medications = profile.get("medications") or "none recorded"
    if profile.get("updated_this_visit"):
        visit_line = "This visit: the user updated period, illnesses, or medications."
    else:
        visit_line = "This visit: the user kept the saved period, illnesses, and medications."
    return "\n".join(
        [
            *age_lines,
            period_line,
            f"Illnesses: {conditions}",
            f"Medications currently taken: {medications}",
            visit_line,
        ]
    )


def _current_profile() -> dict:
    on_file = st.session_state.get("profile_on_file") or _empty_profile()
    return {**on_file, "updated_this_visit": st.session_state.get("profile_updated_this_visit", False)}


def _period_summary(last_period: str | None) -> str:
    if not last_period:
        return "Not added yet"
    period_day = date.fromisoformat(last_period)
    days_ago = (date.today() - period_day).days
    if days_ago == 0:
        ago = "today"
    elif days_ago == 1:
        ago = "yesterday"
    else:
        ago = f"{days_ago} days ago"
    return f"{period_day.strftime('%d %b %Y')} · {ago}"


def _seed_profile_form() -> None:
    on_file = st.session_state.get("profile_on_file") or _empty_profile()
    birth = on_file.get("date_of_birth")
    period = on_file.get("last_period_date")
    st.session_state.profile_birth_date = date.fromisoformat(birth) if birth else None
    st.session_state.profile_period_date = date.fromisoformat(period) if period else None
    st.session_state.profile_conditions = on_file.get("chronic_conditions") or ""
    st.session_state.profile_medications = on_file.get("medications") or ""


def _open_profile_editor() -> None:
    _seed_profile_form()
    st.session_state.profile_error = None
    st.session_state.profile_editing = True


def _close_profile_editor() -> None:
    st.session_state.profile_error = None
    st.session_state.profile_editing = False


def _save_profile_form() -> None:
    born = st.session_state.get("profile_birth_date")
    period = st.session_state.get("profile_period_date")
    if not isinstance(born, date):
        error = "Add your date of birth. We use it to work out your age."
    elif not _valid_birth_date(born):
        error = "Date of birth must correspond to an age from 10 to 100."
    elif not isinstance(period, date):
        error = "Add the first day of your last period."
    elif period > date.today():
        error = "The date of your last period can't be in the future."
    else:
        error = None
    if error:
        st.session_state.profile_error = error
        return
    profile = {
        "date_of_birth": born.isoformat(),
        "last_period_date": period.isoformat(),
        "chronic_conditions": st.session_state.get("profile_conditions", "").strip(),
        "medications": st.session_state.get("profile_medications", "").strip(),
    }
    _write_profile_file(profile)
    st.session_state.profile_on_file = profile
    st.session_state.profile_updated_this_visit = True
    st.session_state.profile_editing = False
    st.session_state.profile_error = None
    st.session_state.profile_notice = "Profile saved."
    analysis = st.session_state.get("analysis")
    if analysis:
        analysis["profile"] = _profile_text(_current_profile())


def _load_profile_into_session() -> None:
    """Load the saved profile once per visit; open the editor if it is incomplete."""
    if st.session_state.get("profile_ready"):
        return
    profile = _read_profile_file()
    st.session_state.profile_on_file = profile
    st.session_state.profile_updated_this_visit = False
    st.session_state.profile_editing = _profile_problem(profile) is not None
    if st.session_state.profile_editing:
        _seed_profile_form()
    st.session_state.profile_ready = True


def _init_session_state() -> None:
    defaults = {
        "transcribed_text": "",
        "transcript_version": 0,
        "intake_version": 0,
        "audio_token": None,
        "transcription_error": None,
        "analysis": None,
        "analysis_error": None,
        "profile_notice": None,
        "profile_error": None,
        "followup_version": 0,
        "followup_text": "",
        "followup_draft_version": 0,
        "followup_audio_token": None,
        "followup_transcription_error": None,
        "followup_error": None,
        "scroll_to_latest": False,
        "scroll_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    _load_profile_into_session()


APP_CSS = """
<style>
    .block-container {padding-top: 3.5rem; padding-bottom: 3rem; max-width: 900px;}

    .st-key-mc_banner {
        position: relative; overflow: hidden; isolation: isolate;
        background: linear-gradient(120deg, #F6D9E3 0%, #FBEEF2 48%, #EDE1F4 100%);
        border: 1px solid #EFD3DE; border-radius: 26px;
        padding: 1.1rem 1.6rem 1.4rem; margin-bottom: 0.4rem;
        box-shadow: 0 10px 30px -18px rgba(142, 58, 94, 0.45);
    }
    .st-key-mc_banner::before, .st-key-mc_banner::after {
        content: ""; position: absolute; z-index: -1; border-radius: 50%; filter: blur(2px);
    }
    .st-key-mc_banner::before {
        width: 260px; height: 260px; right: -70px; top: -110px;
        background: radial-gradient(circle at 35% 35%, rgba(217, 139, 168, 0.55), rgba(217, 139, 168, 0) 70%);
    }
    .st-key-mc_banner::after {
        width: 220px; height: 220px; left: 38%; bottom: -150px;
        background: radial-gradient(circle at 50% 50%, rgba(167, 128, 199, 0.35), rgba(167, 128, 199, 0) 70%);
    }
    .st-key-mc_banner_compact {
        background: linear-gradient(120deg, #F6D9E3 0%, #FBEEF2 60%, #EDE1F4 100%);
        border: 1px solid #EFD3DE; border-radius: 20px;
        padding: 0.45rem 1.1rem; margin-bottom: 0.4rem;
    }

    .st-key-home button {
        background: transparent; border: none; box-shadow: none;
        color: #8E3A5E; padding: 0.1rem 0.2rem; white-space: nowrap;
    }
    .st-key-home button p {font-size: 1.35rem; font-weight: 800; letter-spacing: 0.01em;
        white-space: nowrap;}
    .st-key-home button span[data-testid="stIconMaterial"] {
        background: linear-gradient(135deg, #B0466F, #D98BA8); color: #FFFFFF;
        border-radius: 50%; padding: 0.3rem; font-size: 1.1rem;
        box-shadow: 0 4px 10px -4px rgba(176, 70, 111, 0.7);
    }
    .st-key-home button:hover {background: rgba(255, 255, 255, 0.45);}
    .st-key-home button:hover p {color: #6E2448;}

    .mc-badge-wrap {display: flex; justify-content: flex-end;}
    .mc-badge {font-size: 0.78rem; color: #6B5A66; background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(235, 207, 218, 0.9);
        border-radius: 999px; padding: 0.25rem 0.75rem; white-space: nowrap;}

    .mc-greeting {font-size: 0.85rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #A0517A; margin-top: 0.35rem;}
    .mc-banner-title {font-size: 1.65rem; font-weight: 800; line-height: 1.25;
        color: #2F2533; margin: 0.2rem 0 0.35rem;}
    .mc-banner-title em {font-style: normal; color: #B0466F;}
    .mc-banner-sub {font-size: 0.98rem; color: #5E4E5A; max-width: 560px; line-height: 1.5;}
    .mc-pills {display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem;}
    .mc-pill {display: inline-flex; align-items: center; gap: 0.35rem;
        background: rgba(255, 255, 255, 0.75); border: 1px solid rgba(235, 207, 218, 0.9);
        border-radius: 999px; padding: 0.3rem 0.8rem; font-size: 0.84rem; color: #4A3A46;}

    .mc-notice {display: flex; gap: 0.7rem; align-items: flex-start;
        background: #FFFFFF; border: 1px solid #EBCFDA; border-left: 4px solid #D9534F;
        border-radius: 14px; padding: 0.7rem 1rem; font-size: 0.88rem; line-height: 1.45;
        color: #6B5A66; margin: 0.4rem 0 0.6rem;}
    .mc-notice-icon {font-size: 1.1rem; line-height: 1.3;}
    .mc-notice strong {color: #8E3A5E;}

    .mc-hero {text-align: center; margin: 1.6rem auto 1.4rem; max-width: 620px;}
    .mc-hero h1 {font-size: 2.3rem; line-height: 1.2; margin-bottom: 0.6rem; color: #2F2533;}
    .mc-hero p {font-size: 1.05rem; color: #6B5A66; margin: 0;}
    .mc-mic-hint {text-align: center; color: #8E3A5E; font-weight: 600;
        font-size: 0.95rem; margin: 0.5rem 0 0.35rem;}
    .mc-topics {display: flex; flex-wrap: wrap; justify-content: center; gap: 0.4rem;
        margin: 1.4rem 0 0.4rem;}
    .mc-topic {font-size: 0.82rem; color: #6B5A66; background: #FFFFFF;
        border: 1px dashed #E3C9D3; border-radius: 999px; padding: 0.25rem 0.7rem;}

    [data-testid="stChatMessage"] {background: #FFFFFF; border: 1px solid #F0E1E7;
        border-radius: 18px; padding: 1rem 1.1rem; margin-bottom: 0.75rem;}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #F7ECEF; border-color: #F7ECEF; margin-left: auto; max-width: 85%;}
    [data-testid="stChatMessageAvatarUser"] {background-color: #D98BA8; color: #FFFFFF;}
    [data-testid="stChatMessageAvatarAssistant"] {background-color: #8E3A5E; color: #FFFFFF;}

    #latest-answer {scroll-margin-top: 4.5rem;}

    .mc-sources {margin: 1rem 0 1.25rem; padding-top: 0.85rem; border-top: 1px solid #F0E1E7;}
    .mc-sources-title {font-weight: 700; font-size: 0.92rem; color: #8E3A5E;}
    .mc-sources-intro {font-size: 0.85rem; color: #6B5A66; margin: 0.15rem 0 0.6rem;}
    .mc-chip-row {display: flex; flex-wrap: wrap; gap: 0.5rem;}
    .mc-chip {display: inline-flex; flex-direction: column; gap: 0.05rem;
        padding: 0.45rem 0.9rem; border-radius: 16px; background: #FFF6F8;
        border: 1px solid #EBCFDA; text-decoration: none !important;
        transition: background 0.15s ease, border-color 0.15s ease;}
    a.mc-chip:hover {background: #F7E3EA; border-color: #D9A7BB;}
    .mc-chip-name {font-size: 0.88rem; font-weight: 600; color: #2F2533;}
    .mc-chip-meta {font-size: 0.74rem; color: #A0517A;}
    .mc-chip-static .mc-chip-meta {color: #8A7A86;}

    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: clamp(240px, 24vw, 320px);
        max-width: clamp(240px, 24vw, 320px);
    }

    .mc-side-title {font-size: 1.15rem; font-weight: 700; color: #2F2533; margin-bottom: 0.1rem;}
    .mc-profile {background: #FFFFFF; border: 1px solid #EBCFDA; border-radius: 16px;
        padding: 0.4rem 0.9rem; margin: 0.6rem 0 0.8rem;}
    .mc-profile-row {padding: 0.55rem 0; border-bottom: 1px solid #F5E8ED;}
    .mc-profile-row:last-child {border-bottom: none;}
    .mc-profile-label {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: #8A7A86;}
    .mc-profile-value {font-size: 0.95rem; color: #2F2533; margin-top: 0.1rem;
        overflow-wrap: anywhere;}
    .mc-profile-empty {color: #A89AA4; font-style: italic;}

    div[data-testid="stElementContainer"]:has(iframe[height="0"]),
    div.element-container:has(iframe[height="0"]) {display: none;}
</style>
"""


def _request_rebuild() -> None:
    REBUILD_FLAG.write_text("rebuild", encoding="utf-8")
    if STATS_PATH.exists():
        STATS_PATH.unlink()
    st.cache_resource.clear()


def _render_profile_summary() -> None:
    on_file = st.session_state.get("profile_on_file") or _empty_profile()
    birth = on_file.get("date_of_birth")
    age = f"{_age_years(date.fromisoformat(birth))} years" if birth else None
    rows = [
        ("Age", age),
        ("Last period", _period_summary(on_file.get("last_period_date")) if on_file.get("last_period_date") else None),
        ("Health conditions", on_file.get("chronic_conditions") or None),
        ("Medications", on_file.get("medications") or None),
    ]
    body = []
    for label, value in rows:
        shown = (
            html.escape(value)
            if value
            else '<span class="mc-profile-empty">None noted</span>'
        )
        body.append(
            '<div class="mc-profile-row">'
            f'<div class="mc-profile-label">{label}</div>'
            f'<div class="mc-profile-value">{shown}</div></div>'
        )
    st.markdown('<div class="mc-profile">' + "".join(body) + "</div>", unsafe_allow_html=True)
    st.caption("Has anything changed since your last visit? Keeping this up to date makes guidance more relevant.")
    st.button(
        "Update my profile",
        icon=":material/edit:",
        on_click=_open_profile_editor,
        width="stretch",
    )


def _render_profile_form() -> None:
    today = date.today()
    on_file = st.session_state.get("profile_on_file") or _empty_profile()
    can_cancel = _profile_problem(on_file) is None
    with st.form("profile_form", border=False):
        st.date_input(
            "Date of birth",
            key="profile_birth_date",
            min_value=_shift_years(today, -100),
            max_value=_shift_years(today, -10),
            help="We use this to work out your age.",
        )
        st.date_input(
            "First day of your last period",
            key="profile_period_date",
            min_value=_shift_years(today, -100),
            max_value=today,
        )
        st.text_area(
            "Health conditions",
            key="profile_conditions",
            height=80,
            placeholder="For example: endometriosis, PCOS, hypothyroidism",
        )
        st.text_area(
            "Medications",
            key="profile_medications",
            height=80,
            placeholder="Name and how you take it, for example: levothyroxine daily",
        )
        st.form_submit_button(
            "Save profile",
            type="primary",
            on_click=_save_profile_form,
            width="stretch",
        )
        if can_cancel:
            st.form_submit_button("Cancel", on_click=_close_profile_editor, width="stretch")
    if st.session_state.get("profile_error"):
        st.error(st.session_state.profile_error)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="mc-side-title">Your health profile</div>', unsafe_allow_html=True)
        st.caption("Kept privately on this device. We use it to tailor guidance to you.")
        if st.session_state.profile_editing:
            _render_profile_form()
        else:
            _render_profile_summary()
        notice = st.session_state.get("profile_notice")
        if notice:
            st.success(notice, icon=":material/check_circle:")
            st.session_state.profile_notice = None
        st.divider()
        with st.expander("App maintenance"):
            st.caption("Refresh the health library after new guideline PDFs are added.")
            st.button("Refresh health library", on_click=_request_rebuild, width="stretch")


def _greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def _render_brand(compact: bool) -> None:
    """Banner with the logo, which returns to the start screen."""
    with st.container(key="mc_banner_compact" if compact else "mc_banner"):
        home, badge = st.columns([1, 1], vertical_alignment="center")
        with home:
            st.button(
                "MediCare",
                icon=":material/favorite:",
                key="home",
                help="Back to the start",
                on_click=_start_new_conversation,
                type="tertiary",
                width="content",
            )
        with badge:
            st.markdown(
                '<div class="mc-badge-wrap"><div class="mc-badge">🔒 Private, stays on this device</div></div>',
                unsafe_allow_html=True,
            )
        if not compact:
            st.markdown(
                f'<div class="mc-greeting">{_greeting()} 🌸</div>'
                '<div class="mc-banner-title">Your companion for <em>gynecological health</em></div>'
                '<div class="mc-banner-sub">Talk through periods, pain, cycles, fertility, and '
                "menopause in your own words. Calm, private, and backed by checked sources.</div>"
                '<div class="mc-pills">'
                '<span class="mc-pill">🎙️ Speak or type</span>'
                '<span class="mc-pill">📚 Checked medical sources</span>'
                '<span class="mc-pill">💬 Ask follow-up questions</span>'
                '<span class="mc-pill">🩺 Prepare for your doctor visit</span>'
                "</div>",
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="mc-notice"><span class="mc-notice-icon">⚠️</span><div>'
        "<strong>This is help, not a medical consultation.</strong> "
        "MediCare does not diagnose, prescribe, or replace a clinician. "
        "If symptoms are severe, sudden, or getting worse (heavy bleeding, fainting, "
        "severe pain, chest pain, or trouble breathing), contact emergency services now.</div></div>",
        unsafe_allow_html=True,
    )


def _render_kb_problem(kb_error: str) -> None:
    st.error(
        "The health library isn't ready, so guidance is unavailable right now.",
        icon=":material/error:",
    )
    with st.expander("Technical details"):
        st.code(kb_error, language=None)
        st.button("Try again", on_click=_request_rebuild)


def _ollama_failure_message(exc: Exception) -> str:
    message = str(exc)
    if "11434" in message or "Ollama" in message or "ollama" in message:
        return (
            "MedGemma could not be reached. Confirm Ollama is "
            f"running and `{LLM_MODEL}` is pulled. Details: {message}"
        )
    return message


def _handle_audio() -> None:
    audio = st.audio_input(
        "Record your symptoms",
        key=f"symptom_audio_{st.session_state.intake_version}",
        label_visibility="collapsed",
    )
    if audio is None:
        return
    token = _audio_token(audio)
    if st.session_state.audio_token == token:
        return
    with st.spinner("Listening back to your recording..."):
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


def _handle_followup_audio(version: int) -> None:
    """Transcribe a follow-up recording into the single review box."""
    audio = st.audio_input(
        "Record a follow-up",
        key=f"followup_audio_{version}",
        label_visibility="collapsed",
    )
    if audio is None:
        return
    token = _audio_token(audio)
    if st.session_state.followup_audio_token == token:
        return
    with st.spinner("Listening back to your follow-up..."):
        try:
            transcript = transcribe_upload(audio)
        except Exception as exc:
            st.session_state.followup_audio_token = token
            st.session_state.followup_transcription_error = str(exc)
            return
    st.session_state.followup_audio_token = token
    st.session_state.followup_transcription_error = None
    st.session_state.followup_text = transcript
    st.session_state.followup_draft_version += 1


def _reset_followup_state() -> None:
    st.session_state.followup_version += 1
    st.session_state.followup_text = ""
    st.session_state.followup_draft_version += 1
    st.session_state.followup_audio_token = None
    st.session_state.followup_transcription_error = None
    st.session_state.followup_error = None


def _start_new_conversation() -> None:
    st.session_state.analysis = None
    st.session_state.analysis_error = None
    st.session_state.transcribed_text = ""
    st.session_state.transcript_version += 1
    st.session_state.intake_version += 1
    st.session_state.audio_token = None
    st.session_state.transcription_error = None
    _reset_followup_state()


def _render_welcome(vectorstore: Chroma | None) -> None:
    """First screen: a centered voice prompt and a transcript to check."""
    st.markdown(
        '<div class="mc-hero">'
        "<h1>How are you feeling today?</h1>"
        "<p>Tell me what's going on in your own words: what you're noticing, "
        "when it started, and what worries you. I'll share guidance from trusted "
        "sources, and you can ask follow-up questions.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 6, 1])
    with center:
        problem = _profile_problem(st.session_state.profile_on_file)
        if problem:
            st.info(
                "Start with your profile in the panel on the left. Your age and cycle "
                "help make the guidance relevant to you.",
                icon=":material/person:",
            )
        st.markdown(
            '<div class="mc-mic-hint">Tap the microphone and start speaking</div>',
            unsafe_allow_html=True,
        )
        _handle_audio()
        if st.session_state.transcription_error:
            st.error(st.session_state.transcription_error)

        symptoms = st.text_area(
            "Check what I heard, or type instead",
            value=st.session_state.transcribed_text,
            height=150,
            key=f"symptom_draft_{st.session_state.transcript_version}",
            placeholder="Your words appear here after recording. You can correct them, or simply type.",
        )
        if st.session_state.analysis_error:
            st.error(st.session_state.analysis_error)

        if st.button(
            "Get guidance",
            type="primary",
            icon=":material/favorite:",
            width="stretch",
            disabled=vectorstore is None,
        ):
            text = symptoms.strip()
            if problem:
                st.warning(problem)
            elif not text:
                st.warning("Record or type how you're feeling first.")
            else:
                _reset_followup_state()
                st.session_state.analysis_error = None
                profile_text = _profile_text(_current_profile())
                with st.spinner("Reading trusted sources and preparing your guidance..."):
                    try:
                        documents = retrieve_context(vectorstore, text, profile_text)
                        answer = generate_answer(text, profile_text, documents)
                    except Exception as exc:
                        st.session_state.analysis_error = _ollama_failure_message(exc)
                        st.rerun()
                # Stored before the page refreshes, so a refresh cannot start a second reply.
                st.session_state.analysis = {
                    "symptoms": text,
                    "profile": profile_text,
                    "answer": answer,
                    "sources": _plain_sources(documents),
                    "thread": [],
                }
                st.session_state.scroll_to_latest = True
                st.rerun()

        topics = (
            "Painful periods",
            "Heavy or irregular bleeding",
            "Pelvic pain",
            "Discharge or itching",
            "PMS and mood",
            "Menopause changes",
        )
        st.markdown(
            '<div class="mc-topics">'
            + "".join(f'<span class="mc-topic">{topic}</span>' for topic in topics)
            + "</div>",
            unsafe_allow_html=True,
        )


def _latest_anchor() -> None:
    st.markdown('<div id="latest-answer"></div>', unsafe_allow_html=True)


def _scroll_to_latest() -> None:
    """Scroll the page so the newest reply starts at the top of the view."""
    st.session_state.scroll_nonce += 1
    components.html(
        f"""
        <script>
            // run {st.session_state.scroll_nonce}
            const doc = window.parent.document;
            let tries = 0;
            const go = () => {{
                const target = doc.getElementById("latest-answer");
                if (target) {{
                    target.scrollIntoView({{behavior: "smooth", block: "start"}});
                }} else if (tries++ < 30) {{
                    setTimeout(go, 100);
                }}
            }};
            setTimeout(go, 150);
        </script>
        """,
        height=0,
    )


def _render_assistant_turn(content: str, sources: list[dict], symptoms: str) -> None:
    with st.chat_message("assistant"):
        st.markdown(content)
        _render_learn_more(sources, content, symptoms)


def _render_composer(analysis: dict, vectorstore: Chroma | None) -> None:
    """One follow-up box at the bottom of the thread, by voice or text."""
    version = st.session_state.followup_version
    with st.container(border=True):
        st.markdown("**Anything else you'd like to ask?**")
        st.caption(
            "Speak or type. You can say what you've already discussed with a doctor, "
            "what you'd prefer to avoid, or what still doesn't fit."
        )
        _handle_followup_audio(version)
        if st.session_state.followup_transcription_error:
            st.error(st.session_state.followup_transcription_error)
        followup = st.text_area(
            "Your follow-up",
            value=st.session_state.followup_text,
            key=f"followup_draft_{version}_{st.session_state.followup_draft_version}",
            height=100,
            label_visibility="collapsed",
            placeholder="For example: I already discussed painkillers with my doctor and want to avoid anything that makes me drowsy.",
        )
        if st.session_state.followup_error:
            st.error(st.session_state.followup_error)
        if st.button(
            "Send",
            type="primary",
            icon=":material/favorite:",
            disabled=vectorstore is None,
        ):
            message = followup.strip()
            if not message:
                st.warning("Record or type a follow-up before sending it.")
                return
            if vectorstore is None:
                st.error("The health library isn't ready yet.")
                return
            with st.spinner("Thinking about your follow-up..."):
                try:
                    reply, documents = continue_conversation(vectorstore, analysis, message)
                except Exception as exc:
                    st.session_state.followup_error = _ollama_failure_message(exc)
                    st.rerun()
            analysis["thread"].append({"role": "user", "content": message})
            analysis["thread"].append(
                {
                    "role": "assistant",
                    "content": reply,
                    "sources": _plain_sources(documents),
                }
            )
            st.session_state.followup_text = ""
            st.session_state.followup_draft_version += 1
            st.session_state.followup_error = None
            st.session_state.scroll_to_latest = True
            st.rerun()


def _render_conversation(analysis: dict, vectorstore: Chroma | None) -> None:
    """Chat thread: the first question, every answer, and the follow-up box."""
    thread = analysis.setdefault("thread", [])
    symptoms = analysis.get("symptoms", "")
    answer = analysis.get("answer")
    if not answer:
        st.session_state.analysis = None
        st.rerun()

    messages = [
        {"role": "user", "content": symptoms},
        {"role": "assistant", "content": answer, "sources": analysis.get("sources") or []},
        *thread,
    ]
    last_assistant = max(
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    )

    for index, message in enumerate(messages):
        if index == last_assistant:
            _latest_anchor()
        if message.get("role") == "assistant":
            _render_assistant_turn(message.get("content", ""), message.get("sources") or [], symptoms)
        else:
            with st.chat_message("user"):
                st.markdown(message.get("content", ""))

    if st.session_state.scroll_to_latest:
        st.session_state.scroll_to_latest = False
        _scroll_to_latest()

    _render_composer(analysis, vectorstore)


def main() -> None:
    st.set_page_config(
        page_title="MediCare · Women's health companion",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    _init_session_state()

    kb_error = None
    vectorstore = None
    try:
        vectorstore, _stats = load_knowledge_base()
    except Exception as exc:
        kb_error = str(exc)

    _render_sidebar()
    _render_brand(compact=st.session_state.analysis is not None)
    if kb_error:
        _render_kb_problem(kb_error)

    analysis = st.session_state.analysis
    if analysis is None:
        _render_welcome(vectorstore)
    else:
        _render_conversation(analysis, vectorstore)


if __name__ == "__main__":
    main()
