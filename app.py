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

import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import streamlit as st
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
        "satisfied",
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
        "of the follow-up. End by asking whether this now addresses the concern.\n\n"
        "Health profile:\n{profile}\n\n"
        "Reviewed symptoms:\n{symptoms}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Complete answers already given:\n{complete_answers}\n\n"
        "Earlier follow-ups from the user:\n{earlier_followups}\n\n"
        "User says the latest answer addresses the concern: {satisfied}\n\n"
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


@st.cache_resource(show_spinner="Preparing the local ChromaDB knowledge base...")
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


def continue_conversation(
    vectorstore: Chroma,
    analysis: dict,
    followup: str,
    satisfied: str,
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
        satisfied=satisfied,
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


def _clinical_on_file(profile: dict) -> bool:
    return bool(
        profile.get("last_period_date")
        or profile.get("chronic_conditions")
        or profile.get("medications")
    )


def _load_profile_into_session() -> None:
    """Seed the form once per visit and ask whether the saved details changed."""
    if st.session_state.get("profile_ready"):
        return
    profile = _read_profile_file()
    st.session_state.profile_on_file = profile
    if profile["date_of_birth"]:
        st.session_state.profile_birth_date = date.fromisoformat(profile["date_of_birth"])
    else:
        st.session_state.profile_birth_date = None
    if profile["last_period_date"]:
        st.session_state.profile_period_date = date.fromisoformat(profile["last_period_date"])
    else:
        st.session_state.profile_period_date = None
    st.session_state.profile_conditions = profile["chronic_conditions"]
    st.session_state.profile_medications = profile["medications"]
    # A new visit starts from the saved profile. Editing opens only if they say yes,
    # unless nothing clinical has been saved yet.
    st.session_state.profile_update_choice = "Yes" if not _clinical_on_file(profile) else "No"
    st.session_state.profile_ready = True


def _birth_from_form() -> tuple[str | None, str | None]:
    born = st.session_state.get("profile_birth_date")
    if not isinstance(born, date):
        return None, "Add a date of birth. Age is calculated from it."
    if not _valid_birth_date(born):
        return None, "Date of birth must correspond to an age from 10 to 100."
    return born.isoformat(), None


def _period_from_value(chosen: object) -> tuple[str | None, str | None]:
    if not isinstance(chosen, date):
        return None, "Enter the date of the last period."
    if chosen > date.today():
        return None, "The date of the last period cannot be in the future."
    return chosen.isoformat(), None


def _profile_from_form() -> tuple[dict, str | None, str | None]:
    """Use the saved period, illnesses, and medicines unless this visit updates them."""
    birth_iso, birth_error = _birth_from_form()
    updating = st.session_state.get("profile_update_choice") == "Yes"
    on_file = st.session_state.get("profile_on_file") or _empty_profile()
    if updating:
        period_iso, period_error = _period_from_value(st.session_state.get("profile_period_date"))
        conditions = st.session_state.get("profile_conditions", "").strip()
        medications = st.session_state.get("profile_medications", "").strip()
    else:
        period_iso = on_file.get("last_period_date")
        period_error = None if period_iso else "The saved profile has no last period date. Choose Yes to add it."
        conditions = (on_file.get("chronic_conditions") or "").strip()
        medications = (on_file.get("medications") or "").strip()
    profile = {
        "date_of_birth": birth_iso,
        "last_period_date": period_iso,
        "chronic_conditions": conditions,
        "medications": medications,
        "updated_this_visit": updating,
    }
    return profile, birth_error, period_error


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


def _period_summary(last_period: str | None) -> str:
    if not last_period:
        return "Not recorded"
    period_day = date.fromisoformat(last_period)
    days_ago = (date.today() - period_day).days
    return f"{last_period} ({days_ago} days ago)"


def _init_session_state() -> None:
    defaults = {
        "transcribed_text": "",
        "transcript_version": 0,
        "audio_token": None,
        "transcription_error": None,
        "analysis": None,
        "profile_notice": None,
        "followup_version": 0,
        "followup_text": "",
        "followup_draft_version": 0,
        "followup_audio_token": None,
        "followup_transcription_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    _load_profile_into_session()


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
        _render_source_list(sources)


def _render_profile() -> tuple[dict, str | None, str | None]:
    """Show the saved clinical details and ask whether they should be updated."""
    if st.session_state.pop("profile_saved_collapse", False):
        st.session_state.profile_update_choice = "No"
    st.subheader("Health profile")
    st.caption(
        "Saved on this Mac in `patient_profile.json`. "
        "Age is calculated from the date of birth, so that date does not need to be updated."
    )
    today = date.today()
    st.date_input(
        "Date of birth",
        key="profile_birth_date",
        min_value=_shift_years(today, -100),
        max_value=_shift_years(today, -10),
        help="Saved on this Mac. Age is calculated from this date.",
    )
    born = st.session_state.get("profile_birth_date")
    if isinstance(born, date) and _valid_birth_date(born):
        st.caption(f"Age: {_age_years(born)} years.")

    on_file = st.session_state.get("profile_on_file") or _empty_profile()
    st.markdown("**Period, illnesses, and medications on file**")
    st.markdown(f"- Last period: {_period_summary(on_file.get('last_period_date'))}")
    st.markdown(f"- Illnesses: {on_file.get('chronic_conditions') or 'None recorded'}")
    st.markdown(f"- Medications: {on_file.get('medications') or 'None recorded'}")
    st.radio(
        "Do you want to update your data?",
        ["No", "Yes"],
        key="profile_update_choice",
        horizontal=True,
    )
    if st.session_state.profile_update_choice == "Yes":
        st.date_input(
            "Date of the last period",
            key="profile_period_date",
            max_value=today,
        )
        chosen = st.session_state.get("profile_period_date")
        if isinstance(chosen, date) and chosen <= today:
            st.caption(f"Last period was {(today - chosen).days} days ago.")
        st.text_area(
            "Illnesses",
            key="profile_conditions",
            height=80,
            placeholder="For example: migraine, hypothyroidism, PCOS",
        )
        st.text_area(
            "Medications taken",
            key="profile_medications",
            height=80,
            placeholder="Name and how you take each one, for example: levothyroxine daily",
        )

    profile, birth_error, period_error = _profile_from_form()
    if st.button("Save profile", use_container_width=False):
        if birth_error or (st.session_state.profile_update_choice == "Yes" and period_error):
            st.session_state.profile_notice = None
            st.error(birth_error or period_error)
        else:
            _write_profile_file(profile)
            st.session_state.profile_on_file = {
                "date_of_birth": profile.get("date_of_birth"),
                "last_period_date": profile.get("last_period_date"),
                "chronic_conditions": profile.get("chronic_conditions") or "",
                "medications": profile.get("medications") or "",
            }
            st.session_state.profile_saved_collapse = True
            st.session_state.profile_notice = "Profile saved on this Mac."
            st.rerun()
    if st.session_state.profile_notice and not birth_error:
        st.success(st.session_state.profile_notice)
    return profile, birth_error, period_error


def _ollama_failure_message(exc: Exception) -> str:
    message = str(exc)
    if "11434" in message or "Ollama" in message or "ollama" in message:
        return (
            "MedGemma could not be reached. Confirm Ollama is "
            f"running and `{LLM_MODEL}` is pulled. Details: {message}"
        )
    return message


def _handle_followup_audio(version: int) -> None:
    """Transcribe a follow-up recording into the single review box."""
    audio = st.audio_input("Record a follow-up", key=f"followup_audio_{version}")
    if audio is None:
        return
    token = _audio_token(audio)
    if st.session_state.followup_audio_token == token:
        return
    with st.spinner("Transcribing your follow-up on this Mac..."):
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


def _render_followup(analysis: dict, vectorstore: Chroma | None) -> None:
    """One follow-up box, by voice or text, checked against the full answer."""
    thread = analysis.setdefault("thread", [])
    version = st.session_state.followup_version
    st.divider()
    if thread:
        for index, turn in enumerate(thread):
            role = "user" if turn.get("role") == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(turn.get("content", ""))
                sources = turn.get("sources") or []
                if sources:
                    with st.expander(f"Sources for reply {index // 2 + 1}"):
                        _render_source_list(sources)

    st.subheader("Follow-up")
    satisfied = st.radio(
        "Does this answer address your concern?",
        ["Not yet", "Yes"],
        key=f"concern_{version}_{len(thread)}",
        horizontal=True,
    )
    st.caption(
        "Record or type one message. You can say what you want to avoid, "
        "what you already discussed with a doctor, or what still does not fit. "
        "Check the transcript before you send it. The full answer above is sent with it."
    )
    _handle_followup_audio(version)
    if st.session_state.followup_transcription_error:
        st.error(st.session_state.followup_transcription_error)
    followup = st.text_area(
        "Review and edit your follow-up:",
        value=st.session_state.followup_text,
        key=f"followup_draft_{version}_{st.session_state.followup_draft_version}",
        height=140,
        placeholder="For example: I already discussed painkillers with my doctor, and I want to avoid anything that makes me drowsy.",
    )
    if st.button("Continue conversation", type="primary", disabled=vectorstore is None):
        message = followup.strip()
        if not message:
            st.warning("Record or type a follow-up before sending it.")
            return
        if vectorstore is None:
            st.error("The local knowledge base is not ready.")
            return
        with st.spinner("Continuing with MedGemma on this Mac..."):
            try:
                reply, documents = continue_conversation(
                    vectorstore,
                    analysis,
                    message,
                    satisfied,
                )
            except Exception as exc:
                st.error(_ollama_failure_message(exc))
                return
        thread.append({"role": "user", "content": message})
        thread.append(
            {
                "role": "assistant",
                "content": reply,
                "sources": _plain_sources(documents),
            }
        )
        st.session_state.followup_text = ""
        st.session_state.followup_draft_version += 1
        st.rerun()


def _render_source_list(sources: list[dict]) -> None:
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
        profile, birth_error, period_error = _render_profile()
        st.divider()
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
            if birth_error:
                st.warning(birth_error)
            elif period_error:
                st.warning(period_error)
            elif not symptoms:
                st.warning("Enter or record symptoms before analysis.")
            elif vectorstore is None:
                st.error(kb_error or "The local knowledge base is not ready.")
            else:
                profile_text = _profile_text(profile)
                _write_profile_file(profile)
                st.session_state.profile_on_file = {
                    "date_of_birth": profile.get("date_of_birth"),
                    "last_period_date": profile.get("last_period_date"),
                    "chronic_conditions": profile.get("chronic_conditions") or "",
                    "medications": profile.get("medications") or "",
                }
                st.session_state.profile_notice = "Profile saved on this Mac."
                with st.status("Running local retrieval...", expanded=True) as status:
                    try:
                        status.write(
                            "Embedding the reviewed symptoms and health profile, "
                            "then searching the local collection."
                        )
                        documents = retrieve_context(vectorstore, symptoms, profile_text)
                        status.write(
                            f"Retrieved {len(documents)} chunks from PDFs and "
                            "Hugging Face QA data."
                        )
                        status.write(
                            f"Sending the profile, context, and symptoms to {LLM_MODEL}."
                        )
                        answer = generate_answer(symptoms, profile_text, documents)
                    except Exception as exc:
                        status.update(label="Analysis failed", state="error")
                        st.session_state.analysis = None
                        st.error(_ollama_failure_message(exc))
                    else:
                        status.update(label="Analysis complete", state="complete")
                        st.session_state.followup_version += 1
                        st.session_state.followup_text = ""
                        st.session_state.followup_draft_version += 1
                        st.session_state.followup_audio_token = None
                        st.session_state.followup_transcription_error = None
                        st.session_state.analysis = {
                            "symptoms": symptoms,
                            "profile": profile_text,
                            "answer": answer,
                            "sources": _plain_sources(documents),
                            "thread": [],
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
            st.markdown("**Health profile used**")
            st.text(analysis.get("profile") or "No profile was stored with this analysis.")
            st.markdown("**Reviewed symptoms**")
            st.write(analysis["symptoms"])
            st.markdown("**MedGemma**")
            st.markdown(analysis["answer"])
            _render_sources(analysis["sources"])
            _render_followup(analysis, vectorstore)
            st.caption(
                "Replies are text only. Follow-up recordings are transcribed on this Mac, "
                "then sent only to Ollama on this Mac. Confirm any next step with a licensed clinician."
            )


if __name__ == "__main__":
    main()
