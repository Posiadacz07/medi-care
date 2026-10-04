# BetweenUs

[![Hackathon](https://img.shields.io/badge/Hackathon-HackYeah_2026-blue)](https://hackyeah.pl/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama_Local-black)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **on-device, privacy-first symptom intake & medical guidance prototype** built for **HackYeah 2026**.

BetweenUs Local records user speech, lets the person review and adjust the transcript, and synthesizes answers using a private, locally running Retrieval-Augmented Generation (RAG) pipeline. **All audio, transcripts, vector embeddings, and LLM reasoning stay strictly on your local machine.**

---

> *MEDICAL DISCLAIMER:**  
> This application is a technical prototype developed for demonstration and hackathon evaluation purposes. It **does not** provide medical advice, diagnosis, or treatment plans, and is **not** a certified medical device. Always consult a qualified healthcare professional with any questions regarding medical conditions.

---

## Key Features

- **100% Local Execution**: Speech-to-text, vector search, and LLM generation run locally on CPU/Apple Silicon.
- **Human-in-the-Loop Transcript Editing**: Review and edit Whisper transcription before any model processes it.
- **RAG Grounding**: Answers are grounded in local guideline PDFs and curated medical QA pairs (ChromaDB).
- **Personalized Context**: Keeps an offline, customizable health profile (age, medications, conditions) for tailored responses.
- **Transparent Citations**: Every response displays verified reference links and knowledge base sources.
- **Zero Audio Leakage**: Audio data is wiped after local processing; no telemetry, no cloud inference.

---

## Architecture & Data Flow

```text
[ Microphone ] ──> [ faster-whisper (CPU) ]
                         │
                         ▼
             [ Human Review & Edit ]
                         │
                         ▼
[ ChromaDB (Local Vector Store) ] ──> [ Context Retrieval ]
[ Local Patient Profile JSON    ]            │
                                             ▼
                                  [ MedGemma via Ollama ]
                                             │
                                             ▼
                                [ Streamlit UI + Source Chips ]
```

---

## Project Structure

```text
betweenus-local/
├── app.py                      # Main Streamlit application and RAG pipeline
├── checked_sources.json        # Curated external reference sources shown in UI
├── requirements.txt            # Python dependencies
├── knowledge_base/             # Drop your domain-specific clinical/guideline PDFs here
│   └── *.pdf
├── chroma_db/                  # Generated local ChromaDB vector store (gitignored)
├── patient_profile.json        # User profile saved locally on disk (gitignored)
├── ingestion_stats.json        # Indexing statistics & row tracking (gitignored)
├── .streamlit/
│   └── config.toml             # Streamlit visual theme settings
└── README.md
```

---

## Knowledge Base

The local vector database merges your custom medical literature with public benchmark datasets:

| Source | Target / Scope | Processing Method |
| :--- | :--- | :--- |
| **Local PDFs** | Any `*.pdf` in `./knowledge_base/` | PyPDFLoader + `RecursiveCharacterTextSplitter` |
| **`proadhikary/MENST`** | `training2K.csv` | Formatted Q&A pairs |
| **`parissharpe/naos-nutrition-training-pairs`** | `naos-training-pairs-v2-1-flat.jsonl` | Patient/Dietitian dialogue turns |
| **`lavita/medical-qa-datasets`** | `medical_meadow_health_advice` | Instruction-following clinical pairs |

Each dataset row is parsed into a LangChain document with the format:  
`Patient Question: <text> Medical Recommendation: <text>`

> **Configuration Tip:** Set `MAX_ROWS_PER_HF_SOURCE` in `app.py` (default: `200`) to increase or decrease the initial embedding index size.

---

## Prerequisites & Setup

### System Requirements

- **OS**: macOS (optimized for Apple Silicon / M-series), Linux, or Windows (WSL2 recommended).
- **Python**: 3.10 or higher.
- **Ollama**: Installed and running locally ([Download Ollama](https://ollama.com)).
- **FFmpeg**: Required for audio recording conversion.
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Posiadacz07/medi-care.git
cd medi-care
```

### 2. Set up virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Pull required local models

Make sure Ollama is installed and running, then pull the LLM and embedding models:

```bash
ollama pull medgemma
ollama pull nomic-embed-text
```

### 4. Run the application

```bash
streamlit run app.py
```

*Note: On the first start, the app will download sample Hugging Face pairs and embed them into `./chroma_db/`. All subsequent runs load immediately.*

---

## How to Use

1. **Set Up Health Profile**: Open the sidebar on your first visit to fill in basic metrics (age, conditions, current medications). Everything remains stored locally in `patient_profile.json`.
2. **Record / Type Symptoms**: Click the microphone icon to record your symptoms in English, or type them directly into the input field.
3. **Verify Transcript**: Edit the transcribed text in the review box if Whisper misheard any terminology.
4. **Get Guidance**: Click **Get guidance** to run vector search and generate an answer grounded in the sources.
5. **Explore References**: Inspect the **Want to learn more?** section at the bottom of the response to see source documents and verified links.

---

## Privacy & Local-First Principles

- **No Remote Audio**: Microphone recordings are written to a temporary local file, processed on CPU by `faster-whisper`, and deleted immediately.
- **Zero Cloud LLM Inference**: Transcripts, profile data, and conversation history are dispatched strictly to `http://localhost:11434` (Ollama).
- **Embedded Storage**: Vector indices are created via ChromaDB embedded directly in `./chroma_db/` without external database services.
- **Internet Usage**: Internet connectivity is required **only** during first-time setup (downloading dependencies, Hugging Face subsets, and Ollama weights).

---

## Configuration & Customization

- **Whisper Model**: By default, faster-whisper runs `base` with `int8` quantization. Update `WHISPER_MODEL_SIZE = "small"` or `"medium"` in `app.py` for higher accuracy.
- **Theme**: Customize styling, fonts, and primary colors in `.streamlit/config.toml`.
- **Custom Documents**: Drop any healthcare whitepaper or medical guideline PDF into `knowledge_base/` and hit **Refresh health library** in the sidebar.

---

## Contributing

Contributions, bug reports, and suggestions are welcome!

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewFeature`).
3. Commit your Changes (`git commit -m 'Add NewFeature'`).
4. Push to the Branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE.txt) for more information.

---

## Acknowledgements

- Built with ❤️ during **HackYeah 2026**.
- [Ollama](https://ollama.com) & [MedGemma](https://deepmind.google/models/gemma/medgemma/) for open medical AI models.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for fast on-device speech-to-text.
- [LangChain](https://www.langchain.com/) & [ChromaDB](https://www.trychroma.com/) for the RAG architecture.
