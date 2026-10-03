# MediCare Local

On-device symptom intake for a healthcare hackathon. The app records speech, lets the person correct the transcript, then answers from a private knowledge base. Audio, transcripts, embeddings, and retrieval stay on the Mac.

The assistant does not diagnose or prescribe. It is a prototype for demonstrating local retrieval, not a medical device.

## Goal

Give a clinician or demo judge a local-first flow:

1. The person describes symptoms with the microphone.
2. Whisper transcribes the audio on CPU.
3. The person reviews and edits the text before any model sees it.
4. A single ChromaDB collection, built from local PDFs and Hugging Face question-answer data, supplies the evidence.
5. MedGemma, running in Ollama, writes a text answer grounded in those retrieved chunks.

Patient audio is never sent to a cloud model. The only network use is the first download of public knowledge (Hugging Face datasets and the Whisper weights) and, if you choose, pulling Ollama models.

## Project structure

```text
medi-care/
├── app.py                                              # Streamlit app and RAG pipeline
├── checked_sources.json                                # Checked web pages shown under each answer
├── requirements.txt                                    # Python dependencies
├── knowledge_base/
│   └── sample_clinical_safety_guideline.pdf            # Local PDF source (replace with your guidelines)
├── chroma_db/                                          # Created on first successful index (local)
└── ingestion_stats.json                                # Created after indexing (local)
```

`chroma_db/` and `ingestion_stats.json` are generated at runtime and are gitignored.

## Knowledge base

Both sources are merged into one Chroma collection named `medi_care_knowledge`.

| Source | What is loaded | How it is prepared |
| --- | --- | --- |
| Local PDFs | Every `*.pdf` in `./knowledge_base` | LangChain `PyPDFLoader`, then `RecursiveCharacterTextSplitter` |
| `proadhikary/MENST` | `training2K.csv` | Question and answer formatted as text |
| `parissharpe/naos-nutrition-training-pairs` | `naos-training-pairs-v2-1-flat.jsonl` | User and assistant turns formatted as text |
| `lavita/medical-qa-datasets` | Config `medical_meadow_health_advice` | A sample of instruction / input / output rows |

Each Hugging Face row becomes a LangChain `Document` whose text looks like:

`Patient Question: …. Medical Recommendation: ….`

`MAX_ROWS_PER_HF_SOURCE` in `app.py` defaults to 200 so the first embedding pass finishes on a laptop. Raise it if you want a larger index. A sample PDF is already in `knowledge_base/`; add your own guideline PDFs beside it.

If `./knowledge_base` is missing, or Hugging Face cannot be reached, the app reports that source and continues with whatever else loaded. If nothing loaded, indexing stops with an error instead of building an empty collection.

## Requirements

Hardware target: Apple Silicon Mac (tested against an M3 with 36 GB unified memory).

- Python 3.10 or newer
- [Ollama](https://ollama.com) running locally
- `ffmpeg` recommended, so Whisper can decode microphone audio
- Internet on the first run only

Python packages:

```bash
pip install streamlit chromadb langchain langchain-community langchain-chroma \
    langchain-core langchain-text-splitters pypdf faster-whisper datasets
```

Or:

```bash
pip install -r requirements.txt
```

Local models:

```bash
ollama pull medgemma
ollama pull nomic-embed-text
```

Speech-to-text uses faster-whisper `base` with `device="cpu"` and `compute_type="int8"`. Change `WHISPER_MODEL_SIZE` to `"small"` in `app.py` if you want a more accurate transcript and can spare the extra memory.

## How to run

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve
streamlit run app.py
```

If the Ollama app is already open, you can skip `ollama serve`.

On first launch the app downloads the three Hugging Face sources (up to the row cap), embeds them with `nomic-embed-text`, and writes `./chroma_db`. Later launches reuse that index. After you add PDFs or change the row cap, open **App maintenance** at the bottom of the sidebar and click **Refresh health library**.

### Using the app

The sidebar holds the health profile: age (from date of birth), first day of the last period, health conditions, and medications. Click **Update my profile** to change them. On a first visit the editor opens automatically.

1. On the welcome screen, tap the microphone and speak in English.
2. Check the transcript in **Check what I heard, or type instead**, or type directly.
3. Click **Get guidance**. Retrieval starts only on that click.
4. The conversation opens as a chat. Each reply ends with **Want to learn more?**: source chips linking to checked web pages from `checked_sources.json` and to the knowledge-base documents used for that reply.
5. To continue, speak or type in **Anything else you'd like to ask?** and click **Send**. The page scrolls to the start of the newest reply.
6. **New conversation** clears the thread and returns to the welcome screen.

The color theme lives in `.streamlit/config.toml`.

There is no text-to-speech. The answer is text only.

## Privacy

- Date of birth, last period, illnesses, and medicines are saved only in `patient_profile.json` on this Mac. Age is calculated from the date of birth. The sidebar shows the saved profile on every visit and invites the person to update it. None of this is uploaded.
- Microphone audio is written to a temporary file because faster-whisper needs a path, then the file is deleted.
- The transcript and any follow-up conversation are sent only to Ollama on `localhost`. Follow-up turns stay in the browser session and are not written to disk.
- ChromaDB is an embedded database in `./chroma_db`. It is not a remote server.
- Hugging Face is contacted for public training text, not for patient recordings.
