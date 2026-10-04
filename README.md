# Speech Emotion Recognition — V3 (simplified) · Group 07 · FP P4

Recognise the emotion in a short voice clip (8 classes: neutral, calm, happy, sad, angry, fearful,
disgust, surprised) from speakers the model has never heard, and serve it as an API + web UI.

**Approach (transfer learning, kept simple):** two frozen pre-trained speech encoders,
**WavLM-Large** and the **Whisper-Medium encoder**, turn each clip into a vector (mean + std of a
few middle layers). One logistic regression per encoder, trained on clean + 20 dB-noise copies,
and the two probability vectors are averaged. No fine-tuning, no stacking, no speaker adaptation.

## Results (same 5 held-out test speakers as V1/V2, 300 clips)

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| Majority class | 13.3 % | 0.029 |
| Hand-crafted MFCC/pitch/loudness + logistic regression | 46.3 % | 0.436 |
| V1 mid-review model (reported) | 59.7 % | 0.575 |
| V2 six-encoder fusion (reported) | 73.3 % | 0.726 |
| WavLM-Large probe | 77.3 % | 0.770 |
| **V3: WavLM-Large + Whisper-Medium (selected on dev CV)** | **78.3 %** | **0.775** |

* 95 % speaker-bootstrap CI for test accuracy: 69.3 % – 87.3 % (only 5 test speakers).
* Leave-one-speaker-out over all 24 speakers: **84.7 %** (V2: 83.9 %). Female 88.1 %, male 81.4 %.
* Development CV (19 speakers, 5 speaker folds): 84.2 %. Noisy test audio (20 dB SNR): 74.7 % (V2: 70.3 %).
* Escalation flag (P(sad+angry+fearful+disgust) ≥ 0.5): recall 95.0 %, precision 87.9 %.
* Auto-tagging at confidence ≥ 0.58 (threshold set on dev CV): 87 % of clips tagged at 82.8 % accuracy.
* V2's 80.2 % "caller adaptation" result needs ~10 earlier clips of the same caller; for a single
  voice memo it falls back to the six-encoder fusion, which V3 beats at a third of the size.

## Folder structure
```
V3_Simple/
├── SER_End_to_End.ipynb     THE notebook: audit → EDA → features → layer scan → model comparison →
│                            one test evaluation → explainability → MLflow registration → demo
├── ser/                     small package shared by the notebook and the API (same code = no skew)
│   ├── config.py            paths, labels, speaker split, encoder names
│   ├── audio.py             decode file/bytes → 16 kHz mono, trim silence, noise augmentation
│   ├── encoders.py          frozen encoder → per-layer mean+std embedding
│   ├── model.py             EmotionModel: fit / predict / save / load
│   └── registry.py          MLflow pyfunc wrapper (notebook only)
├── api/app.py               FastAPI service;  api/static/index.html = record/upload web UI
├── scripts/download_encoders.py   saves both encoders locally (used by the Docker build)
├── Dockerfile, requirements-api.txt   serving image (CPU)
├── requirements.txt         notebook environment
└── artifacts/
    ├── model/ser_model.joblib         production model (0.4 MB of probes; trained on all 24 speakers)
    ├── results/                       every table behind the numbers above (CSV/JSON)
    ├── figures/                       every chart (PNG)
    ├── mlflow.db, mlruns/             local MLflow tracking + model registry (created by the notebook)
    └── cache/                         clip embeddings (created by the notebook, ~600 MB)
```

## Data
RAVDESS speech subset (Livingstone & Russo, 2018, PLoS ONE 13(5): e0196391; CC BY-NC-SA 4.0).
Expected layout `Person01/03-01-02-01-01.wav … Person24/…` (1,440 wav files). The code looks in
`$SER_DATA_DIR`, then `./data`, then `../V1_project/data`.

## Replicate the results (for reviewers)
The notebook is already executed: every output, table and figure can be read without running anything.
To re-run it from scratch:

1. **Python 3.9–3.13**, ~8 GB free disk, internet access (first run only), 8 GB+ RAM.
2. Put the RAVDESS folders `Person01 … Person24` in `./data/` (or set `SER_DATA_DIR=/path/to/folder`).
3. Create the environment and run all cells:
   ```bash
   python -m venv .venv
   source .venv/bin/activate            # Windows: .venv\Scripts\activate
   pip install -r requirements.txt      # CPU/Apple: as is · NVIDIA: install torch from pytorch.org first
   jupyter notebook SER_End_to_End.ipynb   # Kernel -> Restart & Run All
   ```
   Or non-interactively: `jupyter nbconvert --to notebook --execute --inplace SER_End_to_End.ipynb`.

**Run time.** The slow part is embedding 1,439 clips × (clean + noisy) with the two encoders, done once
and cached in `artifacts/cache/`: ≈ 22 min on an Apple M4 Pro GPU, ≈ 30 min on its CPU, roughly
1–1.5 h on a typical laptop CPU, a few minutes on an NVIDIA GPU. The first run also downloads ~4.5 GB of
encoder weights from Hugging Face. Everything after the embeddings (layer scan, model comparison,
test evaluation, LOSO, MLflow) takes ≈ 3 min on a CPU.
**Shortcut:** if you received `artifacts_cache.zip`, unzip it into `artifacts/` first (it creates
`artifacts/cache/`): the notebook then skips the embedding step and runs in ≈ 2–3 min. Only the final
demo cell (section 8, the registered model scoring real audio files) still downloads the two encoders.

**Expected outputs.** `artifacts/results/final_results.json` (test accuracy 0.783, LOSO 0.847), the
tables in `artifacts/results/`, the charts in `artifacts/figures/`, the model
`artifacts/model/ser_model.joblib`, and the MLflow registry `artifacts/mlflow.db`
(`mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db` to browse). Seeds are fixed and a re-run
on the same machine reproduces every number exactly. Embeddings computed on a different device
(CPU vs Apple GPU vs NVIDIA) differ in the last floating-point digits, which can flip a
borderline prediction or two (≈ ±0.3 pt).

## Run the API without Docker
```bash
uvicorn api.app:app --port 8000               # UI: http://localhost:8000   docs: /docs
```

## Docker (host anywhere)
```bash
docker build -t ser-api .                     # ~4 GB image: CPU PyTorch + both encoders baked in
docker run -p 8000:8000 ser-api
```
Needs ≥ 4 GB RAM (both encoders are loaded at start-up, ~40 s). On a 2-vCPU cloud instance expect
roughly 1–3 s per clip. Works on any container host (Render, Railway, Google Cloud Run,
AWS App Runner, Hugging Face Spaces with the Docker SDK). The browser only allows microphone access
on `https://` or `localhost`, so host it behind HTTPS for the Record button.

## API
| Method | Path | Input | Example |
|---|---|---|---|
| GET | `/` | – | the web UI (record a memo or upload a file) |
| GET | `/health` | – | encoders + layer windows |
| POST | `/predict` | multipart file (wav, flac, ogg, mp3; m4a/webm via ffmpeg) | `curl -F "file=@memo.wav" localhost:8000/predict` |
| POST | `/predict/raw` | raw file bytes in the body | `curl --data-binary @memo.wav localhost:8000/predict/raw` |
| POST | `/predict/raw?sample_rate=16000` | headerless 16-bit mono PCM | for streaming/mic clients |

Response:
```json
{"emotion": "angry", "confidence": 0.9963, "negative_score": 0.9973,
 "probabilities": {"neutral": 0.0, "calm": 0.0, "happy": 0.0027, "...": 0.0},
 "duration_s": 2.28, "latency_ms": 449}
```
`negative_score` = probability of sad + angry + fearful + disgust (escalate if ≥ 0.5). Treat
`confidence` < 0.58 as uncertain (route to a human).

## Evaluation protocol and leakage controls
* Speaker-independent split identical to V1/V2: test speakers Person01, 09, 12, 17, 19 are never used
  for training, layer choice or model selection. All choices use 5-fold CV over the other 19 speakers
  (folds grouped by speaker, gender-balanced).
* The encoder pair was chosen from V2's search over all 63 encoder combinations
  (`artifacts/results/v2_encoder_combinations.csv`): every 2+-encoder fusion is within ~1.5 pts on dev
  CV, so the two strongest single encoders from different pre-training families were kept. Because that
  table also shows test scores, the leave-one-speaker-out figure (84.7 %) is the least biased estimate.
* Intensity, statement, repetition and gender are never model inputs. No encoder was trained on RAVDESS.
* One byte-identical duplicate clip (Person07, a development speaker) is removed.
* The production model is refit on all 24 speakers after evaluation (same configuration).

## Limitations
Acted studio speech from 24 North-American actors reading two sentences. Real calls are spontaneous,
noisy and accented, so expect lower accuracy until the probes are refit on labelled in-domain calls
(minutes on a CPU). Accuracy varies by speaker (58–100 % in LOSO) and is lower for male voices.
Use as decision support, not to judge individuals.
