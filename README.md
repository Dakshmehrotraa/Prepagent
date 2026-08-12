# PrepAgent — Build & Deploy Guide

Agentic RAG interview-prep copilot. FastAPI backend (FAISS retrieval + LoRA
DistilBERT classifier + Claude tool-use loop) and a Next.js frontend.

```
prepagent/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entrypoint
│   │   ├── config.py                # env-driven settings
│   │   ├── data/problems.json       # seed dataset (21 problems, 10 patterns)
│   │   ├── retrieval/
│   │   │   ├── build_index.py       # offline: builds FAISS index
│   │   │   └── retriever.py         # runtime: queries FAISS index
│   │   ├── models/classifier.py     # loads LoRA adapter (or keyword fallback)
│   │   ├── training/train_lora_classifier.py  # Colab training script
│   │   ├── agent/
│   │   │   ├── tools.py             # tool schemas + dispatcher
│   │   │   └── orchestrator.py      # the agentic loop itself
│   │   └── routes/api.py            # /api/agent, /api/classify, /api/retrieve
│   ├── requirements.txt
│   ├── Dockerfile
│   └── render.yaml
└── frontend/                        # Next.js 14 + Tailwind, App Router
```

## Part 1 — Run it locally

### 1a. Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build the FAISS index from problems.json (downloads all-MiniLM-L6-v2
# from HuggingFace the first time — needs internet)
python -m app.retrieval.build_index

export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` — you should see `/api/agent`,
`/api/classify`, `/api/retrieve`, `/api/health`.

Without a trained classifier, `/api/classify` automatically falls back to a
keyword heuristic (see `app/models/classifier.py`) so the whole pipeline
works end-to-end before you've trained anything.

### 1b. Frontend

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_BASE=http://localhost:8000" > .env.local
npm run dev
```

Visit `http://localhost:3000`.

## Part 2 — Train the LoRA classifier (Colab)

The seed dataset (21 examples) is enough to prove the training pipeline
works, but too small to actually learn 10 classes well. Before training for
real:

1. **Expand `app/data/problems.json`** to 30–50+ labeled examples per
   pattern. Fastest way: pull problem titles + descriptions from public
   LeetCode pattern lists (e.g. "Grokking the Coding Interview" style
   groupings) and tag each with one of the 10 `PATTERN_LABELS`.
2. In Colab (Runtime → Change runtime type → T4 GPU):

```python
!pip install transformers peft datasets accelerate torch scikit-learn -q
!git clone https://github.com/Dakshmehrotraa/prepagent.git
%cd prepagent/backend/app/training
!python train_lora_classifier.py
```

3. Download the resulting `lora_classifier/` folder and drop it into
   `backend/app/models/lora_classifier/` locally. `classifier.py` will
   auto-detect it and switch from the keyword fallback to the real model.
4. Re-run `python -m app.retrieval.build_index` if you also changed
   `problems.json` (the classifier and the retrieval index are trained/built
   independently — updating one dataset feeds both).

Track metrics the same way you did for the GoEmotions distillation project —
`report_to=["wandb"]` in `TrainingArguments` if you want W&B logging.

## Part 3 — Deploy

### 3a. Backend → Render

Render is used instead of Vercel here because the backend needs a
long-running process (FAISS index + model in memory), not a serverless
function — same constraint you hit with SmartPaste's backend.

1. Push this repo to GitHub.
2. On [render.com](https://render.com) → New → Blueprint → point at your repo
   (it reads `backend/render.yaml`). Or manually: New → Web Service → Docker
   → root dir `backend`.
3. Set the `ANTHROPIC_API_KEY` env var in Render's dashboard (marked
   `sync: false` in render.yaml so it isn't committed).
4. First deploy will build the Docker image, which runs
   `build_index.py` during the build step — so the FAISS index ships baked
   into the container. Health check hits `/api/health`.
5. Note the deployed URL, e.g. `https://prepagent-api.onrender.com`.

**Free-tier note:** Render's free web services spin down after inactivity
and cold-start slowly (30–60s) since the container has to reload the
embedding model into memory. Fine for a portfolio demo; mention this if
asked in an interview, or upgrade to a paid instance to avoid it.

### 3b. Frontend → Vercel

Same flow as your SmartPaste and portfolio deployments:

```bash
cd frontend
npm install -g vercel   # if not already installed
vercel
```

Or via the Vercel dashboard: New Project → import the repo → set root
directory to `frontend` → add env var:

```
NEXT_PUBLIC_API_BASE = https://prepagent-api.onrender.com
```

Redeploy after setting the env var. Also update `ALLOWED_ORIGINS` in
Render's env vars to your final Vercel URL so CORS allows it.

### 3c. Verify end-to-end

```bash
curl -X POST https://prepagent-api.onrender.com/api/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I find the longest substring without repeating characters?"}'
```

You should get back a JSON response with `answer` (Claude's guidance) and
`trace` (the tool calls it made — classify then retrieve, or vice versa).

## Part 4 — What to say on your resume / in interviews

This build gives you concrete, defensible numbers instead of vague claims:

- **Retrieval**: FAISS `IndexFlatIP` over `all-MiniLM-L6-v2` embeddings
  (384-dim), cosine similarity via normalized inner product.
- **Classifier**: LoRA adapters on `distilbert-base-uncased`'s attention
  projections (`q_lin`, `v_lin`), rank 8, targeting 10-way sequence
  classification — report the actual accuracy/F1 your `trainer.evaluate()`
  run produces once you've expanded the dataset and trained for real.
- **Agentic loop**: Claude decides autonomously whether to call
  `classify_pattern`, `retrieve_similar_problems`, both, or neither, based
  on the system prompt — this is genuine tool-use, not a hardcoded
  pipeline, and the trace panel in the UI proves it.
- **Latency**: worth actually measuring once deployed (e.g. `time curl ...`)
  rather than asserting "low-latency" unqualified.

## Next steps you could add

- Cache repeated classify/retrieve calls (same problem asked twice)
- Swap the free Render tier for a paid one to kill cold starts
- Add a `/api/feedback` endpoint to log which suggestions were useful —
  gives you a real fine-tuning signal later
- Expand `problems.json` via scripted scraping rather than by hand
