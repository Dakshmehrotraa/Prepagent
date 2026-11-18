# PrepAgent: setup, training and deployment

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

For experiment tracking, set `report_to=["wandb"]` in `TrainingArguments`.

## Part 3 — Deploy

### 3a. Backend → Render

Render is used instead of Vercel for the backend because it needs a
long-running process (FAISS index + model in memory), not a serverless
function.

1. On [render.com](https://render.com) → New → Blueprint → point at your repo
   (it reads `backend/render.yaml`). Or manually: New → Web Service → Docker
   → root dir `backend`.
2. Set the `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`) env var in Render's dashboard (marked
   `sync: false` in render.yaml so it isn't committed).
3. First deploy will build the Docker image, which runs
   `build_index.py` during the build step — so the FAISS index ships baked
   into the container. Health check hits `/api/health`.
4. Note the deployed URL, e.g. `https://prepagent-api.onrender.com`.

**Free-tier note:** Render's free web services spin down after inactivity
and cold-start slowly (30–60s) since the container has to reload the
embedding model into memory. A paid instance avoids this.

### 3b. Frontend → Vercel

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
