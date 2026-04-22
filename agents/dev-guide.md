# Development Guide

## Running Locally

### 1. Backend

```bash
cd fastapi
./venv/bin/uvicorn main:app --reload --port 8000
```

The server will:
- Download the COMPAS dataset from GitHub on startup (takes a few seconds)
- Print "[OK] COMPAS model trained. Accuracy: 0.6837"
- Be ready at http://localhost:8000

Check it is running: `curl http://localhost:8000/health`

### 2. Frontend

```bash
npm run dev
```

Runs at http://localhost:3000 (or 3001 if 3000 is taken).

The frontend expects the backend at `http://localhost:8000` by default.
To change the backend URL, set `FASTAPI_URL` in a `.env.local` file at the project root.

---

## Environment Files

**`/fastapi/.env.local`** (Python backend, loaded by python-dotenv)
```
GROQ_API_KEY=...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
```

**`/.env.local`** (Next.js frontend, optional)
```
FASTAPI_URL=http://localhost:8000
```

---

## Python Virtual Environment

The project uses a venv at `fastapi/venv/` or `fastapi/.venv/` (both work).

To install all dependencies:
```bash
./fastapi/venv/bin/pip install --only-binary :all: -r fastapi/requirements.txt
```

The `--only-binary :all:` flag is required on Python 3.14 because some packages (scipy) do not have pre-built wheels and cannot be compiled from source.

---

## Testing the Full Pipeline Manually

A test model and training script are included in `testing dataset/`:

```bash
# Step 1: Train the test model
./fastapi/venv/bin/python "testing dataset/train_titanic.py"
# Creates: testing dataset/titanic_model.pkl

# Step 2: Test full-analysis endpoint
curl -X POST http://localhost:8000/full-analysis \
  -F "file=@testing dataset/titanic.csv" \
  -F "model_file=@testing dataset/titanic_model.pkl" \
  -F "training_script=@testing dataset/train_titanic.py" \
  -F "target_column=Survived" \
  -F "sensitive_attributes=Sex" | python3 -m json.tool

# Step 3: Note the session_id from the response, then test retrain
curl -X POST http://localhost:8000/mitigate-and-retrain \
  -H "Content-Type: application/json" \
  -d '{"session_id": "PASTE_SESSION_ID_HERE"}' | python3 -m json.tool
```

Expected results for Titanic + Sex:
- demographic_parity ~0.50 (FAIL - women had ~93% survival, men ~43%)
- equal_opportunity ~0.98 (PASS)
- disparate_impact ~0.46 (FAIL)
- Verdict: GUILTY

---

## Adding a New Agent

1. Create `src/app/api/agents/new-agent/route.ts`
2. Follow the same SSE streaming pattern as prosecution/route.ts
3. Use a different model or provider to maintain agent diversity
4. Add the agent call in `src/app/trial/new/page.tsx` in the `runTrial` function sequence

## Adding a New Fairness Metric

1. Add the computation to `compute_fairlearn_metrics()` in `utils.py`
2. Return it in the response JSON
3. Add the display card to the Fairness tab in `trial/new/page.tsx`
4. Add it to the verdict logic in `/full-analysis` endpoint
5. Add it to the PDF generation in the verdict page

## Adding Support for a New Model Format (e.g. ONNX)

1. Modify `load_model_file()` in `utils.py` to detect the file extension
2. Wrap the ONNX model in a sklearn-compatible wrapper class with a `.predict()` method
3. The SHAP explainer will automatically fall back to KernelExplainer for unknown model types

---

## Known Issues

- **Gemini 429 rate limit**: The free tier Gemini key gets rate-limited quickly. Code analysis and retrain both use Gemini. Space out requests or use a paid key.

- **Session loss on server restart**: Sessions are in-memory. If the FastAPI server restarts, all session data (uploaded files references + analysis) is lost. The workspace files still exist on disk but the session dict is cleared.

- **SHAP on very large datasets**: KernelExplainer with nsamples=100 can be slow on datasets with many features. The max_samples=300 cap on the SHAP sample helps but the Kernel fallback is still the bottleneck.

- **Subprocess Python path**: The retrain subprocess uses `python3` by default. If the user's system has a different Python path, the retrain will fail. The `PYTHON_PATH` environment variable can override this.

---

## File Naming and Code Conventions

**Python (backend)**
- Functions: snake_case
- Files: snake_case
- Endpoint paths: kebab-case (`/full-analysis`, `/mitigate-and-retrain`)
- Print logging format: `[COMPONENT] Message` (e.g. `[SHAP]`, `[PROXY]`, `[SESSION abc123]`)

**TypeScript (frontend)**
- Components: PascalCase
- Functions and variables: camelCase
- Files: kebab-case for pages (Next.js convention), PascalCase for shared components if any
- API routes: all route.ts files export async GET/POST functions

**CSS classes**: always Tailwind utility classes, never inline styles or custom CSS except in `globals.css` for the body base styles.
