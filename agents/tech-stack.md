# Tech Stack

## Frontend

**Framework:** Next.js 14 (App Router)
**Language:** TypeScript
**Styling:** Tailwind CSS with a custom theme (no component library)
**Animation:** Framer Motion
**Charts:** Recharts (BarChart, PieChart)
**PDF Generation:** jsPDF + jspdf-autotable
**Icons:** lucide-react

### Tailwind Theme (tailwind.config.ts)

Custom colors used throughout the app:
- `background` - #FFFFFF
- `surface` - #F8FAFC (cards, panels)
- `border` - #E2E8F0
- `foreground` - #0F172A
- `gold` - #D97706 (logo accent)
- `red` - #DC2626 (prosecution, violations)
- `blue` - #2563EB (defense)
- `green` - #059669 (pass states)

Custom fonts:
- `font-sans` - Inter (via next/font/google)
- `font-mono` - JetBrains Mono (via next/font/google)

### State Management

No Redux or Zustand. State is managed with:
- React `useState` / `useEffect` for local component state
- `localStorage` for passing data between pages (analysis results, session ID, jury personas, chat state)

Key localStorage keys:
- `trialAnalysis` - Full JSON response from /full-analysis or /upload-and-analyze
- `trialDatasetName` - Display name of the uploaded file
- `trialSessionId` - Backend session ID for retrain/download calls
- `trialChatState` - Saved courtroom messages + charge index for restore
- `trialJury` - 12 jury persona objects

---

## Backend

**Framework:** FastAPI (Python)
**ASGI Server:** Uvicorn
**Virtual Environment:** `.venv` or `venv` folder inside `/fastapi/`

### Python Dependencies (requirements.txt)

```
fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
python-multipart
fairlearn
shap
joblib
httpx
python-dotenv
```

### Key Libraries and What They Do

**fairlearn**
- `MetricFrame` - computes per-group accuracy, selection rate, FPR, FNR
- `demographic_parity_difference` - difference in positive prediction rates across groups
- `equalized_odds_difference` - max difference in TPR/FPR across groups
- `selection_rate`, `false_positive_rate`, `false_negative_rate` - per-group metric functions

**shap**
- `TreeExplainer` - used for tree-based models (Random Forest, XGBoost, LightGBM, Decision Tree)
- `LinearExplainer` - used for linear models (Logistic Regression, Ridge, Lasso)
- `KernelExplainer` - fallback for any other model type, slower but universal

**scikit-learn**
- `LabelEncoder` - encode categorical string columns to integers
- `StandardScaler` - normalize numeric features for logistic regression
- `LogisticRegression` - the COMPAS demo model + legacy upload-and-analyze fallback
- `compute_sample_weight` - used in bias mitigation to reweight samples by group

**joblib** - load user-uploaded .pkl or .joblib model files

**httpx** - async HTTP client for calling Gemini API

**python-dotenv** - loads `.env.local` into environment variables

---

## LLM Providers

Three different LLM APIs are used, each for a different agent role to create diverse reasoning:

| Agent | Model | Provider | API Key |
|---|---|---|---|
| Prosecution | Llama 3.3 70B | Groq | GROQ_API_KEY |
| Defendant (the model) | Llama 3.3 70B | Groq | GROQ_API_KEY |
| Defense | Claude 3 Haiku | OpenRouter | OPENROUTER_API_KEY |
| Judge | Mistral 7B | OpenRouter | OPENROUTER_API_KEY |
| Jury Generation | Llama 3.1 8B | Groq | GROQ_API_KEY |
| Code Analysis | Gemini 2.0 Flash | Google AI | GEMINI_API_KEY |
| Code Modification | Gemini 2.0 Flash | Google AI | GEMINI_API_KEY |

All agent routes in the frontend use **Server-Sent Events (SSE)** for streaming - the LLM response streams token by token into the chat UI in real time.

Gemini calls in the backend are **non-streaming** (request/response) via httpx because they are used for structured JSON output (code analysis) and code generation (retrain script).

---

## Environment Variables

Place these in `/fastapi/.env.local` (loaded by python-dotenv on startup):

```
GROQ_API_KEY=...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
```

Place these in `/.env.local` at the project root for Next.js:

```
FASTAPI_URL=http://localhost:8000
```

If `FASTAPI_URL` is not set, all proxy routes default to `http://localhost:8000`.
