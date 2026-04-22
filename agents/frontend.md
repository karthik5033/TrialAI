# Frontend Architecture

## Design System

All styling uses Tailwind CSS utility classes. No component library. Theme is in `tailwind.config.ts`.

Core UI patterns:
- Cards: `bg-surface border border-border rounded-xl p-6`
- Primary button: `bg-foreground text-background hover:bg-foreground/90`
- Secondary button: `border border-border hover:bg-surface`
- Section label: `text-xs text-foreground/50 font-bold uppercase`

Standard Framer Motion entry animation:
```tsx
<motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
```

---

## Pages

### `/` (Landing)
Fetches live COMPAS fairness metrics from `/api/analyze` on mount. Displays bias stats and entry points to demo and custom trial.

### `/demo`
COMPAS demo courtroom. Uses static mock data for charges. Streams prosecution, defense, judge agents across 3 charges. 3-column layout with collapsible panels and a bottom jury grid.

### `/demo/verdict`
Fetches from `/api/analyze` and `/api/mitigate`. Shows charges, SHAP chart, before/after mitigation table, PDF download.

### `/trial/upload`
2-step wizard.
- Step 1: Three upload zones - CSV (required), Model .pkl/.joblib (required), Training Script .py (optional)
- Step 2: Data preview, sensitive attribute selector (auto-detected by keyword), target column dropdown
- Submits to `/api/full-analysis` as FormData
- Stores response in `localStorage["trialAnalysis"]` and session_id in `localStorage["trialSessionId"]`
- Redirects to `/trial/new`

### `/trial/new`
Dynamic user trial courtroom. Reads analysis from localStorage on mount. Redirects to upload if not found.

Left panel: dataset name, rows, features, model type, sensitive attributes, animated bias score ring.

Right panel - 5 evidence tabs:
- Fairness: demographic_parity, equal_opportunity, disparate_impact scores with color coding
- Features: SHAP bar chart (red bars = proxy features)
- Code: LLM code analysis risk level, issues list, recommendations list
- Proxies: correlation bars for proxy variable detection
- Counterfactuals: static UX placeholder

Center panel: streaming courtroom chat with 4 agent roles (PROSECUTION, DEFENDANT, DEFENSE, JUDGE).

Bottom panel: 12-person animated jury grid. Generated async via `/api/generate-jury` on mount.

"View Verdict" button at trial end links to `/trial/new/verdict`.

### `/trial/new/verdict`
Reads analysis from localStorage and session_id for retrain/download calls.

Shows:
- Verdict header (GUILTY/NOT GUILTY) with color
- 3 metric charge cards with pass/fail per charge
- SHAP bar chart with proxy features in red
- Code analysis summary if script was uploaded
- Court Reform Order retrain section

Retrain flow: button POSTs `{ session_id }` to `/api/mitigate-and-retrain`. Shows loading state. On success: before/after table, accuracy comparison, modified script preview, download buttons.

Download links: `/api/download/{session_id}/model` and `/api/download/{session_id}/script`.

PDF download via jsPDF + autoTable.

---

## API Routes (Proxy Layer)

All routes in `src/app/api/` proxy to the FastAPI backend.

| Route | Forwards To |
|---|---|
| `/api/analyze` | FastAPI `/analyze` |
| `/api/mitigate` | FastAPI `/mitigate` |
| `/api/simulate` | FastAPI `/simulate` |
| `/api/upload-and-analyze` | FastAPI `/upload-and-analyze` (legacy) |
| `/api/full-analysis` | FastAPI `/full-analysis` |
| `/api/mitigate-and-retrain` | FastAPI `/mitigate-and-retrain` |
| `/api/download/[session_id]/[type]` | FastAPI `/download/{session_id}/{type}` |

Agent routes call LLM APIs directly (not through FastAPI):

| Route | LLM | Provider |
|---|---|---|
| `/api/agents/prosecution` | Llama 3.3 70B | Groq (SSE streaming) |
| `/api/agents/defense` | Claude 3 Haiku | OpenRouter (SSE streaming) |
| `/api/agents/judge` | Mistral 7B | OpenRouter (SSE streaming) |
| `/api/agents/defendant` | Llama 3.3 70B | Groq (SSE streaming) |
| `/api/generate-jury` | Llama 3.1 8B | Groq (JSON response) |

---

## localStorage Keys

| Key | What it holds | Set by | Read by |
|---|---|---|---|
| `trialAnalysis` | Full JSON from /full-analysis | upload page | trial page, verdict page |
| `trialDatasetName` | File name without .csv | upload page | trial page |
| `trialSessionId` | Backend session ID | trial page on load | verdict page |
| `trialChatState` | Messages, chargeIndex, juryState | trial page | trial page (restore) |
| `trialJury` | 12 jury persona objects | trial page | verdict pages |
