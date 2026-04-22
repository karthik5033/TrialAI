# TrialAI - Project Overview

## What This Project Is

TrialAI is an adversarial multi-agent AI courtroom that audits machine learning models for bias. It takes a trained model, a dataset, and optionally a training script, then puts the model on trial using multiple LLM agents that debate whether the model is biased.

The platform is built for the hackathon domain: Unbiased AI Decision Making.

## Core Idea

Instead of just printing fairness numbers, TrialAI dramatizes the bias audit as a courtroom trial:

- The Prosecution argues the model is biased using real Fairlearn metrics and SHAP values
- The Defendant (the model itself, voiced by an LLM) responds and defends itself
- The Defense counsel argues for the model's accuracy and business necessity
- The Judge evaluates both sides and delivers a ruling per charge
- A 12-person synthetic jury observes and votes

If the model is found GUILTY, the system automatically rewrites the training script using Gemini, re-executes it with fairness constraints, and produces a mitigated model for download.

## Two Modes

**Demo Mode** (`/demo`)
- Pre-loaded COMPAS criminal justice dataset
- The backend trains a Logistic Regression on startup and serves it
- Judges can see the full courtroom without uploading anything
- COMPAS is used because it is historically proven to be racially biased (ProPublica 2016)

**Custom Trial Mode** (`/trial/upload` -> `/trial/new` -> `/trial/new/verdict`)
- User uploads their own CSV dataset, a trained sklearn-compatible model (.pkl/.joblib), and optionally a training script (.py)
- Backend loads the actual model, runs real Fairlearn metrics, real SHAP values, and LLM code analysis
- Courtroom trial runs with the real data
- Verdict page shows before/after comparison and offers download of the retrained model

## Repository Structure

```
Cephus-new/
  agents/                   -- This documentation folder
  fastapi/
    main.py                 -- FastAPI backend, all endpoints
    utils.py                -- Fairlearn, SHAP, Gemini utility functions
    requirements.txt        -- Python dependencies
    .env.local              -- API keys (not committed)
    workspaces/             -- Per-session uploaded files (auto-created)
  src/app/
    page.tsx                -- Landing page
    demo/
      page.tsx              -- COMPAS demo courtroom
      verdict/page.tsx      -- Demo verdict + PDF
    trial/
      upload/page.tsx       -- 3-file upload wizard (CSV, model, script)
      new/page.tsx          -- Dynamic user trial courtroom
      new/verdict/page.tsx  -- Real verdict, retrain, downloads
      [id]/page.tsx         -- Legacy trial with counterfactual simulator
    api/
      analyze/route.ts
      mitigate/route.ts
      simulate/route.ts
      full-analysis/route.ts
      mitigate-and-retrain/route.ts
      download/[session_id]/[type]/route.ts
      upload-and-analyze/route.ts
      agents/
        prosecution/route.ts
        defense/route.ts
        judge/route.ts
        defendant/route.ts
      generate-jury/route.ts
  testing dataset/
    titanic.csv
    titanic_model.pkl
    train_titanic.py
    adult-all.csv
    pima-indians-diabetes.data.csv
```
