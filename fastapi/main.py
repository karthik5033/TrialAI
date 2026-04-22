"""TrialAI Bias Analysis Service v3.0 — Real fairness auditing."""

import os
import io
import uuid
import json
import subprocess
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import (
    load_model_file, prepare_dataset, compute_fairlearn_metrics,
    compute_shap_values, detect_proxy_features,
    analyze_code_with_llm, generate_mitigated_code,
    generate_fallback_training_script,
)

# Load environment
load_dotenv(Path(__file__).parent / ".env.local")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WORKSPACE_DIR = Path(__file__).parent / "workspaces"

# ─── Session Store ────────────────────────────────────────────────────────────
sessions: dict = {}

# ─── Global State (COMPAS demo) ──────────────────────────────────────────────
model_state: dict = {}


# ─── Lifespan: Load COMPAS on startup ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    WORKSPACE_DIR.mkdir(exist_ok=True)
    print("[STARTUP] Loading COMPAS dataset...")
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

    try:
        df = pd.read_csv(url)
        keep_cols = ["age", "sex", "race", "priors_count", "days_b_screening_arrest",
                     "c_charge_degree", "two_year_recid", "decile_score"]
        df = df[keep_cols].dropna()
        raw_race = df["race"].copy()

        label_encoders = {}
        for col in ["sex", "race", "c_charge_degree"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        feature_cols = [c for c in df.columns if c != "two_year_recid"]
        X_raw = df[feature_cols].values
        y = df["two_year_recid"].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        accuracy = accuracy_score(y, lr.predict(X))
        print(f"[OK] COMPAS model trained. Accuracy: {accuracy:.4f}")

        model_state.update({
            "model": lr, "scaler": scaler, "df": df, "X": X, "y": y,
            "feature_cols": feature_cols, "label_encoders": label_encoders,
            "raw_race": raw_race, "accuracy": accuracy,
        })
    except Exception as e:
        print(f"[WARN] COMPAS load failed: {e}")

    yield
    model_state.clear()
    sessions.clear()


app = FastAPI(
    title="TrialAI Bias Analysis Service",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ─── Request Models ──────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    dataset_name: str

class MitigateRequest(BaseModel):
    dataset_name: str
    technique: str

class SimulateRequest(BaseModel):
    race: str
    age: int
    prior_arrests: int

class RetrainRequest(BaseModel):
    session_id: str


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPAS DEMO ENDPOINTS (fixed with real Fairlearn)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "dataset_rows": len(model_state.get("df", [])),
        "model_accuracy": round(model_state.get("accuracy", 0), 4),
        "active_sessions": len(sessions),
    }


@app.post("/analyze")
async def analyze_dataset(request: AnalyzeRequest):
    """Analyze COMPAS demo dataset with real Fairlearn + SHAP."""
    lr = model_state["model"]
    X = model_state["X"]
    y = model_state["y"]
    feature_cols = model_state["feature_cols"]
    raw_race = model_state["raw_race"]

    y_pred = lr.predict(X)

    # Real Fairlearn metrics
    fairness = compute_fairlearn_metrics(y, y_pred, raw_race.values)

    # Real SHAP values
    try:
        X_df = pd.DataFrame(X, columns=feature_cols)
        shap_results = compute_shap_values(lr, X_df, feature_cols)
        # Mark proxy features
        for sv in shap_results:
            sv["is_proxy"] = sv["feature"].lower() in ["race", "zipcode", "zip"]
    except Exception as e:
        print(f"[SHAP] Failed for COMPAS: {e}")
        # Fallback to coefficient-based importance
        coefs = np.abs(lr.coef_[0])
        max_c = coefs.max() if coefs.max() > 0 else 1.0
        shap_results = [
            {"feature": col.replace("_", " ").title(),
             "importance": round(float(coefs[i] / max_c), 4),
             "is_proxy": col.lower() in ["race", "zipcode"]}
            for i, col in enumerate(feature_cols)
        ]
        shap_results.sort(key=lambda x: x["importance"], reverse=True)
        shap_results = shap_results[:8]

    race_counts = raw_race.value_counts().to_dict()

    return {
        "dataset_name": request.dataset_name,
        "fairness_metrics": fairness,
        "shap_values": shap_results,
        "demographic_breakdown": race_counts,
        "verdict": "GUILTY" if any(v < 0.80 for k, v in fairness.items()
                                    if k in ("demographic_parity", "equal_opportunity", "disparate_impact")) else "NOT GUILTY",
        "summary": f"Fairlearn + SHAP analysis on {len(X)} samples across {len(feature_cols)} features.",
    }


@app.post("/mitigate")
async def mitigate_bias(request: MitigateRequest):
    """Mitigate COMPAS bias via reweighting."""
    lr_original = model_state["model"]
    X = model_state["X"]
    y = model_state["y"]
    sensitive = model_state["raw_race"].values

    y_pred_before = lr_original.predict(X)
    before_metrics = compute_fairlearn_metrics(y, y_pred_before, sensitive)

    sample_weights = compute_sample_weight(class_weight="balanced", y=sensitive)
    lr_mitigated = LogisticRegression(max_iter=1000, random_state=42)
    lr_mitigated.fit(X, y, sample_weight=sample_weights)

    y_pred_after = lr_mitigated.predict(X)
    after_metrics = compute_fairlearn_metrics(y, y_pred_after, sensitive)

    improvement = {}
    for key in ["demographic_parity", "equal_opportunity", "disparate_impact"]:
        bv = before_metrics.get(key, 0)
        av = after_metrics.get(key, 0)
        improvement[key] = round(((av - bv) / bv) * 100, 1) if bv > 0 else 0.0

    retrial_passed = all(after_metrics.get(k, 0) >= 0.80
                         for k in ("demographic_parity", "equal_opportunity", "disparate_impact"))

    return {
        "dataset_name": request.dataset_name,
        "technique": request.technique,
        "before": {k: before_metrics[k] for k in ("demographic_parity", "equal_opportunity", "disparate_impact")},
        "after": {k: after_metrics[k] for k in ("demographic_parity", "equal_opportunity", "disparate_impact")},
        "improvement": improvement,
        "retrial_passed": retrial_passed,
        "summary": f"Reweighting applied on {len(X)} samples.",
    }


@app.post("/simulate")
async def simulate_prediction(request: SimulateRequest):
    """Counterfactual simulation on COMPAS model."""
    if not model_state:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    lr = model_state["model"]
    scaler = model_state["scaler"]
    feature_cols = model_state["feature_cols"]
    encoders = model_state["label_encoders"]
    df = model_state["df"]

    req_race = request.race
    if req_race == "African American":
        req_race = "African-American"

    base_row = {}
    for col in feature_cols:
        if col == "age":
            base_row[col] = request.age
        elif col == "priors_count":
            base_row[col] = request.prior_arrests
        elif col == "race":
            pass
        else:
            base_row[col] = df[col].mode()[0]

    def predict_for_race(race_val):
        row = base_row.copy()
        try:
            encoded_race = encoders["race"].transform([race_val])[0]
        except Exception:
            encoded_race = df["race"].mode()[0]
        row["race"] = encoded_race
        x_raw = np.array([[row[col] for col in feature_cols]])
        x_scaled = scaler.transform(x_raw)
        pred = lr.predict(x_scaled)[0]
        return "Denied" if pred == 1 else "Approved"

    original_prediction = predict_for_race(req_race)
    counterfactual_prediction = predict_for_race("Caucasian")

    return {
        "original_prediction": original_prediction,
        "counterfactual_prediction": counterfactual_prediction,
        "changed": original_prediction != counterfactual_prediction,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PIPELINE: Full Analysis (user model + dataset + training script)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/full-analysis")
async def full_analysis(
    file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_attributes: str = Form(...),
    training_script: UploadFile = File(None),
):
    """Accept dataset + model + optional training script. Run Fairlearn + SHAP + LLM analysis."""
    session_id = str(uuid.uuid4())[:8]
    workspace = WORKSPACE_DIR / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        # ── Save uploaded files ──
        dataset_path = workspace / "dataset.csv"
        model_path = workspace / f"model{Path(model_file.filename).suffix}"
        dataset_path.write_bytes(await file.read())
        model_path.write_bytes(await model_file.read())

        script_content = None
        if training_script:
            script_path = workspace / "training_script.py"
            script_content = (await training_script.read()).decode("utf-8")
            script_path.write_text(script_content)

        # ── Parse inputs ──
        sensitive_list = [s.strip() for s in sensitive_attributes.split(",")]
        primary_sensitive = sensitive_list[0]

        # ── Load model ──
        model = load_model_file(str(model_path))
        model_type = type(model).__name__
        print(f"[SESSION {session_id}] Loaded model: {model_type}")

        # ── Load dataset ──
        df = pd.read_csv(str(dataset_path))
        if target_column not in df.columns:
            return JSONResponse(
                {"error": f"Target column '{target_column}' not found. Available: {list(df.columns)}"},
                status_code=400,
            )

        X, y, feature_cols, raw_sensitive, encoders = prepare_dataset(df, target_column, sensitive_list)

        if len(X) < 10:
            return JSONResponse({"error": "Dataset has fewer than 10 rows after cleaning."}, status_code=400)

        # ── Find primary sensitive column ──
        sensitive_col_key = next(
            (k for k in raw_sensitive if k.lower() == primary_sensitive.lower()), None
        )
        if not sensitive_col_key:
            return JSONResponse(
                {"error": f"Sensitive attribute '{primary_sensitive}' not found. Available: {list(df.columns)}"},
                status_code=400,
            )
        sensitive_values = raw_sensitive[sensitive_col_key]

        # ── Run predictions (smart feature matching) ──
        try:
            y_pred = model.predict(X.values)
        except Exception as pred_err:
            # Try to match model's expected features
            matched = False
            if hasattr(model, "feature_names_in_"):
                model_features = list(model.feature_names_in_)
                available = [f for f in model_features if f in X.columns]
                if len(available) == len(model_features):
                    X = X[available]
                    feature_cols = available
                    y_pred = model.predict(X.values)
                    matched = True
            
            if not matched and hasattr(model, "n_features_in_"):
                n_expected = model.n_features_in_
                if n_expected <= len(feature_cols):
                    # Try numeric-only columns matching expected count
                    X_numeric = X.select_dtypes(include=[np.number])
                    if len(X_numeric.columns) == n_expected:
                        X = X_numeric
                        feature_cols = list(X_numeric.columns)
                        y_pred = model.predict(X.values)
                        matched = True
                    elif len(X_numeric.columns) > n_expected:
                        # Take first n columns
                        X = X_numeric.iloc[:, :n_expected]
                        feature_cols = list(X.columns)
                        y_pred = model.predict(X.values)
                        matched = True

            if not matched:
                return JSONResponse(
                    {"error": f"Model prediction failed: {str(pred_err)}. Expected features: {getattr(model, 'feature_names_in_', 'unknown')}"},
                    status_code=400,
                )

        accuracy = round(accuracy_score(y, y_pred), 4)

        # ── Fairlearn metrics ──
        fairness = compute_fairlearn_metrics(y, y_pred, sensitive_values)

        # ── SHAP ──
        try:
            shap_results = compute_shap_values(model, X, feature_cols)
        except Exception as e:
            print(f"[SHAP] Failed: {e}")
            shap_results = [{"feature": col, "importance": 0, "raw_shap": 0} for col in feature_cols[:8]]

        # ── Proxy detection ──
        proxies = detect_proxy_features(df, sensitive_col_key, feature_cols)

        # Mark proxy features in SHAP results
        proxy_names = {p["feature"].lower() for p in proxies}
        for sv in shap_results:
            sv["is_proxy"] = sv["feature"].lower().replace(" ", "_") in proxy_names or \
                             sv["feature"].lower() in [s.lower() for s in sensitive_list]

        # ── LLM code analysis ──
        code_analysis = None
        if script_content and GEMINI_API_KEY:
            code_analysis = await analyze_code_with_llm(script_content, GEMINI_API_KEY)

        # ── Demographics ──
        demo_breakdown = pd.Series(sensitive_values).value_counts().to_dict()

        # ── Verdict ──
        verdict = "GUILTY" if any(
            fairness.get(k, 1.0) < 0.80
            for k in ("demographic_parity", "equal_opportunity", "disparate_impact")
        ) else "NOT GUILTY"

        bias_score = round((1 - (fairness["demographic_parity"] + fairness["equal_opportunity"] + fairness["disparate_impact"]) / 3) * 100)

        # ── Store session ──
        session_data = {
            "workspace": str(workspace),
            "dataset_path": str(dataset_path),
            "model_path": str(model_path),
            "target_column": target_column,
            "sensitive_attributes": sensitive_list,
            "script_content": script_content,
            "model_type": model_type,
            "fairness_metrics": fairness,
            "shap_values": shap_results,
            "proxies": proxies,
            "code_analysis": code_analysis,
            "accuracy": accuracy,
            "verdict": verdict,
        }
        sessions[session_id] = session_data

        return {
            "session_id": session_id,
            "dataset_name": file.filename,
            "model_type": model_type,
            "rows": len(X),
            "features": len(feature_cols),
            "model_accuracy": accuracy,
            "target_column": target_column,
            "sensitive_attributes": sensitive_list,
            "fairness_metrics": fairness,
            "shap_values": shap_results,
            "proxy_features": proxies,
            "code_analysis": code_analysis,
            "demographic_breakdown": demo_breakdown,
            "verdict": verdict,
            "bias_score": bias_score,
            "summary": f"Fairlearn + SHAP analysis on {len(X)} samples. Model type: {model_type}.",
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PIPELINE: Mitigate & Retrain
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/mitigate-and-retrain")
async def mitigate_and_retrain(request: RetrainRequest):
    """Use LLM to modify training script, retrain model, re-evaluate."""
    session = sessions.get(request.session_id)
    if not session:
        return JSONResponse({"error": "Session not found. Run /full-analysis first."}, status_code=404)

    if not GEMINI_API_KEY:
        return JSONResponse({"error": "GEMINI_API_KEY not configured."}, status_code=500)

    workspace = Path(session["workspace"])
    output_model_path = str(workspace / "mitigated_model.pkl")

    # ── Determine script to use ──
    script_content = session.get("script_content")
    if not script_content:
        # Generate a fallback training script from the detected model type
        print(f"[SESSION {request.session_id}] No script uploaded, generating fallback for {session['model_type']}")
        script_content = generate_fallback_training_script(
            model_type=session["model_type"],
            dataset_path=str(session["dataset_path"]),
            target_column=session["target_column"],
            sensitive_attrs=session["sensitive_attributes"],
            output_model_path=output_model_path,
        )

    try:
        # ── Generate mitigated code with retry loop ──
        import sys
        venv_python = sys.executable
        python_path = os.environ.get("PYTHON_PATH", venv_python)
        modified_script_path = workspace / "modified_training_script.py"

        modified_code = None
        retrain_success = False
        retrain_stdout = ""
        retrain_stderr = ""
        max_attempts = 3

        for attempt in range(max_attempts):
            if attempt == 0:
                # First attempt: generate from the original script
                modified_code = await generate_mitigated_code(
                    script_content=script_content,
                    analysis_results={
                        "fairness_metrics": session["fairness_metrics"],
                        "proxy_features": session["proxies"],
                        "code_analysis": session["code_analysis"],
                    },
                    dataset_path=str(session["dataset_path"]),
                    target_column=session["target_column"],
                    sensitive_attrs=session["sensitive_attributes"],
                    output_model_path=output_model_path,
                    api_key=GEMINI_API_KEY,
                )
            else:
                # Retry: feed the error back to the LLM to fix its own code
                print(f"[SESSION {request.session_id}] Retry {attempt}/{max_attempts-1} - feeding error back to LLM")
                fix_prompt = f"""The following Python script failed with an error. Fix the script so it runs successfully.

FAILED SCRIPT:
```python
{modified_code}
```

ERROR OUTPUT:
```
{retrain_stderr[-1500:]}
```

REQUIREMENTS:
1. Fix the error while keeping all fairness mitigation logic intact.
2. The dataset is at: "{session["dataset_path"]}"
3. Target column: "{session["target_column"]}"
4. Save the model to: "{output_model_path}" using joblib.dump()
5. Print a JSON line: {{"accuracy": 0.XX, "status": "success"}}
6. Make sure to separate features (X) from target (y) BEFORE defining any ColumnTransformer or feature lists.
7. The script must be fully self-contained and runnable.

Return ONLY the fixed Python script, no explanations or markdown fences."""

                from utils import call_gemini
                modified_code = await call_gemini(fix_prompt, GEMINI_API_KEY)
                modified_code = modified_code.strip()
                if modified_code.startswith("```"):
                    modified_code = modified_code.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            # Write and execute
            modified_script_path.write_text(modified_code)
            print(f"[SESSION {request.session_id}] Attempt {attempt+1}/{max_attempts} - executing with: {python_path}")

            result = subprocess.run(
                [python_path, str(modified_script_path)],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=120,
            )

            retrain_stdout = result.stdout
            retrain_stderr = result.stderr
            retrain_success = result.returncode == 0 and Path(output_model_path).exists()

            print(f"[SESSION {request.session_id}] Attempt {attempt+1} exit={result.returncode}, model_exists={Path(output_model_path).exists()}")
            if retrain_stderr:
                print(f"[SESSION {request.session_id}] STDERR (last 300): {retrain_stderr[-300:]}")

            if retrain_success:
                break

        if not retrain_success:
            return {
                "session_id": request.session_id,
                "retrain_success": False,
                "error": f"Retrain failed after {max_attempts} attempts (exit code {result.returncode})",
                "stdout": retrain_stdout[-2000:] if retrain_stdout else "",
                "stderr": retrain_stderr[-2000:] if retrain_stderr else "",
                "original_script": script_content,
                "modified_script": modified_code,
                "before": {k: float(session["fairness_metrics"][k]) for k in ("demographic_parity", "equal_opportunity", "disparate_impact")},
            }

        # ── Re-evaluate the retrained model ──
        new_model = load_model_file(output_model_path)
        df = pd.read_csv(session["dataset_path"])
        X, y, feature_cols, raw_sensitive, _ = prepare_dataset(
            df, session["target_column"], session["sensitive_attributes"]
        )

        sensitive_col_key = next(
            (k for k in raw_sensitive if k.lower() == session["sensitive_attributes"][0].lower()), None
        )
        sensitive_values = raw_sensitive[sensitive_col_key] if sensitive_col_key else np.zeros(len(y))

        # Smart feature matching (same logic as /full-analysis)
        try:
            y_pred_new = new_model.predict(X.values)
        except Exception:
            matched = False
            if hasattr(new_model, "feature_names_in_"):
                model_features = list(new_model.feature_names_in_)
                available = [f for f in model_features if f in X.columns]
                if len(available) == len(model_features):
                    X = X[available]
                    y_pred_new = new_model.predict(X.values)
                    matched = True

            if not matched and hasattr(new_model, "n_features_in_"):
                n_expected = new_model.n_features_in_
                X_numeric = X.select_dtypes(include=[np.number])
                if len(X_numeric.columns) >= n_expected:
                    X = X_numeric.iloc[:, :n_expected]
                    y_pred_new = new_model.predict(X.values)
                    matched = True

            if not matched:
                # Last resort: try numeric-only
                X_numeric = X.select_dtypes(include=[np.number])
                y_pred_new = new_model.predict(X_numeric.values)

        new_accuracy = round(accuracy_score(y, y_pred_new), 4)
        after_metrics = compute_fairlearn_metrics(y, y_pred_new, sensitive_values)

        # ── Compute improvement ──
        before = {k: float(session["fairness_metrics"][k]) for k in ("demographic_parity", "equal_opportunity", "disparate_impact")}
        after = {k: float(after_metrics[k]) for k in ("demographic_parity", "equal_opportunity", "disparate_impact")}
        improvement = {}
        for key in before:
            bv = before[key]
            av = after[key]
            improvement[key] = round(float(((av - bv) / bv) * 100), 1) if bv > 0 else 0.0

        retrial_passed = all(after.get(k, 0) >= 0.80 for k in after)

        # Update session
        sessions[request.session_id]["mitigated_model_path"] = output_model_path
        sessions[request.session_id]["modified_script"] = modified_code
        sessions[request.session_id]["after_metrics"] = after_metrics

        return {
            "session_id": request.session_id,
            "retrain_success": True,
            "original_script": script_content,
            "modified_script": modified_code,
            "before": before,
            "after": after,
            "improvement": improvement,
            "original_accuracy": float(session["accuracy"]),
            "new_accuracy": float(new_accuracy),
            "retrial_passed": retrial_passed,
            "model_type": type(new_model).__name__,
            "summary": f"Retrained {type(new_model).__name__}. Accuracy: {session['accuracy']} -> {new_accuracy}.",
        }

    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Retrain script timed out (120s limit)."}, status_code=500)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"Mitigation failed: {str(e)}"}, status_code=500)


# ─── Download Endpoints ──────────────────────────────────────────────────────

@app.get("/download/{session_id}/model")
async def download_model(session_id: str):
    session = sessions.get(session_id)
    if not session or not session.get("mitigated_model_path"):
        return JSONResponse({"error": "No mitigated model found."}, status_code=404)
    return FileResponse(session["mitigated_model_path"], filename="mitigated_model.pkl")


@app.get("/download/{session_id}/script")
async def download_script(session_id: str):
    session = sessions.get(session_id)
    if not session or not session.get("modified_script"):
        return JSONResponse({"error": "No modified script found."}, status_code=404)
    workspace = Path(session["workspace"])
    script_path = workspace / "modified_training_script.py"
    if script_path.exists():
        return FileResponse(str(script_path), filename="modified_training_script.py")
    return JSONResponse({"error": "Script file not found."}, status_code=404)


# ─── Legacy endpoint (keep for backwards compat) ─────────────────────────────

@app.post("/upload-and-analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_attributes: str = Form(...),
):
    """Legacy: CSV-only upload. Trains logistic regression internally."""
    try:
        sensitive_list = [s.strip().lower() for s in sensitive_attributes.split(",")]
        primary_sensitive = sensitive_list[0]

        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if target_column not in df.columns:
            return JSONResponse(
                {"error": f"Target column '{target_column}' not found."},
                status_code=400,
            )

        X, y, feature_cols, raw_sensitive, encoders = prepare_dataset(df, target_column, sensitive_list)
        if len(X) < 10:
            return JSONResponse({"error": "Too few rows after cleaning."}, status_code=400)

        # Find sensitive column
        sensitive_col_key = next((k for k in raw_sensitive if k.lower() == primary_sensitive), None)

        # Train logistic regression
        from sklearn.preprocessing import StandardScaler as SS
        scaler = SS()
        X_scaled = scaler.fit_transform(X.values)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)
        accuracy = accuracy_score(y, lr.predict(X_scaled))

        y_pred = lr.predict(X_scaled)
        sensitive_values = raw_sensitive.get(sensitive_col_key, np.zeros(len(y)))
        fairness = compute_fairlearn_metrics(y, y_pred, sensitive_values)

        # SHAP
        try:
            X_df = pd.DataFrame(X_scaled, columns=feature_cols)
            shap_results = compute_shap_values(lr, X_df, feature_cols)
        except Exception:
            coefs = np.abs(lr.coef_[0])
            max_c = coefs.max() if coefs.max() > 0 else 1.0
            shap_results = [
                {"feature": col.replace("_", " ").title(), "importance": round(float(coefs[i] / max_c), 4), "is_proxy": col.lower() in sensitive_list}
                for i, col in enumerate(feature_cols)
            ]
            shap_results.sort(key=lambda x: x["importance"], reverse=True)
            shap_results = shap_results[:8]

        for sv in shap_results:
            if "is_proxy" not in sv:
                sv["is_proxy"] = sv["feature"].lower().replace(" ", "_") in [s.lower() for s in sensitive_list]

        demo_breakdown = pd.Series(sensitive_values).value_counts().to_dict() if sensitive_col_key else {}

        return {
            "dataset_name": file.filename,
            "rows": len(X),
            "features": len(feature_cols),
            "model_accuracy": round(accuracy, 4),
            "target_column": target_column,
            "sensitive_attributes": sensitive_list,
            "fairness_metrics": fairness,
            "shap_values": shap_results,
            "demographic_breakdown": demo_breakdown,
            "verdict": "GUILTY" if any(fairness.get(k, 1) < 0.80 for k in ("demographic_parity", "equal_opportunity", "disparate_impact")) else "NOT GUILTY",
            "summary": f"Analysis on {len(X)} samples across {len(feature_cols)} features.",
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)
