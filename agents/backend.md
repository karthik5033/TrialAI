# Backend Architecture

## Entry Point

`fastapi/main.py` - All FastAPI endpoints are here.
`fastapi/utils.py` - All utility functions used by the endpoints.

Run the backend with:
```bash
./fastapi/venv/bin/uvicorn main:app --reload --port 8000
```

---

## Lifespan (Startup / Shutdown)

The `@asynccontextmanager async def lifespan(app)` function runs when the server starts.

On startup it:
1. Creates the `workspaces/` directory for session file storage
2. Downloads the COMPAS dataset from ProPublica's GitHub CSV
3. Keeps 8 relevant columns: age, sex, race, priors_count, days_b_screening_arrest, c_charge_degree, two_year_recid, decile_score
4. Label-encodes categorical columns (sex, race, charge degree)
5. Trains a LogisticRegression model on the full dataset
6. Stores everything in the global `model_state` dict

On shutdown (after yield): clears `model_state` and `sessions`.

The global `sessions` dict stores per-session data indexed by an 8-char UUID. Each session holds paths to uploaded files, analysis results, and retrain outputs.

---

## Endpoints

### GET /health
Returns server status, number of COMPAS rows loaded, model accuracy, and active session count.

### POST /analyze
Analyzes the pre-loaded COMPAS model.
- Input: `{ "dataset_name": "COMPAS" }`
- Runs predictions on the training data
- Computes real Fairlearn metrics using the raw race column as sensitive feature
- Computes SHAP values (tries LinearExplainer for LogisticRegression, falls back to KernelExplainer)
- Returns fairness_metrics, shap_values, demographic_breakdown, verdict

### POST /mitigate
Retrains the COMPAS model using sample reweighting by race group.
- Input: `{ "dataset_name": "COMPAS", "technique": "reweighting" }`
- Computes before metrics on the original model
- Retrains with `compute_sample_weight(class_weight="balanced", y=sensitive_features)`
- Computes after metrics on the retrained model
- Returns before, after, improvement percentages, retrial_passed

### POST /simulate
Counterfactual prediction for the COMPAS demo.
- Input: `{ "race": "African-American", "age": 25, "prior_arrests": 3 }`
- Predicts outcome for the given race and for "Caucasian" as counterfactual
- Returns original_prediction, counterfactual_prediction, changed (bool)

### POST /full-analysis  <-- MAIN NEW ENDPOINT
Accepts: multipart/form-data with:
- `file` - CSV dataset
- `model_file` - .pkl or .joblib trained model
- `target_column` - string name of target column
- `sensitive_attributes` - comma-separated column names
- `training_script` (optional) - .py file

Processing pipeline:
1. Save files to `workspaces/{session_id}/`
2. Load model via joblib (falls back to pickle)
3. Parse CSV with `prepare_dataset()` which encodes categoricals and preserves raw sensitive values
4. Smart feature matching: tries model.feature_names_in_ first, then numeric column count matching
5. Run model.predict() on the dataset
6. Compute Fairlearn MetricFrame metrics
7. Compute SHAP values with auto-detected explainer
8. Detect proxy features via correlation analysis
9. If training script provided: call Gemini to analyze code for bias patterns
10. Store session data in global `sessions` dict
11. Return full analysis JSON with session_id

Verdict rule: GUILTY if any of demographic_parity, equal_opportunity, or disparate_impact < 0.80

### POST /mitigate-and-retrain  <-- NEW RETRAIN ENDPOINT
Input: `{ "session_id": "abc12345" }`

Processing pipeline:
1. Look up session from `sessions` dict
2. Call `generate_mitigated_code()` - sends original script + analysis to Gemini
3. Gemini rewrites the script to inject fairness constraints (reweighting, proxy removal, ExponentiatedGradient)
4. Save modified script to workspace
5. Execute with `subprocess.run([python3, modified_script_path], timeout=120)`
6. Load the retrained model from `mitigated_model.pkl`
7. Re-run Fairlearn metrics on the new model
8. Compute before/after deltas
9. Return original script, modified script, before, after, improvement, retrial_passed

### GET /download/{session_id}/model
Returns the `mitigated_model.pkl` file from the session workspace as a file download.

### GET /download/{session_id}/script
Returns the `modified_training_script.py` file as a file download.

### POST /upload-and-analyze (Legacy)
CSV-only upload. Trains a LogisticRegression internally. No model file required.
Used by the original upload flow for backwards compatibility.

---

## utils.py Functions

### load_model_file(file_path)
Tries joblib.load first (most common for sklearn), falls back to pickle.load.
Supports .pkl, .joblib, and any extension.

### prepare_dataset(df, target_column, sensitive_attrs)
- Drops NA rows
- Saves raw string values of sensitive columns before encoding
- Encodes all object/category columns with LabelEncoder
- Fills remaining NaN with column medians
- Returns: X (DataFrame), y (array), feature_cols (list), raw_sensitive (dict), encoders (dict)

### compute_fairlearn_metrics(y_true, y_pred, sensitive_features)
- Bins continuous sensitive features (like age) into quartiles if more than 10 unique values
- Computes demographic_parity_difference, equalized_odds_difference via Fairlearn
- Computes selection rates, accuracy, FPR, FNR per group via MetricFrame
- Derives disparate_impact_ratio as min_selection_rate / max_selection_rate
- Returns legacy keys (demographic_parity, equal_opportunity, disparate_impact) for frontend compatibility

Threshold: all three legacy values must be >= 0.80 for NOT GUILTY.

### compute_shap_values(model, X, feature_names, max_samples=300)
- Samples up to 300 rows for speed
- Auto-detects explainer: TreeExplainer for tree models, LinearExplainer for linear, KernelExplainer for all others
- Handles 3D SHAP arrays from binary classifiers (samples, features, classes) by taking the last class slice
- Returns top 10 features sorted by mean absolute SHAP, normalized to [0, 1]

### detect_proxy_features(df, sensitive_col, feature_cols)
- Encodes all columns to numeric
- Computes Pearson correlation between each feature and the sensitive column
- Returns features with abs(correlation) > 0.25, sorted descending

### call_gemini(prompt, api_key, model="gemini-2.0-flash")
- Async POST to Google AI API
- Returns the text response string

### analyze_code_with_llm(script_content, api_key)
- Sends training script to Gemini with a structured prompt
- Asks for JSON with: issues, risk_level (HIGH/MODERATE/LOW), recommendations, uses_sensitive_features, has_class_balancing, model_type_detected
- Parses and returns the JSON, returns error dict on failure

### generate_mitigated_code(script_content, analysis_results, dataset_path, target_column, sensitive_attrs, output_model_path, api_key)
- Sends original script + full analysis context to Gemini
- Instructs Gemini to: apply reweighting, remove proxy features, use ExponentiatedGradient if possible, save model to output_model_path, print JSON accuracy line to stdout
- Returns the modified Python script as a string
- Strips markdown fences if Gemini wraps the code in them
