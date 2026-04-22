# Bias Analysis Pipeline

## How Bias is Detected

The pipeline does not use any hardcoded thresholds or column names to detect bias. It works dynamically on any dataset/model combination by:

1. Loading the user's actual trained model
2. Running it against the dataset to get predictions
3. Grouping predictions by the user-specified sensitive attributes
4. Measuring fairness across those groups using Fairlearn

---

## Step 1: Data Preparation

Function: `prepare_dataset(df, target_column, sensitive_attrs)`

The dataset is processed without any assumptions about column names:
- All NA rows are dropped
- Raw string values of sensitive columns are saved before encoding (needed for Fairlearn which wants readable group labels)
- All object/string/category columns are label-encoded to integers
- The target column is encoded if it is categorical
- Remaining NaN values are filled with column medians

The model then receives the encoded feature matrix X and the raw string sensitive values are passed separately to Fairlearn.

---

## Step 2: Prediction

`model.predict(X)` is called directly on the loaded model.

Smart feature matching handles mismatches between the dataset and model:
1. If the model has `feature_names_in_` (sklearn >= 1.0), it filters X to exactly those columns in that order
2. If the model has `n_features_in_`, it tries numeric-only columns first, then takes the first N columns
3. If neither works, returns a descriptive error

---

## Step 3: Fairlearn Metrics

Function: `compute_fairlearn_metrics(y_true, y_pred, sensitive_features)`

Sensitive features are passed as the raw string values (e.g. "female", "male", "African-American").

If a sensitive column has more than 10 unique values (continuous like age), it is binned into quartiles automatically using pd.qcut.

Metrics computed:
- `demographic_parity_difference`: max difference in positive prediction rates across groups. 0 = perfect parity.
- `equalized_odds_difference`: max difference in true positive rate or false positive rate across groups. 0 = perfect equalization.
- `disparate_impact_ratio`: min group selection rate / max group selection rate. 1.0 = perfect. Below 0.8 = legal 4/5ths rule violation.

Per-group breakdowns also computed: selection_rate, accuracy, false_positive_rate, false_negative_rate.

Legacy scalar values for frontend compatibility:
- `demographic_parity` = 1 - abs(dp_difference), clamped to [0, 1]
- `equal_opportunity` = 1 - abs(eo_difference), clamped to [0, 1]
- `disparate_impact` = the ratio directly

**Verdict rule**: GUILTY if any of the three legacy values < 0.80.

---

## Step 4: SHAP Analysis

Function: `compute_shap_values(model, X, feature_names, max_samples=300)`

Model type is detected by class name to pick the best explainer:

Tree-based models (RandomForest, GradientBoosting, XGBoost, LightGBM, DecisionTree, ExtraTrees, AdaBoost):
- Uses `shap.TreeExplainer` - fast and exact

Linear models (LogisticRegression, LinearRegression, Ridge, Lasso, SGD, ElasticNet):
- Uses `shap.LinearExplainer` with a background sample

All other models:
- Uses `shap.KernelExplainer` with predict_proba as the prediction function
- Slower (model-agnostic) but works universally

SHAP output handling:
- Tree models for binary classification return a 3D array (samples, features, 2 classes) - the last class (positive) is taken
- Multi-class list output takes the last class
- All values are converted to mean absolute SHAP across samples
- Normalized to [0, 1] relative to the highest-importance feature
- Top 10 features returned sorted by importance

---

## Step 5: Proxy Detection

Function: `detect_proxy_features(df, sensitive_col, feature_cols)`

Computes Pearson correlation between each feature and the sensitive column.
All columns are encoded to numeric first.

Features with abs(correlation) > 0.25 are flagged as proxy variables.
Features with correlation > 0.50 are considered high-risk proxies.

Examples: zipcode correlates with race, income correlates with race, name-encoded features can correlate with gender.

---

## Step 6: LLM Code Analysis (optional)

Function: `analyze_code_with_llm(script_content, api_key)`

Sends the training script text to Gemini 2.0 Flash with a prompt asking it to:
- Identify bias-prone coding patterns (no train/test split, using sensitive columns directly, no class balancing, data leakage)
- Assess overall risk level: HIGH, MODERATE, or LOW
- List specific recommendations
- Detect whether the script uses sensitive features directly
- Detect whether the script applies any class balancing

Returns structured JSON. Falls back to an error dict if the API call fails - this never blocks the analysis.

---

## Step 7: Mitigation and Retrain

Function call chain: `/mitigate-and-retrain` endpoint -> `generate_mitigated_code()` -> subprocess execution -> re-evaluation

### Code Generation

`generate_mitigated_code(script_content, analysis_results, dataset_path, target_column, sensitive_attrs, output_model_path, api_key)`

Sends to Gemini 2.0 Flash:
- The full original training script
- The full bias analysis results (metrics, proxies, code issues)
- Exact instructions: read from this path, save model to this path, print JSON accuracy to stdout

Gemini is instructed to apply where appropriate:
- Sample reweighting with `compute_sample_weight`
- Removal or downweighting of proxy features
- `fairlearn.reductions.ExponentiatedGradient` with `DemographicParity` constraint
- Post-processing with `ThresholdOptimizer` if applicable

The same algorithm/model type is preserved when possible.

### Execution

`subprocess.run([python3, modified_script_path], capture_output=True, timeout=120)`

The modified script:
- Must be fully self-contained (all imports included)
- Reads the dataset from the workspace path
- Saves the model to `mitigated_model.pkl` in the workspace
- Prints a JSON line to stdout: `{"accuracy": 0.XX, "status": "success"}`

On success, the retrained model is loaded and evaluated with the same Fairlearn pipeline.

### Re-evaluation

Same `compute_fairlearn_metrics` call on the new model's predictions.
Before/after comparison is computed.
`retrial_passed` = True if all three legacy metrics are >= 0.80 after retraining.
