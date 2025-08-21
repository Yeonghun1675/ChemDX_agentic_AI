from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# LightGBM (optional dependency)
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.utils import make_tool_message

# =====================
# Agent meta
# =====================
name = "MLAgent"
role = "Build and evaluate a composition-based ML regression model"
context = (
    "Given a dataset (already loaded/refined by another agent) with chemical formulas (formula) and a target property, "
    "automatically generate composition features (matminer, Magpie preset), split train/val/test (0.80/0.05/0.15), "
    "train LightGBM, Random Forest, Decision Tree, save artifacts, compare, and show only the best model's scatter."
)
system_prompt = f"You are the {name}. Your role: {role}. Context: {context}"

sample_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={"temperature": 0.0, "parallel_tool_calls": False},
    system_prompt=system_prompt,
)

REQUEST_HEADER = "[NEED_ACTION] MatDXAgent.refine_db"

def _mk_refine_request_msg(tried_paths: list[str]) -> str:
    return "\n".join([
        REQUEST_HEADER,
        "reason: Refined CSV not found",
        "expected_output: MatDX_EF_Refined.csv",
        "action: Run MatDXAgent to load & refine DB, then save refined CSV",
        "hint: call_MatDX_agent(ctx, 'Load the MatDX DB and refine it for ML (save MatDX_EF_Refined.csv)')",
        "tried_paths:",
        *[f"  - {p}" for p in tried_paths],
    ])

# =====================
# Path utils
# =====================
def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(_abs(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

def _resolve_csv(csv_path: str | None) -> str:
    from pathlib import Path
    import inspect, chemdx_agent
    candidates = []
    if csv_path:
        p = Path(csv_path).expanduser()
        candidates.append(p if p.is_absolute() else Path.cwd() / p)
    candidates.append(Path.cwd() / "raw_database" / "MatDX_EF_Refined.csv")
    candidates.append(Path.cwd() / "MatDX_EF_Refined.csv")
    pkg_root = Path(inspect.getfile(chemdx_agent)).resolve().parent
    candidates.append(pkg_root / "raw_database" / "MatDX_EF_Refined.csv")
    tried = []
    for c in candidates:
        tried.append(str(c))
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError("\n".join(tried))

# =====================
# Featurizer
# =====================
class MagpieFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, preset: str = "magpie", ignore_errors: bool = True):
        self.preset = preset
        self.ignore_errors = ignore_errors
        self._featurizer = ElementProperty.from_preset(self.preset)
        self._feature_names: list[str] | None = None
    @property
    def feature_names_(self) -> list[str]:
        if self._feature_names is None:
            self._feature_names = self._featurizer.feature_labels()
        return self._feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        comps = [Composition(str(f)) for f in X]
        feat = self._featurizer.featurize_many(comps, ignore_errors=self.ignore_errors)
        arr = np.array(feat, dtype=float)
        self._feature_names = self._featurizer.feature_labels()
        return arr

# =====================
# Tool A) Outlier (single pass)
# =====================
@sample_agent.tool_plain
def analyze_and_clean_outliers(
    csv_path: str | None = None,
    target_col: str = "formation_energy_per_atom",
    iqr_k: float = 1.5,
    formula_col: str = "formula",
    out_clean_csv: str = "MatDX_EF_Refined_clean.csv",
    out_outliers_csv: str = "MatDX_EF_Refined_outliers.csv",
) -> str:
    try:
        resolved = _resolve_csv(csv_path)
    except FileNotFoundError as e:
        return _mk_refine_request_msg(str(e).splitlines())

    df = pd.read_csv(resolved)
    if target_col not in df.columns or formula_col not in df.columns:
        return f"[ERROR] Need columns '{formula_col}' and '{target_col}'. CSV={resolved}"

    y = df[target_col].astype(float)
    q1, q3 = np.percentile(y, [25, 75]); iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    mask = (y >= lo) & (y <= hi)

    df_clean = df.loc[mask].reset_index(drop=True)
    df_out   = df.loc[~mask].reset_index(drop=True)

    clean_abs = _abs(out_clean_csv); _ensure_parent_dir(clean_abs); df_clean.to_csv(clean_abs, index=False)
    out_abs   = _abs(out_outliers_csv); _ensure_parent_dir(out_abs); df_out.to_csv(out_abs, index=False)

    lines = [
        f"[CSV] {resolved}",
        f"[TARGET] {target_col}",
        f"[IQR] Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f}, range=[{lo:.6f}, {hi:.6f}] (k={iqr_k})",
        f"[ROWS] total={len(df)}, kept={len(df_clean)}, outliers={len(df_out)}",
        f"[SAVED] clean -> {clean_abs}",
        f"[SAVED] outliers -> {out_abs}",
    ]
    lines += ["[HEAD of outliers]", df_out[[formula_col, target_col]].head(5).to_string(index=False) if len(df_out) else "  (none)"]
    return "\n".join(lines)

# =====================
# Helpers
# =====================
def _split_80_05_15(Xf: np.ndarray, y: np.ndarray, seed: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(Xf, y, test_size=0.20, random_state=seed, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.75, random_state=seed, shuffle=True)
    return X_train, y_train, X_val, y_val, X_test, y_test

def _metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def _save_scatter(y_true, y_pred, path: str, title: str):
    spath = _abs(path)
    _ensure_parent_dir(spath)
    plt.figure(figsize=(5.6, 5.6), dpi=150)
    plt.scatter(y_true, y_pred, s=12, alpha=0.75, label="Predictions (test)")
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(title)
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(spath, bbox_inches="tight"); plt.close()
    return spath

def _show_png_inline(path: str):
    try:
        from IPython.display import display, Image as IPyImage
        display(IPyImage(filename=_abs(path)))
    except Exception:
        pass

def _fit_one_model(
    mt: str,
    X_tr_i, y_tr, X_val_i, y_val, X_te_i,
    lgbm_params: dict,
    rf_params: dict,
    dt_params: dict,
    random_state: int
) -> Dict[str, Any]:
    if mt == "lgbm":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm not installed")
        base_params = {k: v for k, v in lgbm_params.items() if k != "early_stopping_rounds"}
        model = LGBMRegressor(**base_params, random_state=random_state, n_jobs=-1)
        es_rounds = lgbm_params.get("early_stopping_rounds", 100)
        if lgb is not None and hasattr(lgb, "early_stopping"):
            callbacks = [lgb.early_stopping(es_rounds, verbose=False)]
            model.fit(X_tr_i, y_tr, eval_set=[(X_val_i, y_val)], eval_metric="rmse", callbacks=callbacks)
        else:
            model.fit(X_tr_i, y_tr, eval_set=[(X_val_i, y_val)], eval_metric="rmse",
                      early_stopping_rounds=es_rounds)
        best_iter = getattr(model, "best_iteration_", None) or base_params.get("n_estimators", 1000)
        final_params = {k: v for k, v in base_params.items() if k != "n_estimators"}
        final_params["n_estimators"] = best_iter
        model_final = LGBMRegressor(**final_params, random_state=random_state, n_jobs=-1)
    elif mt == "rf":
        model = RandomForestRegressor(**rf_params, random_state=random_state, n_jobs=-1)
        model.fit(X_tr_i, y_tr)
        model_final = RandomForestRegressor(**rf_params, random_state=random_state, n_jobs=-1)
    else:
        model = DecisionTreeRegressor(**dt_params, random_state=random_state)
        model.fit(X_tr_i, y_tr)
        model_final = DecisionTreeRegressor(**dt_params, random_state=random_state)

    y_tr_pred  = model.predict(X_tr_i)
    y_val_pred = model.predict(X_val_i)
    m_tr  = _metrics(y_tr,  y_tr_pred)
    m_val = _metrics(y_val, y_val_pred)

    X_trval_i = np.vstack([X_tr_i, X_val_i])
    y_trval   = np.concatenate([y_tr, y_val])
    model_final.fit(X_trval_i, y_trval)
    y_te_pred_final = model_final.predict(X_te_i)

    return {
        "model_final": model_final,
        "train_metrics": m_tr,
        "val_metrics": m_val,
        "test_pred": y_te_pred_final,
    }

# =====================
# Tool B) 3-model compare + best-only display
# =====================
@sample_agent.tool_plain
def construct_and_compare_models_MatDX(
    csv_path: str | None = None,
    target_col: str = "formation_energy_per_atom",
    formula_col: str = "formula",
    random_state: int = 42,
    iqr_k: float = 1.5,
    # LGBM
    lgbm_num_leaves: int = 63,
    lgbm_learning_rate: float = 0.05,
    lgbm_n_estimators: int = 2000,
    lgbm_subsample: float = 0.8,
    lgbm_colsample_bytree: float = 0.8,
    lgbm_reg_lambda: float = 0.0,
    lgbm_reg_alpha: float = 0.0,
    lgbm_early_stopping_rounds: int = 100,
    # RF
    rf_n_estimators: int = 500,
    rf_max_depth: Optional[int] = None,
    # DT
    dt_max_depth: Optional[int] = None,
    dt_min_samples_split: int = 2,
    dt_min_samples_leaf: int = 1,
    # outputs
    out_dir: str = ".",
    compare_csv: str = "model_comparison.csv",
    best_summary_txt: str = "best_model_summary.txt",
    show_best_plot: bool = True,
) -> str:
    np.random.seed(random_state)
    try:
        resolved = _resolve_csv(csv_path)
    except FileNotFoundError as e:
        return _mk_refine_request_msg(str(e).splitlines())

    df = pd.read_csv(resolved)
    for col in [formula_col, target_col]:
        if col not in df.columns:
            return f"[ERROR] Missing '{col}' in CSV: {resolved}"

    y_raw = df[target_col].astype(float)
    q1, q3 = np.percentile(y_raw, [25, 75]); iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    mask = (y_raw >= lo) & (y_raw <= hi)
    removed = int((~mask).sum()); kept = int(mask.sum())
    df = df.loc[mask].reset_index(drop=True)

    X_form = df[formula_col].astype(str).values
    y = df[target_col].astype(float).values
    feat = MagpieFeaturizer(preset="magpie", ignore_errors=True)
    X_all = feat.transform(X_form)

    X_tr, y_tr, X_val, y_val, X_te, y_te = _split_80_05_15(X_all, y, seed=random_state)
    imputer = SimpleImputer(strategy="median")
    X_tr_i = imputer.fit_transform(X_tr)
    X_val_i = imputer.transform(X_val)
    X_te_i  = imputer.transform(X_te)

    out_dir_abs = _abs(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)  # 최상위 out_dir 보장
    feat_labels = ElementProperty.from_preset("magpie").feature_labels()

    specs = {
        "lgbm": dict(
            lgbm_params=dict(
                n_estimators=lgbm_n_estimators,
                num_leaves=lgbm_num_leaves,
                learning_rate=lgbm_learning_rate,
                subsample=lgbm_subsample,
                colsample_bytree=lgbm_colsample_bytree,
                reg_lambda=lgbm_reg_lambda,
                reg_alpha=lgbm_reg_alpha,
                early_stopping_rounds=lgbm_early_stopping_rounds,
            ), rf_params={}, dt_params={}
        ),
        "rf": dict(lgbm_params={}, rf_params=dict(n_estimators=rf_n_estimators, max_depth=rf_max_depth), dt_params={}),
        "dt": dict(lgbm_params={}, rf_params={}, dt_params=dict(max_depth=dt_max_depth, min_samples_split=dt_min_samples_split, min_samples_leaf=dt_min_samples_leaf)),
    }
    model_order = ["lgbm", "rf", "dt"] if LGBMRegressor is not None else ["rf", "dt"]

    rows_compare = []
    per_model_summaries = []

    for mt in model_order:
        r = _fit_one_model(mt, X_tr_i, y_tr, X_val_i, y_val, X_te_i,
                           specs[mt]["lgbm_params"], specs[mt]["rf_params"], specs[mt]["dt_params"], random_state)

        m_val = r["val_metrics"]
        m_te  = _metrics(y_te, r["test_pred"])

        # predictions CSV
        pred_path = _abs(os.path.join(out_dir_abs, f"{mt}_model_predictions.csv"))
        _ensure_parent_dir(pred_path)
        pd.DataFrame({
            "split": ["val"]*len(y_val) + ["test"]*len(y_te),
            "y_true": np.concatenate([y_val, y_te]),
            "y_pred": np.concatenate([r["model_final"].predict(X_val_i), r["test_pred"]]),
        }).to_csv(pred_path, index=False)

        # scatter
        scat_path = _abs(os.path.join(out_dir_abs, f"{mt}_model_scatter.png"))
        _save_scatter(y_te, r["test_pred"], scat_path, f"{mt.upper()} (Magpie) • Test (15%)")

        # importances
        imp_path = ""
        if hasattr(r["model_final"], "feature_importances_"):
            importances = r["model_final"].feature_importances_
            order = np.argsort(importances)[::-1]
            topk = min(50, len(importances))
            top_rows = [{"rank": i+1, "feature": feat_labels[idx], "importance": float(importances[idx])}
                        for i, idx in enumerate(order[:topk])]
            imp_path = _abs(os.path.join(out_dir_abs, f"{mt}_model_feature_importances_top.csv"))
            _ensure_parent_dir(imp_path)
            pd.DataFrame(top_rows).to_csv(imp_path, index=False)

        rows_compare.append({
            "model": mt,
            "val_RMSE": m_val["RMSE"], "val_MAE": m_val["MAE"], "val_R2": m_val["R2"],
            "test_RMSE": m_te["RMSE"], "test_MAE": m_te["MAE"], "test_R2": m_te["R2"],
            "scatter_path": scat_path, "pred_path": pred_path, "importance_path": imp_path,
        })
        per_model_summaries.append(f"- {mt.upper()} → Val RMSE={m_val['RMSE']:.4f}, Test RMSE={m_te['RMSE']:.4f}  (scatter: {scat_path})")

    # comparison CSV
    cmp_path = _abs(os.path.join(out_dir_abs, compare_csv))
    _ensure_parent_dir(cmp_path)
    pd.DataFrame(rows_compare).to_csv(cmp_path, index=False)

    # best summary
    best_row = min(rows_compare, key=lambda d: (d["test_RMSE"], -d["test_R2"]))
    best_txt = [
        f"[CSV] {resolved}",
        f"[ROWS] kept={kept}, removed_outliers={removed} (IQR_k={iqr_k})",
        "[SPLIT] train=0.80, val=0.05, test=0.15",
        "[PER-MODEL]",
        *per_model_summaries,
        "",
        f"[BEST] {best_row['model'].upper()}  —  Test RMSE={best_row['test_RMSE']:.6f}, MAE={best_row['test_MAE']:.6f}, R²={best_row['test_R2']:.6f}",
        f"       scatter: {best_row['scatter_path']}",
        f"       preds  : {best_row['pred_path']}",
        (f"       import : {best_row['importance_path']}" if best_row['importance_path'] else "       import : (n/a)"),
        f"[COMPARISON CSV] {cmp_path}",
    ]
    best_path = _abs(os.path.join(out_dir_abs, best_summary_txt))
    _ensure_parent_dir(best_path)
    with open(best_path, "w", encoding="utf-8") as f:
        f.write("\n".join(best_txt))

    print("\n".join(best_txt))
    if show_best_plot:
        _show_png_inline(best_row["scatter_path"])
    return "\n".join(best_txt)

# =====================
# Agent caller
# =====================
async def call_ML_agent(ctx: RunContext[AgentState], message2agent: str):
    agent_name = name
    deps = ctx.deps
    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    user_prompt = f"Current Task of your role: {message2agent}"
    result = await sample_agent.run(user_prompt, deps=deps)
    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()
    logger.info(f"[{agent_name}] Action: {output.action}")
    list_tool_log = make_tool_message(result)
    for log in list_tool_log:
        logger.info(log)
    logger.info(f"[{agent_name}] Result: {output.result}")
    return output
