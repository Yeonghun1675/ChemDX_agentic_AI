from typing import Optional, List, Tuple
import os
import re
import difflib
import requests
import pandas as pd

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger


name = "MPStructureAgent"
role = "Given a formula, find matching entries in the DB (Inorganic phosphor/Host), then fetch POSCAR from Materials Project by MP-ID."
context = "DB path resolves from PHOSPHOR_DB_PATH or project root. Materials Project API key from arg, env MP_API_KEY, or built-in default."

# Built-in default API key fallback (provided by user)
DEFAULT_MP_API_KEY = "mZiPJo4FOYWypJE3mSFtM0CK7UmO7jJl"

system_prompt = f"""You are {name}. {role}. {context}"""

mp_structure_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={"temperature": 0.0, "parallel_tool_calls": False},
    system_prompt=system_prompt,
)


_df_cache: Optional[pd.DataFrame] = None
_path_cache: Optional[str] = None


def _resolve_path(file_path: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if file_path:
        candidates.append(file_path)
    if env_path := os.getenv("PHOSPHOR_DB_PATH"):
        candidates.append(env_path)
    candidates.extend([
        "Inorganic_Phosphor_Optical_Properties_DB.csv",
        "data/Inorganic_Phosphor.csv",
        "Inorganic_Phosphor.csv",
    ])
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _find_col(columns: List[str], keywords: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    # exact
    for k in keywords:
        if k.lower() in lower:
            return lower[k.lower()]
    # contains
    for k in keywords:
        kl = k.lower()
        for c in lower:
            if kl in c:
                return lower[c]
    return None


def _get_numeric(row: pd.Series, col: str) -> Optional[float]:
    try:
        return float(pd.to_numeric(row[col])) if col in row and pd.notna(row[col]) else None
    except Exception:
        return None


@mp_structure_agent.tool_plain
def load_db(file_path: Optional[str] = None) -> str:
    """Load the phosphor DB CSV into memory."""
    global _df_cache, _path_cache
    path = _resolve_path(file_path)
    if not path:
        return "Error: Could not resolve DB path. Set PHOSPHOR_DB_PATH or place CSV in project root."
    try:
        _df_cache = pd.read_csv(path)
        _path_cache = path
        return f"Loaded {len(_df_cache)} rows from '{path}'"
    except Exception as exc:
        return f"Error loading DB '{path}': {exc}"


def _normalize_str(s: str) -> str:
    return str(s).strip().lower()


def _extract_first_mpid(value: str) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"mp-\s*\d+", str(value), flags=re.IGNORECASE)
    if match:
        return match.group(0).replace(" ", "").lower()
    return None


@mp_structure_agent.tool_plain
def find_material_mp_id(formula: str, file_path: Optional[str] = None, top_k: int = 5) -> str:
    """Find MP-ID by matching formula against 'Inorganic phosphor' or 'Host'. If no exact match, return similar candidates.
    output: A summary with exact match MP-ID or top-k similar candidates.
    """
    global _df_cache, _path_cache
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_db(file_path)
        if status.startswith("Error"):
            return status
    if _df_cache is None:
        return "Error: DB not loaded"

    cols = list(_df_cache.columns)
    formula_col = _find_col(cols, ["inorganic phosphor", "host", "formula", "compound", "name"])  # primary display
    inorg_col = _find_col(cols, ["inorganic phosphor", "phosphor"])  # search 1
    host_col = _find_col(cols, ["host"])  # search 2
    mp_col = _find_col(cols, ["mp-id", "mp id", "mpid", "mp_id"])  # MP-ID

    if not any([inorg_col, host_col]):
        return "Error: Could not find 'Inorganic phosphor' or 'Host' columns in DB"

    target = _normalize_str(formula)

    # Exact match search on inorganic phosphor then host
    exact_df = pd.DataFrame()
    for col in [inorg_col, host_col]:
        if not col:
            continue
        try:
            tmp = _df_cache[_df_cache[col].astype(str).str.strip().str.lower() == target]
            exact_df = pd.concat([exact_df, tmp]) if not tmp.empty else exact_df
        except Exception:
            pass

    if not exact_df.empty:
        row = exact_df.iloc[0]
        mpid = _extract_first_mpid(str(row.get(mp_col))) if mp_col else None
        if mpid:
            return f"Exact match: {row.get(formula_col) or row.get(inorg_col) or row.get(host_col)} | MP-ID: {mpid}"
        return f"Exact match found but no MP-ID present for '{formula}'"

    # Similarity across both columns
    candidates: List[Tuple[str, float]] = []
    seen = set()
    for col in [inorg_col, host_col]:
        if not col:
            continue
        series = _df_cache[col].astype(str).fillna("")
        for val in series.unique().tolist():
            norm = _normalize_str(val)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            score = difflib.SequenceMatcher(None, target, norm).ratio()
            candidates.append((val, score))

    if not candidates:
        return f"Not found: '{formula}'"

    candidates.sort(key=lambda x: x[1], reverse=True)
    lines = [f"No exact match for '{formula}'. Similar candidates:"]
    for cand, score in candidates[:max(1, int(top_k))]:
        # retrieve first matching row for display
        mask = False
        if inorg_col:
            mask = mask | (_df_cache[inorg_col].astype(str) == str(cand))
        if host_col:
            mask = mask | (_df_cache[host_col].astype(str) == str(cand))
        sub = _df_cache[mask]
        row = sub.iloc[0] if not sub.empty else None
        mpid = _extract_first_mpid(str(row.get(mp_col))) if (row is not None and mp_col) else None
        lines.append(f"- {cand} | similarity={score:.3f} | MP-ID={mpid or 'N/A'}")
    return "\n".join(lines)


def _ensure_poscar_dir() -> str:
    dir_path = os.path.join(os.getcwd(), "POSCAR")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def _save_poscar_text(text: str, base_name: str) -> Tuple[str, str]:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)
    dir_path = _ensure_poscar_dir()
    file_path = os.path.join(dir_path, f"{safe_name}.poscar")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as exc:
        return f"Error saving POSCAR: {exc}", safe_name
    return file_path, safe_name


def _fetch_poscar_via_mpapi(mpid: str, key: str) -> Optional[str]:
    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception:
        return None
    try:
        with MPRester(key) as mpr:
            structure = mpr.get_structure_by_material_id(mpid)
            if structure is None:
                return None
            # to POSCAR string
            poscar_text = structure.to(fmt="poscar")
            return poscar_text
    except Exception:
        return None


def _fetch_poscar_via_legacy(mpid: str, key: str) -> Optional[str]:
    # Legacy REST v2
    # Example endpoint: https://materialsproject.org/rest/v2/materials/mp-149/vasp/poscar
    url = f"https://materialsproject.org/rest/v2/materials/{mpid}/vasp/poscar"
    headers = {"X-API-KEY": key}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            # Try query param style
            alt = f"{url}?API_KEY={key}"
            resp = requests.get(alt, timeout=30)
        if resp.status_code != 200:
            return None
        ct = resp.headers.get("Content-Type", "")
        if "application/json" in ct:
            data = resp.json()
            # Expect {"response": [{"poscar": "..."}]}
            resp_list = data.get("response") if isinstance(data, dict) else None
            if isinstance(resp_list, list) and len(resp_list) > 0:
                item = resp_list[0]
                poscar = item.get("poscar") if isinstance(item, dict) else None
                return poscar
            return None
        # Otherwise assume raw POSCAR
        return resp.text
    except Exception:
        return None


@mp_structure_agent.tool_plain
def fetch_poscar_by_mp_id(mp_id: str, api_key: Optional[str] = None) -> str:
    """Fetch POSCAR from Materials Project by MP-ID. Provide api_key or set env MP_API_KEY. Saves to POSCAR/ directory.
    Strategy: mp-api (preferred) → legacy REST v2 → new API text endpoint.
    """
    key = api_key or os.getenv("MP_API_KEY") or DEFAULT_MP_API_KEY
    if not key:
        return "Error: Missing Materials Project API key. Provide api_key or set MP_API_KEY env var."

    mpid = mp_id.strip()
    if not re.match(r"^mp-\d+$", mpid):
        return f"Error: Invalid MP-ID '{mp_id}'. Expected pattern mp-XXXXX"

    # 1) Try mp-api
    poscar_text = _fetch_poscar_via_mpapi(mpid, key)
    # 2) Fallback: legacy REST v2
    if not poscar_text:
        poscar_text = _fetch_poscar_via_legacy(mpid, key)
    # 3) Last resort: previous text endpoint (kept for compatibility)
    if not poscar_text:
        base = "https://api.materialsproject.org/materials"
        url = f"{base}/{mpid}/structure?format=poscar"
        headers = {"X-API-KEY": key}
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                if "application/json" in resp.headers.get("Content-Type", ""):
                    data = resp.json()
                    poscar_text = data.get("data") if isinstance(data, dict) else None
                else:
                    poscar_text = resp.text
        except Exception:
            poscar_text = None

    if not poscar_text:
        return "Error: POSCAR not found from Materials Project using available methods."

    file_path, _ = _save_poscar_text(poscar_text, mpid)
    if file_path.startswith("Error"):
        return file_path
    preview = "\n".join(poscar_text.splitlines()[:5])
    return f"Saved POSCAR to: {file_path}\nPreview:\n{preview}"


@mp_structure_agent.tool_plain
def poscar_by_formula(formula: str, file_path: Optional[str] = None, api_key: Optional[str] = None, top_k: int = 5) -> str:
    """Find POSCAR for a formula: search DB by 'Inorganic phosphor'/'Host'. If MP-ID exists, fetch POSCAR. If not, report unknown structure.
    If no exact match, show similar candidates.
    """
    global _df_cache, _path_cache
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_db(file_path)
        if status.startswith("Error"):
            return status
    if _df_cache is None:
        return "Error: DB not loaded"

    cols = list(_df_cache.columns)
    inorg_col = _find_col(cols, ["inorganic phosphor", "phosphor"])  # search 1
    host_col = _find_col(cols, ["host"])  # search 2
    mp_col = _find_col(cols, ["mp-id", "mp id", "mpid", "mp_id"])  # MP-ID

    if not any([inorg_col, host_col]):
        return "Error: Could not find 'Inorganic phosphor' or 'Host' columns in DB"

    target = _normalize_str(formula)

    # Exact
    exact_df = pd.DataFrame()
    for col in [inorg_col, host_col]:
        if not col:
            continue
        try:
            tmp = _df_cache[_df_cache[col].astype(str).str.strip().str.lower() == target]
            exact_df = pd.concat([exact_df, tmp]) if not tmp.empty else exact_df
        except Exception:
            pass

    if not exact_df.empty:
        row = exact_df.iloc[0]
        mpid = _extract_first_mpid(str(row.get(mp_col))) if mp_col else None
        if mpid:
            poscar = fetch_poscar_by_mp_id(mpid, api_key)
            if poscar.startswith("Error"):
                return poscar
            return poscar
        return "Structure not available yet (no MP-ID)."

    # Similar
    candidates: List[Tuple[str, float]] = []
    seen = set()
    for col in [inorg_col, host_col]:
        if not col:
            continue
        series = _df_cache[col].astype(str).fillna("")
        for val in series.unique().tolist():
            norm = _normalize_str(val)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            score = difflib.SequenceMatcher(None, target, norm).ratio()
            candidates.append((val, score))

    if not candidates:
        return f"Not found: '{formula}'"

    candidates.sort(key=lambda x: x[1], reverse=True)
    lines = [f"No exact match for '{formula}'. Similar candidates (MP-ID if present):"]
    for cand, score in candidates[:max(1, int(top_k))]:
        mask = False
        if inorg_col:
            mask = mask | (_df_cache[inorg_col].astype(str) == str(cand))
        if host_col:
            mask = mask | (_df_cache[host_col].astype(str) == str(cand))
        sub = _df_cache[mask]
        row = sub.iloc[0] if not sub.empty else None
        mpid = _extract_first_mpid(str(row.get(mp_col))) if (row is not None and mp_col) else None
        lines.append(f"- {cand} | similarity={score:.3f} | MP-ID={mpid or 'N/A'}")
    return "\n".join(lines)


async def call_mp_structure_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call MP structure agent to execute the task: {role}

    args:
        message2agent: (str) Describe exactly the formula and desired output (e.g., POSCAR). If you have an API key, include it or ensure MP_API_KEY is set.
    """
    agent_name = name
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await mp_structure_agent.run(message2agent, deps=deps)
    output = result.output

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")

    return output
