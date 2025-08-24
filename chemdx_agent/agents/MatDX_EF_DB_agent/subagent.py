from typing import List, Dict, Any
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.utils import make_tool_message

import os
import ast
import pandas as pd

# =========================
# DB 로드
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.abspath(os.path.join(current_dir, "../../databases/MatDX_EF.csv"))

if not os.path.exists(csv_path):
    alt_path = os.path.abspath(os.path.join(current_dir, "../../databases/MatDX_EF.csv"))
    if os.path.exists(alt_path):
        csv_path = alt_path

df = pd.read_csv(csv_path)

# =========================
# Agent 설정
# =========================
name = "MatDX_EF_DB_agent"
role = "Query MatDX formation energy database"
context = (
    "You are connected to a MatDX formation energy database with 5,000 matreials."
    "You can search for material chemical formula, spacegroup, lattice parameter, atomic coordinate, formation energy (eV/atom), and formation energy (eV)."
)

system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Currunt Role: {role}
Important Context: {context}
"""

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

MatDX_EF_DB_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=system_prompt,
)

@MatDX_EF_DB_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal=deps.main_task,
        working_memory=deps.working_memory_description,
    )

# =========================
# 내부 헬퍼
# =========================
def _to_value_per_atom(x: Any):
    """
    formation_energy 셀에서 value_per_atom(float)만 추출.
    - dict: dict["value_per_atom"]
    - str : ast.literal_eval 후 dict 처리 시도
    - float/int: 그대로 반환
    - 그 외/실패: None
    """
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        v = x.get("value_per_atom")
        return float(v) if isinstance(v, (int, float)) else None
    if isinstance(x, str):
        try:
            obj = ast.literal_eval(x)
            if isinstance(obj, dict):
                v = obj.get("value_per_atom")
                return float(v) if isinstance(v, (int, float)) else None
        except Exception:
            return None
    return None

# =========================
# Tools
# =========================
@MatDX_EF_DB_agent.tool_plain
def Load_MatDX_EF_DB_function(args_1: str, args_2: List[str]) -> str:
    """
    Load the MatDX_EF_DB and return available columns
    """
    if not os.path.exists(csv_path):
        return f"[ERROR] File not found: {csv_path}"
    try:
        _ = pd.read_csv(csv_path)  # 테스트 로드
    except Exception as e:
        return f"[ERROR] Failed to load CSV: {e}"
    columns = list(df.columns)
    return "Available columns: " + ", ".join(columns)

@MatDX_EF_DB_agent.tool_plain
def Find_material_function(limit: int = 10, offset: int = 0) -> dict:
    """
    Return distinct materials as (formula, space_group, id) with PREVIEW ONLY.
    Never returns the full list to avoid context bloat.
    """
    required = ["formula", "space_group", "id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"Required columns not found: {', '.join(missing)}"}

    uniq = (
        df[required]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .assign(label=lambda x: x["formula"] + " (Space Group: " + x["space_group"] + ", ID: " + x["id"] + ")")
    )
    all_labels = uniq["label"].tolist()
    total = len(all_labels)

    start = max(offset, 0)
    end = min(start + max(limit, 0), total)
    examples = all_labels[start:end]

    return {
        "total_materials": total,
        "offset": start,
        "limit": limit,
        "examples": examples,  # 프리뷰만
    }

@MatDX_EF_DB_agent.tool_plain
def get_formation_energy_function(
    formula: str,
    per_group_limit: int = 10,
    include_results: bool = False  # 기본 False: 상세 레코드 미반환
) -> dict:
    """
    Return formation energy per atom for a formula grouped by (space_group, id),
    but ONLY preview per group by default.
    """
    required = ["formula", "space_group", "id", "formation_energy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"Required columns not found in DB: {', '.join(missing)}"}

    key = formula.strip().lower()
    subset = df[df["formula"].astype(str).str.strip().str.lower() == key]
    if subset.empty:
        return {"formula": formula, "message": "No exact match found.", "groups": []}

    groups_out = []
    for (sg, mid), g in subset.groupby(["space_group", "id"], dropna=False):
        # dict/str → value_per_atom(float) 변환
        energies = g["formation_energy"].apply(_to_value_per_atom).dropna().astype(float).tolist()
        if not energies:
            continue

        group_payload = {
            "label": f"{formula} (Space Group: {sg}, ID: {mid})",
            "space_group": sg,
            "id": mid,
            "data_points": len(energies),
            "energies_preview": energies[:per_group_limit],
        }

        if include_results:
            cols = ["formula", "space_group", "id", "formation_energy"]
            cols = [c for c in cols if c in g.columns]
            gp = g[cols].copy()
            gp["formation_energy_eV_per_atom"] = gp["formation_energy"].apply(_to_value_per_atom)
            gp = gp.dropna(subset=["formation_energy_eV_per_atom"])
            group_payload["results_preview"] = (
                gp[["formula", "space_group", "id", "formation_energy_eV_per_atom"]]
                .head(per_group_limit)
                .to_dict(orient="records")
            )

        groups_out.append(group_payload)

    return {"formula": formula, "groups": groups_out}

@MatDX_EF_DB_agent.tool_plain
def get_formation_energy_distribution_function(preview: int = 10) -> dict:
    """
    Overall distribution stats of formation_energy (eV/atom). Compact.
    """
    if "formation_energy" not in df.columns:
        return {"error": "'formation_energy' column not found in DB."}

    s = df["formation_energy"].apply(_to_value_per_atom)
    valid = pd.to_numeric(s, errors="coerce").dropna()
    if valid.empty:
        return {"message": "No formation energy data available."}

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    std = float(valid.std(ddof=1)) if len(valid) > 1 else None

    return {
        "count_valid": int(valid.size),
        "count_missing": int(len(df) - valid.size),
        "min": float(valid.min()),
        "q1": float(q1),
        "median": float(valid.median()),
        "q3": float(q3),
        "iqr": float(iqr),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": std,
        "sample_preview": valid.head(preview).tolist()
    }

@MatDX_EF_DB_agent.tool_plain
def get_materials_overview_function(top_k: int = 10) -> dict:
    """
    Overview of materials & polymorphs (compact counts).
    - Materials: unique formulas
    - Polymorphs: same formula with multiple distinct space_groups
    """
    required = ["formula", "space_group", "id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"Required columns not found: {', '.join(missing)}"}

    formulas = df["formula"].astype(str).str.strip()
    sgs = df["space_group"].astype(str)
    ids = df["id"].astype(str)

    total_rows = int(len(df))
    total_formulas = int(formulas.nunique())
    total_variants_by_sg = int(pd.MultiIndex.from_arrays([formulas, sgs]).nunique())
    total_variants_by_id = int(pd.MultiIndex.from_arrays([formulas, sgs, ids]).nunique())

    # formula별 space_group 개수 → polymorph 판단
    n_sg_by_formula = (
        pd.DataFrame({"formula": formulas, "space_group": sgs})
        .drop_duplicates()
        .groupby("formula")["space_group"]
        .nunique()
        .sort_values(ascending=False)
    )
    polymorph_formulas = int((n_sg_by_formula >= 2).sum())

    # 상위 다형성 예시
    k = max(int(top_k), 1)
    top = n_sg_by_formula.head(k)
    sg_samples = (
        pd.DataFrame({"formula": formulas, "space_group": sgs})
        .drop_duplicates()
        .groupby("formula")["space_group"]
        .apply(lambda s: s.head(3).tolist())
    )

    top_list = []
    for f, n in top.items():
        top_list.append({
            "formula": f,
            "n_space_groups": int(n),
            "example_space_groups": sg_samples.get(f, [])[:3]
        })

    return {
        "total_rows": total_rows,
        "total_formulas": total_formulas,
        "total_variants_by_sg": total_variants_by_sg,
        "total_variants_by_id": total_variants_by_id,
        "polymorph_formulas": polymorph_formulas,
        "top_polymorphs": top_list
    }

    

# =========================
# Call agent
# =========================
async def call_MatDX_EF_DB_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call general agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
    """
    agent_name = "MatDX_EF_DB_agent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}ㅤ")

    user_prompt = f"Current Task of your role: {message2agent}"

    result = await MatDX_EF_DB_agent.run(
        user_prompt, deps=deps
    )

    output = result.output

    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()
    logger.info(f"[{agent_name}] Action: {output.action}ㅤ")

    list_tool_log = make_tool_message(result)
    for log in list_tool_log:
        logger.info(log)

    logger.info(f"[{agent_name}] Result: {output.result}ㅤ")

    return output
