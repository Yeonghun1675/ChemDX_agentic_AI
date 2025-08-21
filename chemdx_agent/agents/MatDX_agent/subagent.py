from typing import List
import pandas as pd
import os
import ast
from pymatgen.core import Structure, Lattice
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.utils import make_tool_message

name = "MatDXAgent"
role = "Load MatDX database (MatDX_EF.csv) and refine the DB"
context = (
    "The agent should be able to read a CSV file using pandas and refine it for Machine Learning Model Construction. "
    "When loading or refining, the agent must also output a detailed summary including: "
    "- RAW preview (first few rows), "
    "- SUMMARY (shape, columns, sample rows), "
    "- FORMATION ENERGY SUMMARY (count, mean, min, max)."
)

system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Current Role: {role}
Important Context: {context}
"""

sample_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=system_prompt,
)

# =====================
# Tools
# =====================
@sample_agent.tool_plain
def process_MatDX_DB(
    csv_path: str = "databases/MatDX_EF.csv",
    out_csv: str = "MatDX_EF_Refined.csv",
    mode: str = "load",               # "load" or "refine"
    include_total: bool = False,
    preview_rows: int = 3
) -> str:
    """
    Process the MatDX database.

    mode = "load": 
        - Print raw preview, summary, and formation energy stats.
    mode = "refine": 
        - Print raw preview, summary, formation energy stats, and refine (formula, space_group, FE/atom).
    """
    import pandas as pd
    import ast, os, logging
    from pathlib import Path
    import inspect, chemdx_agent
    import numpy as np

    logger = logging.getLogger(__name__)

    # ---------- resolve CSV path ----------
    pkg_dir = Path(inspect.getfile(chemdx_agent)).resolve().parent
    pkg_root = pkg_dir
    user_p = Path(csv_path).expanduser()
    candidates = [user_p] if user_p.is_absolute() else [
        Path.cwd() / user_p,
        pkg_root / user_p
    ]
    default_pkg_csv = pkg_root / "databases" / "MatDX_EF.csv"
    if default_pkg_csv not in candidates:
        candidates.append(default_pkg_csv)

    chosen = None
    tried = []
    for c in candidates:
        tried.append(str(c))
        if c.exists():
            chosen = c.resolve()
            break
    if chosen is None:
        tried_msg = "\n - " + "\n - ".join(tried)
        raise FileNotFoundError("[MatDX] Could not find CSV. Tried paths:" + tried_msg)

    df = pd.read_csv(chosen)

    # ---------- raw preview ----------
    output = []
    try:
        raw_preview = df.head(preview_rows).to_string(index=False)
        output.append("[RAW PREVIEW]\n" + raw_preview)
    except Exception:
        output.append("[RAW PREVIEW]\n<Could not generate preview>")

    # ---------- summary ----------
    summary = []
    summary.append(f"[SUMMARY] Loaded CSV: {chosen}")
    summary.append(f" - Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    cols_preview = ", ".join(map(str, df.columns[:6]))
    summary.append(f" - Columns: {cols_preview}" + (" ..." if len(df.columns) > 6 else ""))
    try:
        sample = df.head(preview_rows).to_string(index=False)
        summary.append(" - Sample (first rows):\n" + sample)
    except Exception:
        summary.append(" - Could not generate sample")
    output.append("\n".join(summary))

    # ---------- formation energy stats ----------
    fe_values = []
    for idx, row in df.iterrows():
        try:
            fe_dict = ast.literal_eval(str(row["formation_energy"]))
            # 구조에서 natoms 추출
            try:
                struct_first = ast.literal_eval(str(row["structure"]))[0]
                natoms = int(struct_first.get("natoms", 1))
            except Exception:
                natoms = 1

            if "value_per_atom" in fe_dict and fe_dict["value_per_atom"] is not None:
                fe_per_atom = float(fe_dict["value_per_atom"])
            else:
                fe_total = float(fe_dict["value"])
                fe_per_atom = fe_total / max(1, natoms)
            fe_values.append(fe_per_atom)
        except Exception:
            continue

    if fe_values:
        fe_arr = np.array(fe_values)
        stats = [
            "[FORMATION ENERGY SUMMARY]",
            f" - Count: {len(fe_arr)}",
            f" - Mean : {fe_arr.mean():.4f} eV/atom",
            f" - Min  : {fe_arr.min():.4f} eV/atom",
            f" - Max  : {fe_arr.max():.4f} eV/atom",
        ]
        output.append("\n".join(stats))
    else:
        output.append("[FORMATION ENERGY SUMMARY]\n - No valid data found")

    # ---------- refine (if requested) ----------
    if mode.lower() == "refine":
        records = []
        for idx, row in df.iterrows():
            try:
                fe_dict = ast.literal_eval(str(row["formation_energy"]))
                try:
                    struct_first = ast.literal_eval(str(row["structure"]))[0]
                    natoms = int(struct_first.get("natoms", 1))
                except Exception:
                    natoms = 1

                if "value_per_atom" in fe_dict and fe_dict["value_per_atom"] is not None:
                    fe_per_atom = float(fe_dict["value_per_atom"])
                else:
                    fe_total = float(fe_dict["value"])
                    fe_per_atom = fe_total / max(1, natoms)

                rec = {
                    "formula": row.get("formula", ""),
                    "space_group": row.get("space_group", ""),
                    "formation_energy_per_atom": fe_per_atom,
                }
                if include_total:
                    rec["formation_energy_total"] = float(
                        fe_dict.get("value", fe_per_atom * natoms)
                    )
                records.append(rec)

            except Exception as e:
                logger.warning(f"[WARN] row={idx} parse error: {e}")
                continue

        if not records:
            raise ValueError("No valid formation_energy found!")

        refined = pd.DataFrame(records)
        out_csv = os.path.abspath(os.path.expanduser(out_csv))
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        refined.to_csv(out_csv, index=False)

        output.append(f"[INFO] Refined CSV saved: {out_csv} (rows={refined.shape[0]})")

    return "\n\n".join(output)



# =====================
# Agent caller
# =====================
async def call_MatDX_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""
    this agent can:
    - load MatDX_EF.csv and print RAW / SUMMARY / FORMATION ENERGY SUMMARY.
    - refine the DB and save MatDX_EF_Refined.csv with [formula, space_group, formation_energy_per_atom(, formation_energy_total)].
    - handle basic path resolution (CWD / package defaults).

    in this case, use this subagent:
    - if the user asks to build/train/evaluate an ML model → use MLAgent (call_ML_agent). It will auto-run refine if needed.
    - if the user wants information about the MatDX DB (columns, shape, previews, FE stats) → use THIS agent in 'load' mode to read and report.
    - if the user wants to prepare data for ML (preprocessing/refinement) → use THIS agent in 'refine' mode to create MatDX_EF_Refined.csv and report the saved path.
    """
    
    agent_name = name
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}ㅤ")

    user_prompt = f"Current Task of your role: {message2agent}"

    result = await sample_agent.run(user_prompt, deps=deps)

    output = result.output
    
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()
    logger.info(f"[{agent_name}] Action: {output.action}ㅤ")

    list_tool_log = make_tool_message(result)
    for log in list_tool_log:
        logger.info(log)
    
    logger.info(f"[{agent_name}] Result: {output.result}ㅤ")

    return output
