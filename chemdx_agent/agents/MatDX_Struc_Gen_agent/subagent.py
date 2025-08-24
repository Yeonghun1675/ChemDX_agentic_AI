from typing import List, Dict, Any, Optional, TypedDict
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.utils import make_tool_message

import os
import ast
import pandas as pd

# --- pymatgen (structure generation and file export) ---
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pathlib import Path
from pymatgen.io.vasp import Poscar  # for POSCAR export

# =========================
# Load MatDX database
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.abspath(os.path.join(current_dir, "../../databases/MatDX_EF.csv"))

if not os.path.exists(csv_path):
    alt_path = os.path.abspath(os.path.join(current_dir, "../../databases/MatDX_EF.csv"))
    if os.path.exists(alt_path):
        csv_path = alt_path

df = pd.read_csv(csv_path)

# =========================
# Agent configuration
# =========================
name = "MatDX_Struc_Gen_agent"
role = "Generate structure file based on MatDX Database"
context = (
    "You are connected to a MatDX formation energy database with 5,000 materials. "
    "You can search for material chemical formula, space group, lattice parameter, atomic coordinate, and formation energy values. "
    "When handling tabular data, you must use pandas DataFrame (never numpy arrays). "
    "Additionally, you should utilize the pymatgen module to save or export structure files when needed. "
    "Always return structured results."
)

system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Current Role: {role}
Important Context: {context}
"""

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

MatDX_Struc_Gen_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=system_prompt,
)

@MatDX_Struc_Gen_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal=deps.main_task,
        working_memory=deps.working_memory_description,
    )

# -----------------------------

class PoscarResult(TypedDict, total=False):
    ok: bool
    path: Optional[str]
    poscar_content: Optional[str]  # POSCAR content as text
    warnings: List[str]
    error: Optional[str]
    meta: Dict[str, Any]  # e.g., {"formula": "...", "space_group": "...", "id": "..."}

# =========================
# POSCAR utility functions
# =========================
def _write_poscar(structure: Structure, outfile: str) -> None:
    """Write a VASP POSCAR file to the given path."""
    Poscar(structure).write_file(outfile)

def _read_poscar_content(outfile: str) -> str:
    """Read and return the POSCAR file content as plain text."""
    try:
        with open(outfile, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[DFT_POSCAR] Failed to read POSCAR content: {e}")
        return f"Error reading POSCAR content: {str(e)}"

# =========================
# Internal helper: parse structure cell into pymatgen Structure
# =========================
def _parse_structure_cell(cell: Any) -> Optional[Structure]:
    """Parse a structure cell from DB row into pymatgen Structure object."""
    if cell is None:
        return None

    try:
        obj = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except Exception:
        return None

    data_block = None
    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict) and "data" in first and isinstance(first["data"], dict):
            data_block = first["data"]
    elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
        data_block = obj["data"]

    if not data_block:
        return None

    a = data_block.get("a")
    b = data_block.get("b")
    c = data_block.get("c")
    atoms = data_block.get("atoms", [])

    if not (isinstance(a, list) and isinstance(b, list) and isinstance(c, list)):
        return None

    # convert meter → Angstrom
    m2a = 1e10
    aA = [float(a[0]) * m2a, float(a[1]) * m2a, float(a[2]) * m2a]
    bA = [float(b[0]) * m2a, float(b[1]) * m2a, float(b[2]) * m2a]
    cA = [float(c[0]) * m2a, float(c[1]) * m2a, float(c[2]) * m2a]

    lattice = Lattice([aA, bA, cA])

    species: List[str] = []
    coordsA: List[List[float]] = []
    for at in atoms:
        try:
            x = float(at["x"]) * m2a
            y = float(at["y"]) * m2a
            z = float(at["z"]) * m2a
            el = str(at["element"])
        except Exception:
            continue
        species.append(el)
        coordsA.append([x, y, z])

    if not species or not coordsA:
        return None

    try:
        struct = Structure(lattice, species, coordsA, coords_are_cartesian=True)
        return struct
    except Exception:
        return None

# =========================
# Tool: generate and save POSCAR
# =========================
@MatDX_Struc_Gen_agent.tool
async def struct_gen_function(
    ctx: RunContext[AgentState],
    formula: str,
    material_id: Optional[str] = None,
    space_group_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
    preview_lines: int = 60,
    return_text: bool = True
) -> Result:
    """Generate POSCAR file for a given formula (and optional material_id)."""

    for col in ("formula", "id", "space_group", "structure"):
        if col not in df.columns:
            return {"status": "error", "message": f"Required column '{col}' not found in DB."}

    key = formula.strip().lower()
    sub = df[df["formula"].astype(str).str.strip().str.lower() == key]
    # optional space group filtering
    if space_group_filter:
        sg_key = space_group_filter.strip().lower()
        sub_sg = sub[sub["space_group"].astype(str).str.strip().str.lower() == sg_key]
        if not sub_sg.empty:
            sub = sub_sg
    if sub.empty:
        return {"status": "error", "formula": formula, "message": "No exact match found for the given formula."}

    chosen_row = None
    if material_id:
        sub2 = sub[sub["id"].astype(str) == str(material_id)]
        if not sub2.empty:
            # prefer rows we can parse
            for _, r in sub2.iterrows():
                if _parse_structure_cell(r["structure"]) is not None:
                    chosen_row = r
                    break
            if chosen_row is None:
                chosen_row = sub2.iloc[0]
        else:
            return {"status": "error", "formula": formula,
                    "message": f"No entry found for formula='{formula}' with id='{material_id}'."}
    if chosen_row is None:
        # pick the first parseable row; else first row
        for _, r in sub.iterrows():
            if _parse_structure_cell(r["structure"]) is not None:
                chosen_row = r
                break
        if chosen_row is None:
            chosen_row = sub.iloc[0]

    row = chosen_row
    struct = _parse_structure_cell(row["structure"])

    warnings: List[str] = []
    # If DB structure is missing/unparseable, attempt fallback via Materials Project
    if struct is None:
        warnings.append("DB structure missing/unparseable; attempting Materials Project fallback")
        try:
            from chemdx_agent.agents.mat_proj_lookup_agent.subagent import get_best_structure
            spec: Dict[str, Any] = {}
            if space_group_filter:
                spec["spacegroup"] = space_group_filter
            mp_result = get_best_structure(formula, spec or None, payload_format="pmg_dict")
            if mp_result.get("ok"):
                payload_format = mp_result.get("payload_format", "pmg_dict")
                payload = mp_result.get("structure_payload")
                if payload_format == "pmg_dict" and isinstance(payload, dict):
                    try:
                        struct = Structure.from_dict(payload)
                    except Exception:
                        struct = None
                elif payload_format == "cif" and isinstance(payload, dict) and isinstance(payload.get("cif"), str):
                    try:
                        struct = Structure.from_str(payload.get("cif"), fmt="cif")
                    except Exception:
                        struct = None
                if struct is None:
                    return PoscarResult(
                        ok=False,
                        error=f"Failed to build structure from Materials Project payload for {formula}",
                        warnings=warnings
                    )
                # replace row space group with MP metadata if available
                try:
                    sg = str(mp_result.get("meta", {}).get("spacegroup", row.get("space_group", "")))
                except Exception:
                    sg = str(row.get("space_group", ""))
            else:
                return PoscarResult(
                    ok=False,
                    error=f"Materials Project fallback failed: {mp_result.get('error', 'Unknown error')}",
                    warnings=warnings
                )
        except Exception as e:
            return PoscarResult(
                ok=False,
                error=f"Failed to parse DB structure and MP fallback errored: {str(e)}",
                warnings=warnings
            )

    # prepare output directory
    if out_dir is None:
        out_dir = Path(current_dir).joinpath("../../outputs/struc_output").resolve()
    else:
        out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mid = str(row["id"]) if "id" in row else ""
    sg = str(row["space_group"]) if "space_group" in row else locals().get("sg", "")
    safe_formula = "".join(ch for ch in formula if ch.isalnum() or ch in ("_", "-", "+"))
    safe_id = "".join(ch for ch in mid if ch.isalnum() or ch in ("_", "-", "+"))

    # save both POSCAR and CIF files with clean names
    poscar_path = out_dir / f"{safe_formula}.vasp"
    cif_path = out_dir / f"{safe_formula}.cif"
    
    _write_poscar(struct, str(poscar_path))
    CifWriter(struct).write_file(str(cif_path))
    
    poscar_content = _read_poscar_content(str(poscar_path)) if return_text else None
    cif_content = None
    if return_text:
        try:
            with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
                cif_content = f.read()
        except Exception as e:
            logger.error(f"[DFT_POSCAR] Failed to read CIF content: {e}")
            cif_content = f"Error reading CIF content: {str(e)}"

    # build preview
    preview = None
    if poscar_content and return_text:
        lines = poscar_content.splitlines(True)
        preview = "".join(lines[:max(1, int(preview_lines))])

    # log POSCAR content in a clean format
    logger.info(f"[DFT_POSCAR] POSCAR content for {formula}:")
    logger.info(f"Generated POSCAR Files")
    logger.info(f"POSCAR from {formula}")
    logger.info(f"")
    if poscar_content:
        logger.info(poscar_content)
    logger.info(f"Downloaded POSCAR {formula}")

    # Create the main POSCAR output with better formatting
    if poscar_content:
        # Format POSCAR content with proper line breaks and indentation
        lines = poscar_content.strip().split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()  # Remove extra whitespace
            if not line:  # Skip empty lines
                continue
                
            if i == 0:  # Chemical formula line
                formatted_lines.append(f"  {line}")
            elif i == 1:  # Scaling factor
                formatted_lines.append(f"  {line}")
            elif 2 <= i <= 4:  # Lattice vectors
                formatted_lines.append(f"  {line}")
            elif i == 5:  # Element symbols
                formatted_lines.append(f"  {line}")
            elif i == 6:  # Number of atoms
                formatted_lines.append(f"  {line}")
            elif i == 7:  # Coordinate system
                formatted_lines.append(f"  {line}")
            else:  # Atomic coordinates
                formatted_lines.append(f"    {line}")
        
        # Add proper spacing between sections
        formatted_poscar = f"Generated POSCAR Files\nPOSCAR from {formula}\n\n" + '\n'.join(formatted_lines) + f"\n\nDownloaded POSCAR {formula}"
    else:
        formatted_poscar = f"Generated POSCAR Files\nPOSCAR from {formula}\n\n[POSCAR content not available]\n\nDownloaded POSCAR {formula}"
    
    # Return Result type with poscar_content in result field like dft_poscar_agent.py
    from chemdx_agent.schema import Result
    
    return Result(
        action=f"Generated POSCAR and CIF files for {formula} with space group {sg}",
        result=formatted_poscar,  # ✅ 포맷팅된 POSCAR 내용을 result에 직접 출력
        metadata={
            "status": "ok",
            "formula": formula,
            "id": mid,
            "space_group": sg,
            "format": "both",
            "poscar_path": str(poscar_path),
            "cif_path": str(cif_path),
            "poscar_display": f"./outputs/struc_output/{poscar_path.name}",
            "cif_display": f"./outputs/struc_output/{cif_path.name}",
            "poscar_content": poscar_content,
            "cif_content": cif_content,
            "message": "Both POSCAR and CIF files successfully generated.",
            "warnings": warnings,
            "source": "MatDX_DB" if not warnings else "MP_fallback"
        }
    )

# =========================
# Call agent
# =========================
async def call_MatDX_Struc_Gen_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call general agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
    """
    agent_name = "MatDX_Struc_Gen_Agent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}ㅤ")

    user_prompt = f"Current Task of your role: {message2agent}"

    result = await MatDX_Struc_Gen_agent.run(
        user_prompt, deps=deps
    )

    output = result.output

    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()
    logger.info(f"[{agent_name}] Action: {output.action}ㅤ")

    # Log the POSCAR content directly like dft_poscar_agent.py
    if hasattr(output, 'poscar_content') and output.poscar_content:
        logger.info(f"[{agent_name}] POSCAR Content:\n{output.poscar_content}")
    elif hasattr(output, 'result') and output.result:
        logger.info(f"[{agent_name}] Result: {output.result}")

    list_tool_log = make_tool_message(result)
    for log in list_tool_log:
        logger.info(log)

    return output
