
# chemdx_agent/dft_poscar_agent.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal, TypedDict
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger

# Remove circular import - let the main agent handle coordination
# try:
#     from chemdx_agent.agents.mat_proj_lookup_agent.subagent import call_materials_project_agent
# except Exception:
#     call_materials_project_agent = None  # coordinator must inject/handle this

import os
import uuid
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import SupercellTransformation



NAME = "DFT_POSCAR_Agent"
ROLE = "Generate VASP POSCAR files by retrieving authoritative structures from Materials Project via the MaterialsProjectAgent."
CONTEXT = """You must NOT hallucinate structures. Always obtain a structure via the MaterialsProjectAgent.
If multiple candidates are returned, prefer the one with the lowest energy_above_hull unless the user specifies
a space group, crystal system, or mp-id to use. You can convert structures to primitive or conventional cells,
and can optionally build a supercell before writing POSCAR. Always return structured results."""

SYSTEM_PROMPT = f"""You are the {NAME}.
You ONLY generate POSCAR files based on real structures fetched via the MaterialsProjectAgent.
If a request cannot be satisfied (no structure found), return a clear error.
If multiple structures match, apply user constraints; otherwise choose the lowest energy_above_hull.
"""

WORKING_MEMORY_PROMPT = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

poscar_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,          
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=SYSTEM_PROMPT,
)

@poscar_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return WORKING_MEMORY_PROMPT.format(
        main_goal=deps.main_task,
        working_memory=deps.working_memory_description,
    )


# -----------------------------

class PoscarResult(TypedDict, total=False):
    ok: bool
    path: Optional[str]
    warnings: List[str]
    error: Optional[str]
    meta: Dict[str, Any]  # e.g., {"mp_id": "...", "formula_pretty": "...", "spacegroup": "...", "cell": "conventional"}


# -----------------------------

def _outfile_path(prefix: str = "POSCAR") -> str:
    # Always write file named POSCAR in a dedicated folder with unique run-id next to it
    run_dir = Path("poscars") / f"{prefix}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir / "POSCAR")

def _to_target_cell(structure: Structure, cell: Literal["primitive", "conventional"]) -> Structure:
    if cell == "primitive":
        return SpacegroupAnalyzer(structure, symprec=1e-3).get_primitive_standard_structure()
    # conventional (standard) setting:
    return SpacegroupAnalyzer(structure, symprec=1e-3).get_conventional_standard_structure()

def _apply_supercell(structure: Structure, supercell: Optional[List[List[int]]]) -> Structure:
    if not supercell:
        return structure
    transf = SupercellTransformation(supercell)
    return transf.apply_transformation(structure)

def _structure_summary(structure: Structure) -> Dict[str, Any]:
    sga = SpacegroupAnalyzer(structure, symprec=1e-3)
    spg = sga.get_space_group_symbol()
    num_sites = len(structure)
    a, b, c = structure.lattice.a, structure.lattice.b, structure.lattice.c
    alpha, beta, gamma = structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma
    return {
        "spacegroup": spg,
        "num_sites": num_sites,
        "lattice": {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma},
        "formula_pretty": structure.composition.reduced_formula,
    }

def _write_poscar(structure: Structure, outfile: str) -> None:
    Poscar(structure).write_file(outfile)


# -----------------------------

@poscar_agent.tool_plain
def generate_poscar_from_material(
    material: str,
    specification: Optional[Dict[str, Any]] = None,
    cell: Literal["primitive", "conventional"] = "conventional",
    supercell: Optional[List[List[int]]] = None,
    title: Optional[str] = None
) -> PoscarResult:
    """
    Fetch a structure via MaterialsProjectAgent and write a POSCAR after optional cell conversion and supercelling.

    Args:
        material: free-form query like "Bi2Te3", "Silicon", or an mp-id "mp-149".
        specification: optional constraints to help disambiguate results, e.g.:
            {
              "spacegroup": "R-3m",
              "crystal_system": "trigonal",
              "mp_id": "mp-1234"
            }
        cell: choose "primitive" or "conventional" cell in output.
        supercell: optional 3x3 integer matrix, e.g. [[2,0,0],[0,2,0],[0,0,1]]
        title: optional title for metadata (not used in POSCAR text; VASP standard file is named 'POSCAR').

    Returns:
        PoscarResult: { ok, path, warnings, error, meta }
    """
    warnings: List[str] = []
    spec = specification or {}

    # 1) Ask the MaterialsProjectAgent for candidates
    try:
        # Compose a clear message for your MP agent. Adapt this to your actual MP agent API.
        mp_query_msg = (
            "Find the best structure for the given material. "
            "Return (1) a serialized structure (CIF string or pymatgen-as-dict), "
            "and (2) metadata: mp_id, energy_above_hull, spacegroup, formula_pretty. "
            f"Material query: {material}. "
            f"User specification (optional): {spec}."
        )
        # We rely on your existing orchestration pattern
        mp_response = poscar_agent.run_sync = False  # silence linters about attribute; not used
        # Call your MP agent:
        # The call must be awaited at a higher level; here we call synchronously via a blocking run helper.
        # If your environment requires async only, move this orchestration into the coordinator.
        # For most setups, you will wire this via the coordinator, not within the tool.
        raise RuntimeError(
            "This tool expects the coordinator to first call MaterialsProjectAgent and then pass the structure here. "
            "Wire this by splitting into two steps: (A) call MP agent to get structure, "
            "(B) call generate_poscar_from_structure (provided below)."
        )
    except Exception as e:
        # We deliberately don't perform cross-agent calls inside the tool to keep responsibilities clean.
        return PoscarResult(
            ok=False,
            error=str(e),
            warnings=["Coordinator must supply structure via the separate tool `generate_poscar_from_structure`."]
        )



@poscar_agent.tool_plain
def generate_poscar_from_structure(
    structure_payload: Dict[str, Any],
    payload_format: Literal["cif", "pmg_dict"] = "pmg_dict",
    cell: Literal["primitive", "conventional"] = "conventional",
    supercell: Optional[List[List[int]]] = None,
    meta: Optional[Dict[str, Any]] = None
) -> PoscarResult:
    """
    Write a POSCAR from a provided structure payload (CIF string or pymatgen Structure.as_dict()).

    Args:
        structure_payload: either a CIF text (if payload_format='cif') or a dict from Structure.as_dict()
        payload_format: 'cif' or 'pmg_dict'
        cell: 'primitive' or 'conventional'
        supercell: optional 3x3 integer matrix
        meta: optional metadata (mp_id, formula_pretty, energy_above_hull, spacegroup, etc.)

    Returns:
        PoscarResult
    """
    warnings: List[str] = []
    try:
        if payload_format == "cif":
            cif_text = structure_payload.get("cif") if isinstance(structure_payload, dict) else None
            if not isinstance(cif_text, str) or len(cif_text.strip()) == 0:
                return PoscarResult(ok=False, error="Missing or empty CIF text in structure_payload.")
            structure = Structure.from_str(cif_text, fmt="cif")
        elif payload_format == "pmg_dict":
            # Expect a dict produced by pymatgen's Structure.as_dict()
            structure = Structure.from_dict(structure_payload)
        else:
            return PoscarResult(ok=False, error=f"Unsupported payload_format '{payload_format}'.")

        # Convert to requested cell
        structure = _to_target_cell(structure, cell)

        # Apply optional supercell
        structure = _apply_supercell(structure, supercell)

        # Write POSCAR
        outfile = _outfile_path(prefix="mp")
        _write_poscar(structure, outfile)

        # Build meta block
        auto_meta = _structure_summary(structure)
        if meta:
            auto_meta.update(meta)
        auto_meta["cell"] = cell
        if supercell:
            auto_meta["supercell"] = supercell

        return PoscarResult(ok=True, path=outfile, warnings=warnings, meta=auto_meta)

    except Exception as e:
        logger.error(f"[DFT_POSCAR] Failed to write POSCAR: {e}")
        return PoscarResult(ok=False, error=str(e), warnings=warnings)



async def call_dft_poscar_agent(ctx: RunContext[AgentState], message2agent: str):
    """
    Coordinator entrypoint. Recommended wiring:

    1) Coordinator calls MaterialsProjectAgent to fetch a structure for the user's query
       (apply user constraints like mp_id, spacegroup, crystal system, lowest energy_above_hull).
       The MP agent should return either:
         - a CIF string, or
         - a pymatgen Structure.as_dict() payload,
         - plus metadata (mp_id, energy_above_hull, spacegroup, formula_pretty).

    2) Coordinator then calls this agent's tool `generate_poscar_from_structure(...)`
       with that payload + desired cell/supercell.

    This function simply runs the agent for any natural-language guidance, but the
    actual POSCAR writing should be done by the tool above.
    """
    agent_name = NAME
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    user_prompt = f"Current Task of your role: {message2agent}"

    result = await poscar_agent.run(user_prompt, deps=deps)

    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()

    logger.info(f"[{agent_name}] Action: {getattr(output, 'action', None)}")
    logger.info(f"[{agent_name}] Result: {getattr(output, 'result', None)}")
    return output

