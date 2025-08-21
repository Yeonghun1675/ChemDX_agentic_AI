
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


name = "DFTPOSCARAgent"
role = "Generate VASP POSCAR files by retrieving authoritative structures from Materials Project via the MaterialsProjectAgent."
context = "You must NOT hallucinate structures. Always obtain a structure via the MaterialsProjectAgent.If multiple candidates are returned, prefer the one with the lowest energy_above_hull unless the user specifiesa space group, crystal system, or mp-id to use. You can convert structures to primitive or conventional cells. Always return structured results."


system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Current Role: {role}
Important Context: {context}
"""

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

dft_poscar_agent = Agent(
    model = "openai:gpt-4o",
    output_type = Result,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt = system_prompt,
)


@dft_poscar_agent.system_prompt(dynamic=True)
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

@dft_poscar_agent.tool
async def generate_poscar_via_mp(
    ctx: RunContext[AgentState],
    material: str,
    specification: Optional[Dict[str, Any]] = None,
    payload_format: Literal["pmg_dict", "cif"] = "pmg_dict",
    cell: Literal["primitive", "conventional"] = "conventional",
    supercell: Optional[List[List[int]]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> PoscarResult:
    """
    One-shot pipeline:
      1) Call MaterialsProjectAgent to fetch the best structure for `material`.
      2) Write a POSCAR using the returned payload.

    Returns:
      PoscarResult = { ok, path?, warnings?, error?, meta? }
    """
    # Import here to avoid circular imports at module import time
    try:
        from chemdx_agent.agents.mat_proj_lookup_agent.subagent import mp_lookup_agent
    except Exception as e:
        return PoscarResult(ok=False, error=f"Failed to import MaterialsProjectAgent: {e}")

    # 1) Fetch structure from MP
    mp_req = {
        "material": material,
        "specification": specification or {},
        "payload_format": payload_format,
    }
    mp_res = await mp_lookup_agent.run(mp_req, deps=ctx.deps, usage=ctx.usage)
    mp_out = mp_res.output if isinstance(mp_res.output, dict) else getattr(mp_res, "output", {})

    if not isinstance(mp_out, dict) or not mp_out.get("ok"):
        return PoscarResult(ok=False, error=f"MP lookup failed: {mp_out.get('error', 'unknown error')}")

    structure_payload = mp_out.get("structure_payload")
    pf = mp_out.get("payload_format", payload_format)
    if structure_payload is None or pf not in ("pmg_dict", "cif"):
        return PoscarResult(ok=False, error="Invalid MP payload: missing structure_payload or unsupported payload_format.")

    # Merge meta for traceability
    meta: Dict[str, Any] = {}
    if isinstance(mp_out.get("meta"), dict):
        meta.update(mp_out["meta"])
    if isinstance(extra_meta, dict):
        meta.update(extra_meta)
    meta.update({"material_query": material})

    # 2) Write POSCAR using your existing pure tool
    poscar_res = generate_poscar_from_structure(
        structure_payload=structure_payload,
        payload_format=pf,
        cell=cell,
        supercell=supercell,
        meta=meta,
    )
    return poscar_res

@dft_poscar_agent.tool_plain
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

    try:
        # This tool is designed to work with the coordinator pattern
        # The coordinator should first call MaterialsProjectAgent to get the structure
        # and then call generate_poscar_from_structure with the result
        
        warnings.append("This tool requires coordination with MaterialsProjectAgent")
        warnings.append("Use the workflow: (1) call MP agent to get structure, (2) call generate_poscar_from_structure")
        
        return PoscarResult(
            ok=False,
            error="This tool requires coordination with MaterialsProjectAgent. Use the workflow: (1) call MP agent to get structure, (2) call generate_poscar_from_structure",
            warnings=warnings
        )
        
    except Exception as e:
        return PoscarResult(
            ok=False,
            error=str(e),
            warnings=warnings
        )



@dft_poscar_agent.tool_plain
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


@dft_poscar_agent.tool
async def generate_poscar_coordinated(
    ctx: RunContext[AgentState],
    material: str,
    specification: Optional[Dict[str, Any]] = None,
    cell: Literal["primitive", "conventional"] = "conventional",
    supercell: Optional[List[List[int]]] = None,
    title: Optional[str] = None
) -> PoscarResult:
    """
    Complete workflow: Fetch structure from Materials Project and generate POSCAR file.
    
    Args:
        material: free-form query like "Bi2Te3", "Silicon", or an mp-id "mp-149".
        specification: optional constraints to help disambiguate results
        cell: choose "primitive" or "conventional" cell in output.
        supercell: optional 3x3 integer matrix
        title: optional title for metadata

    Returns:
        PoscarResult: { ok, path, warnings, error, meta }
    """
    warnings: List[str] = []
    spec = specification or {}

    try:
        # Step 1: Import and call Materials Project agent to get structure
        try:
            from chemdx_agent.agents.mat_proj_lookup_agent.subagent import mp_agent
        except ImportError as e:
            return PoscarResult(
                ok=False, 
                error=f"Failed to import MaterialsProjectAgent: {e}",
                warnings=warnings
            )

        # Step 2: Get the best structure from MP
        mp_query = f"Get the best structure for material '{material}' with specification {spec}"
        mp_result = await mp_agent.run(mp_query, deps=ctx.deps, usage=ctx.usage)
        
        if not hasattr(mp_result, 'output') or not mp_result.output:
            return PoscarResult(
                ok=False,
                error="Materials Project agent returned no output",
                warnings=warnings
            )
        
        # Extract the result from MP agent - handle different response formats
        mp_output = mp_result.output
        mp_data = None
        
        # Try different ways to extract the data
        if hasattr(mp_output, 'result'):
            mp_data = mp_output.result
        elif hasattr(mp_output, 'action'):
            # Sometimes the result is in the action field
            mp_data = mp_output.action
        else:
            mp_data = mp_output
            
        # Debug logging
        logger.info(f"[DFT_POSCAR] MP agent response type: {type(mp_data)}")
        logger.info(f"[DFT_POSCAR] MP agent response: {mp_data}")
        
        # Check if MP agent succeeded
        if not isinstance(mp_data, dict):
            # Try to parse if it's a string
            if isinstance(mp_data, str):
                # Check if this is a human-readable response with embedded structure data
                if "Structure Payload" in mp_data and "pmg_dict format" in mp_data:
                    # Extract the JSON structure from the text response
                    try:
                        import json
                        import re
                        
                        # Find the JSON structure in the text
                        json_match = re.search(r'\{.*\}', mp_data, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            # Clean up the JSON string (remove newlines and fix formatting)
                            json_str = re.sub(r'\s+', ' ', json_str)
                            json_str = json_str.replace('true', 'True').replace('false', 'False')
                            
                            structure_payload = json.loads(json_str)
                            
                            # Create a proper response format
                            mp_data = {
                                "ok": True,
                                "structure_payload": structure_payload,
                                "payload_format": "pmg_dict",
                                "meta": {
                                    "material_query": material,
                                    "extracted_from_text": True
                                }
                            }
                            
                            logger.info(f"[DFT_POSCAR] Successfully extracted structure from text response")
                        else:
                            return PoscarResult(
                                ok=False,
                                error="Could not find JSON structure in MP agent text response",
                                warnings=warnings
                            )
                    except Exception as parse_error:
                        return PoscarResult(
                            ok=False,
                            error=f"Failed to parse structure from MP agent text response: {parse_error}",
                            warnings=warnings
                        )
                else:
                    try:
                        import json
                        mp_data = json.loads(mp_data)
                    except:
                        return PoscarResult(
                            ok=False,
                            error=f"Could not parse MP agent response: {mp_data}",
                            warnings=warnings
                        )
            else:
                return PoscarResult(
                    ok=False,
                    error=f"Unexpected MP agent response format: {type(mp_data)}",
                    warnings=warnings
                )
        
        if not mp_data.get("ok"):
            error_msg = mp_data.get("error", "Unknown error from Materials Project agent")
            return PoscarResult(
                ok=False,
                error=f"Materials Project lookup failed: {error_msg}",
                warnings=warnings
            )
        
        # Extract structure payload and metadata
        structure_payload = mp_data.get("structure_payload")
        payload_format = mp_data.get("payload_format", "pmg_dict")
        meta = mp_data.get("meta", {})
        
        if not structure_payload:
            return PoscarResult(
                ok=False,
                error="No structure payload received from Materials Project agent",
                warnings=warnings
            )
        
        # Step 3: Generate POSCAR from the structure
        poscar_result = generate_poscar_from_structure(
            structure_payload=structure_payload,
            payload_format=payload_format,
            cell=cell,
            supercell=supercell,
            meta=meta
        )
        
        # Add coordination info to metadata
        if poscar_result.get("ok") and poscar_result.get("meta"):
            poscar_result["meta"]["coordination"] = "Generated via Materials Project coordination"
            poscar_result["meta"]["material_query"] = material
            poscar_result["meta"]["specification"] = spec
        
        return poscar_result
        
    except Exception as e:
        logger.error(f"[DFT_POSCAR] Coordination failed: {e}")
        return PoscarResult(
            ok=False,
            error=f"Coordination with Materials Project agent failed: {str(e)}",
            warnings=warnings
        )


@dft_poscar_agent.tool
async def generate_poscar_for_material_with_best_zt_temperature(
    ctx: RunContext[AgentState],
    formula: str,
    min_temperature: float = None,
    max_temperature: float = None,
    cell: Literal["primitive", "conventional"] = "conventional",
    supercell: Optional[List[List[int]]] = None
) -> PoscarResult:
    """
    Complete workflow for specific materials like Bi2Te3:
    1. Find best temperature for ZT performance for the given material
    2. Get crystal structure from Materials Project
    3. Generate POSCAR file with optimization info
    
    Args:
        formula: chemical formula (e.g., "Bi2Te3")
        min_temperature: minimum temperature in K (optional)
        max_temperature: maximum temperature in K (optional)
        cell: choose "primitive" or "conventional" cell in output
        supercell: optional 3x3 integer matrix
        
    Returns:
        PoscarResult: { ok, path, warnings, error, meta }
    """
    warnings: List[str] = []
    
    try:
        # Step 1: Find best temperature for ZT performance
        try:
            from chemdx_agent.agents.tme_db_agent.subagent import find_best_temperature_for_zt
        except ImportError as e:
            return PoscarResult(
                ok=False, 
                error=f"Failed to import DatabaseAgent: {e}",
                warnings=warnings
            )

        # Query database for temperature optimization
        temp_result = find_best_temperature_for_zt(formula, min_temperature, max_temperature)
        
        if temp_result.get('error'):
            warnings.append(f"Database lookup: {temp_result['error']}")
            warnings.append("Proceeding with POSCAR generation anyway")
        else:
            best_temp = temp_result.get('best_temperature')
            best_zt = temp_result.get('best_zt')
            warnings.append(f"Found optimal temperature: {best_temp}K with ZT = {best_zt}")
            warnings.append(f"Total data points analyzed: {temp_result.get('total_data_points', 0)}")
        
        # Step 2: Get crystal structure from Materials Project
        try:
            from chemdx_agent.agents.mat_proj_lookup_agent.subagent import get_best_structure
        except ImportError as e:
            return PoscarResult(
                ok=False, 
                error=f"Failed to import MaterialsProjectAgent: {e}",
                warnings=warnings
            )

        # Query MP for the structure
        mp_result = get_best_structure(formula, payload_format="pmg_dict")
        
        if not mp_result.get("ok"):
            return PoscarResult(
                ok=False,
                error=f"Materials Project lookup failed: {mp_result.get('error', 'Unknown error')}",
                warnings=warnings
            )
        
        # Extract structure payload and metadata
        structure_payload = mp_result.get("structure_payload")
        payload_format = mp_result.get("payload_format", "pmg_dict")
        meta = mp_result.get("meta", {})
        
        if not structure_payload:
            return PoscarResult(
                ok=False,
                error="No structure payload received from Materials Project agent",
                warnings=warnings
            )
        
        # Step 3: Generate POSCAR from the structure
        poscar_result = generate_poscar_from_structure(
            structure_payload=structure_payload,
            payload_format=payload_format,
            cell=cell,
            supercell=supercell,
            meta=meta
        )
        
        # Add material-specific optimization metadata
        if poscar_result.get("ok") and poscar_result.get("meta"):
            poscar_result["meta"]["material_optimization_task"] = True
            poscar_result["meta"]["target_material"] = formula
            if not temp_result.get('error'):
                poscar_result["meta"]["optimal_temperature_K"] = temp_result.get('best_temperature')
                poscar_result["meta"]["optimal_zt_value"] = temp_result.get('best_zt')
                poscar_result["meta"]["temperature_range_analyzed"] = temp_result.get('temperature_range')
            poscar_result["meta"]["coordination"] = "Generated via material-specific temperature optimization workflow"
        
        return poscar_result
        
    except Exception as e:
        logger.error(f"[DFT_POSCAR] Material optimization workflow failed: {e}")
        return PoscarResult(
            ok=False,
            error=f"Material optimization workflow failed: {str(e)}",
            warnings=warnings
        )

@dft_poscar_agent.tool
async def generate_poscar_for_thermoelectric(
    ctx: RunContext[AgentState],
    temperature_min: float = 600.0,
    property_name: str = "ZT",
    max_results: int = 1,
    cell: Literal["primitive", "conventional"] = "conventional",
    supercell: Optional[List[List[int]]] = None
) -> PoscarResult:
    """
    Complete workflow for thermoelectric materials: 
    1. Query database for best performing materials at specified temperature
    2. Get crystal structure from Materials Project
    3. Generate POSCAR file
    
    Args:
        temperature_min: minimum temperature in K (default: 600K for high temp)
        property_name: property to rank by (default: "ZT")
        max_results: number of top materials to consider (default: 1)
        cell: choose "primitive" or "conventional" cell in output
        supercell: optional 3x3 integer matrix
        
    Returns:
        PoscarResult: { ok, path, warnings, error, meta }
    """
    warnings: List[str] = []
    
    try:
        # Step 1: Import and call Database agent to get top thermoelectric materials
        try:
            from chemdx_agent.agents.tme_db_agent.subagent import database_agent
        except ImportError as e:
            return PoscarResult(
                ok=False, 
                error=f"Failed to import DatabaseAgent: {e}",
                warnings=warnings
            )

        # Query database for top performers at specified temperature
        db_query = f"Get top {max_results} materials by {property_name} with minimum temperature {temperature_min}K"
        db_result = await database_agent.run(db_query, deps=ctx.deps, usage=ctx.usage)
        
        if not hasattr(db_result, 'output') or not db_result.output:
            return PoscarResult(
                ok=False,
                error="Database agent returned no output",
                warnings=warnings
            )
        
        # Extract database results
        db_output = db_result.output
        if hasattr(db_output, 'result'):
            db_data = db_output.result
        else:
            db_data = db_output
            
        # Check if database query succeeded
        if not isinstance(db_data, dict) or not db_data.get("results"):
            error_msg = db_data.get("error", "Unknown error from database agent") if isinstance(db_data, dict) else "Invalid response format"
            return PoscarResult(
                ok=False,
                error=f"Database query failed: {error_msg}",
                warnings=warnings
            )
        
        # Get the top material
        top_materials = db_data.get("results", [])
        if not top_materials:
            return PoscarResult(
                ok=False,
                error=f"No materials found with temperature >= {temperature_min}K",
                warnings=warnings
            )
        
        best_material = top_materials[0]
        formula = best_material.get("formula")
        zt_value = best_material.get("avg_ZT", "N/A")
        
        if not formula:
            return PoscarResult(
                ok=False,
                error="Top material missing formula information",
                warnings=warnings
            )
        
        warnings.append(f"Selected material: {formula} with {property_name} = {zt_value}")
        warnings.append(f"Temperature range: {best_material.get('temperature_range', 'N/A')}")
        
        # Step 2: Get crystal structure from Materials Project
        try:
            from chemdx_agent.agents.mat_proj_lookup_agent.subagent import mp_agent
        except ImportError as e:
            return PoscarResult(
                ok=False, 
                error=f"Failed to import MaterialsProjectAgent: {e}",
                warnings=warnings
            )

        # Query MP for the structure
        mp_query = f"Get the best structure for material '{formula}'"
        mp_result = await mp_agent.run(mp_query, deps=ctx.deps, usage=ctx.usage)
        
        if not hasattr(mp_result, 'output') or not mp_result.output:
            return PoscarResult(
                ok=False,
                error="Materials Project agent returned no output",
                warnings=warnings
            )
        
        # Extract MP results
        mp_output = mp_result.output
        if hasattr(mp_output, 'result'):
            mp_data = mp_output.result
        else:
            mp_data = mp_output
            
        # Check if MP lookup succeeded
        if not isinstance(mp_data, dict) or not mp_data.get("ok"):
            error_msg = mp_data.get("error", "Unknown error from Materials Project agent") if isinstance(mp_data, dict) else "Invalid response format"
            return PoscarResult(
                ok=False,
                error=f"Materials Project lookup failed: {error_msg}",
                warnings=warnings
            )
        
        # Extract structure payload and metadata
        structure_payload = mp_data.get("structure_payload")
        payload_format = mp_data.get("payload_format", "pmg_dict")
        meta = mp_data.get("meta", {})
        
        if not structure_payload:
            return PoscarResult(
                ok=False,
                error="No structure payload received from Materials Project agent",
                warnings=warnings
            )
        
        # Step 3: Generate POSCAR from the structure
        poscar_result = generate_poscar_from_structure(
            structure_payload=structure_payload,
            payload_format=payload_format,
            cell=cell,
            supercell=supercell,
            meta=meta
        )
        
        # Add thermoelectric-specific metadata
        if poscar_result.get("ok") and poscar_result.get("meta"):
            poscar_result["meta"]["thermoelectric_task"] = True
            poscar_result["meta"]["temperature_min"] = temperature_min
            poscar_result["meta"]["property_ranked_by"] = property_name
            poscar_result["meta"]["selected_material"] = formula
            poscar_result["meta"]["zt_value"] = zt_value
            poscar_result["meta"]["coordination"] = "Generated via thermoelectric workflow"
        
        return poscar_result
        
    except Exception as e:
        logger.error(f"[DFT_POSCAR] Thermoelectric workflow failed: {e}")
        return PoscarResult(
            ok=False,
            error=f"Thermoelectric workflow failed: {str(e)}",
            warnings=warnings
        )


# -----------------------------

# call agent function
async def call_poscar_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call general agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
        This agent is used to generate a POSCAR file for a given material from the materials project. 
        It can call MaterialsProjectAgent to fetch a structure for the user's query
        The MP agent would return either:
         - a CIF string, or
         - a pymatgen Structure.as_dict() payload,
         - plus metadata (mp_id, energy_above_hull, spacegroup, formula_pretty).

         - then it can call `generate_poscar_from_structure(...)` with that payload + desired cell/supercell.
    """
    agent_name = "DFTPOSCARAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}ㅤ")

    user_prompt = "Current Task of your role: {message2agent}"

    result = await dft_poscar_agent.run(
        user_prompt, deps=deps
    )

    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()

    logger.info(f"[{agent_name}] Action: {output.action}ㅤ")
    logger.info(f"[{agent_name}] Result: {output.result}ㅤ")

    return output

