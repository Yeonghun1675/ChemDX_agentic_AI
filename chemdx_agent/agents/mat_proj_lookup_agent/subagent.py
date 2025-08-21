# chemdx_agent/materials_agent.py
from __future__ import annotations

from ast import DictComp
from typing import List, Dict, Any, Optional, Literal, TypedDict, Union
from pydantic_ai import Agent, RunContext
from requests.utils import dict_to_sequence

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger

import os
import re

from mp_api.client import MPRester
from pymatgen.core import Structure

# -----------------------------

name = "MaterialsProjectAgent"
role = "Query Materials Project for structures and metadata; return a serializable structure payload plus key metadata."
context = "You must NOT invent data. Always return real entries from Materials Project.If multiple candidates match, use constraints (mp_id, spacegroup, crystal_system, elements) to narrow down.Prefer lowest energy_above_hull when no other constraints are provided.Return both a structure payload (pmg_dict or CIF) and a concise metadata dict"


system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Currunt Role: {role}
Important Context: {context}
"""

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

mp_agent = Agent(
    model = "openai:gpt-4o",
    output_type = Result,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt = system_prompt,
)


@mp_agent.tool
async def fetch_structure(
    ctx: RunContext[AgentState],
    material: str,
    specification: Optional[Dict[str, Any]] = None,
    payload_format: str = "pmg_dict",
) -> Dict[str, Any]:
    """
    Query MP for `material` and return:
      { ok: bool, structure_payload: dict|{cif: str}, payload_format: 'pmg_dict'|'cif', meta: {...}, error? }
    """
    # IMPLEMENT: actually call mp_api here; below is a placeholder structured stub.
    try:
        # ... your real MP call ...
        # structure_dict = Structure.as_dict() or {'cif': '...'}
        return {
            "ok": True,
            "structure_payload": {"dummy": "replace_with_structure_as_dict"},
            "payload_format": payload_format,
            "meta": {"material_query": material, "specification": specification or {}},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@mp_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal=deps.main_task,
        working_memory=deps.working_memory_description,
    )



class MPStructureResult(TypedDict, total=False):
    ok: bool
    structure_payload: Dict[str, Any]  # Structure.as_dict() OR {"cif": "..."} if payload_format="cif"
    payload_format: Literal["pmg_dict", "cif"]
    meta: Dict[str, Any]               # mp_id, formula_pretty, energy_above_hull, spacegroup, crystal_system, material_ids (aliases), warnings
    error: Optional[str]

class MPCandidate(TypedDict, total=False):
    mp_id: str
    formula_pretty: str
    energy_above_hull: float
    spacegroup: str
    crystal_system: str


_ID_RE = re.compile(r"^mp-\d+$", re.IGNORECASE)

def _get_api_key(ctx: Optional[RunContext[AgentState]] = None) -> Optional[str]:
    # 1) prefer deps/materials_api_key if you store it there
    if ctx and getattr(ctx, "deps", None) and hasattr(ctx.deps, "materials_api_key"):
        if ctx.deps.materials_api_key:
            return ctx.deps.materials_api_key
    return "Dhh7A13C7WRm72FnGobshRFyCeEM9X7h"

def _doc_to_meta(doc) -> Dict[str, Any]:
    # mp-api docs: summary search returns rich fields
    spg = getattr(doc, "symmetry", None).symbol if getattr(doc, "symmetry", None) else None
    crys = getattr(doc, "symmetry", None).crystal_system if getattr(doc, "symmetry", None) else None
    return {
        "mp_id": doc.material_id,
        "formula_pretty": getattr(doc, "formula_pretty", None),
        "energy_above_hull": getattr(doc, "energy_above_hull", None),
        "spacegroup": spg,
        "crystal_system": crys,
    }

def _best_doc(docs, spec: Dict[str, Any]) -> Optional[Any]:
    """Pick best doc: honor mp_id/spacegroup/crystal_system if given, else lowest EAH."""
    if not docs:
        return None
    # exact mp_id
    mp_id = spec.get("mp_id")
    if mp_id:
        for d in docs:
            if d.material_id.lower() == mp_id.lower():
                return d
    # filter by spacegroup/crystal_system
    sg = spec.get("spacegroup")
    cs = spec.get("crystal_system")
    filtered = []
    for d in docs:
        sym = getattr(d, "symmetry", None)
        if sg and (not sym or (sym.symbol != sg)):
            continue
        if cs and (not sym or (sym.crystal_system != cs)):
            continue
        filtered.append(d)
    pool = filtered if filtered else docs
    # lowest energy_above_hull
    return min(pool, key=lambda x: (x.energy_above_hull if x.energy_above_hull is not None else 1e9))

def _serialize_structure(structure: Structure, payload_format: Literal["pmg_dict","cif"]) -> Dict[str, Any]:
    if payload_format == "pmg_dict":
        return structure.as_dict()
    # cif
    return {"cif": structure.to(fmt="cif")}

def _search_kwargs_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Map user spec to mp-api summary.search kwargs."""
    kwargs: Dict[str, Any] = {}
    # elements filters
    if "elements" in spec and isinstance(spec["elements"], (list, tuple)):
        kwargs["elements"] = list(spec["elements"])
    if "exclude_elements" in spec and isinstance(spec["exclude_elements"], (list, tuple)):
        kwargs["exclude_elements"] = list(spec["exclude_elements"])
    # spacegroup/crystal system can't be passed directly to summary.search as kwargs in all versions;
    # we'll filter after search if needed.
    return kwargs



@mp_agent.tool_plain
def list_candidates(
    query: str,
    specification: Optional[Dict[str, Any]] = None,
    limit: int = 20
) -> List[MPCandidate]:
    """
    Return a shortlist of candidate materials for a query (formula or mp-id).
    Use this to show the user options if ambiguous.
    """
    api_key = _get_api_key()
    if not api_key:
        return [{"mp_id": "", "formula_pretty": "", "energy_above_hull": 0.0, "spacegroup": "ERROR", "crystal_system": "Missing MP_API_KEY"}]

    spec = specification or {}
    try:
        with MPRester(api_key) as mpr:
            if _ID_RE.match(query):
                # Direct mp-id lookup → one candidate
                doc = mpr.materials.summary.search(material_ids=[query])[0]
                return [MPCandidate(
                    mp_id=doc.material_id,
                    formula_pretty=doc.formula_pretty,
                    energy_above_hull=doc.energy_above_hull,
                    spacegroup=doc.symmetry.symbol if doc.symmetry else None,
                    crystal_system=doc.symmetry.crystal_system if doc.symmetry else None
                )]

            # formula/name search
            kwargs = _search_kwargs_from_spec(spec)
            docs = mpr.materials.summary.search(formula=query, **kwargs)
            # post-filter sg/cs if provided
            sg = spec.get("spacegroup")
            cs = spec.get("crystal_system")
            if sg or cs:
                tmp = []
                for d in docs:
                    sym = getattr(d, "symmetry", None)
                    if sg and (not sym or sym.symbol != sg):
                        continue
                    if cs and (not sym or sym.crystal_system != cs):
                        continue
                    tmp.append(d)
                docs = tmp

            out: List[MPCandidate] = []
            for d in docs[:max(1, limit)]:
                out.append(MPCandidate(
                    mp_id=d.material_id,
                    formula_pretty=d.formula_pretty,
                    energy_above_hull=d.energy_above_hull,
                    spacegroup=d.symmetry.symbol if d.symmetry else None,
                    crystal_system=d.symmetry.crystal_system if d.symmetry else None
                ))
            return out or []
    except Exception as e:
        logger.error(f"[MP list_candidates] {e}")
        return []


@mp_agent.tool_plain
def get_best_structure(
    query: str,
    specification: Optional[Dict[str, Any]] = None,
    payload_format: Literal["pmg_dict","cif"] = "pmg_dict"
) -> MPStructureResult:
    """
    Fetch the best-matching structure and return a serializable payload + metadata.
    - query: 'Bi2Te3', 'silicon', or 'mp-149'
    - specification: optional dict with:
        { "mp_id": "...", "spacegroup": "R-3m", "crystal_system": "trigonal",
          "elements": ["Bi","Te"], "exclude_elements": ["O"] }
    - payload_format: 'pmg_dict' (default) or 'cif'
    """
    api_key = _get_api_key()
    if not api_key:
        return MPStructureResult(ok=False, error="Missing MP_API_KEY (env) or deps.materials_api_key.", meta={"warnings": ["no_api_key"]})

    spec = specification or {}
    try:
        with MPRester(api_key) as mpr:
            if _ID_RE.match(query):
                docs = mpr.materials.summary.search(material_ids=[query])
            else:
                kwargs = _search_kwargs_from_spec(spec)
                docs = mpr.materials.summary.search(formula=query, **kwargs)

            if not docs:
                return MPStructureResult(ok=False, error=f"No MP entries found for '{query}' with spec={spec}.")

            # pick best doc
            doc = _best_doc(docs, spec)
            if doc is None:
                return MPStructureResult(ok=False, error="Unable to select a best candidate.")

            # fetch structure by mp-id
            structure: Structure = mpr.get_structure_by_material_id(doc.material_id)
            payload = _serialize_structure(structure, payload_format)

            meta = _doc_to_meta(doc)
            meta["candidate_count"] = len(docs)

            return MPStructureResult(
                ok=True,
                structure_payload=payload,
                payload_format=payload_format,
                meta=meta
            )
    except Exception as e:
        logger.error(f"[MP get_best_structure] {e}")
        return MPStructureResult(ok=False, error=str(e), meta={})




# -----------------------------

async def call_mp_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call general agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
    
    This agent is used to query the materials project for a given material and return the structure and metadata.
    It can call the list_candidates and get_best_structure tools to fetch the structure and metadata.
    """
    agent_name = "MaterialsProjectAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    user_prompt = f"Current Task of your role: {message2agent}"

    result = await mp_agent.run(user_prompt, deps=deps)

    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()

    logger.info(f"[{agent_name}] Action: {getattr(output, 'action', None)}")
    logger.info(f"[{agent_name}] Result: {getattr(output, 'result', None)}")
    return output

