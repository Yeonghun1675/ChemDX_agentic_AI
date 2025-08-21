from typing import List, Dict, Any
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger

import pandas as pd
import os
import numpy as np

# Get the directory of the current file and construct the CSV path
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "thermoelectrics.csv")
df = pd.read_csv(csv_path)

name = "DatabaseAgent"
role = "Query thermoelectric materials database"
context = """You are connected to a thermoelectric materials database with 5,205+ materials. You can search for material properties, return results at specific temperatures, and compare entries. You have access to data on 
Formula	temperature(K), seebeck_coefficient(μV/K),	electrical_conductivity(S/m),	thermal_conductivity(W/mK),	power_factor(W/mK2),	ZT	and reference study. 
IMPORTANT: When users don't specify a clear composition, always offer them the top performing materials or help them narrow down their search to avoid overwhelming results. Always clarify if multiple candidate compositions are found.
"""

system_prompt = f"""You are the {name}. 
You can use available tools or request help from specialized sub-agents (e.g., VisualizationAgent, MaterialsProjectAgent). You must only carry out the role assigned to you. 
If a request is outside your capabilities, ask for support from the appropriate agent instead of handling it yourself.

Your Current Role: {role}
Important Context: {context}
"""

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

database_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=system_prompt,
)

@database_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal=deps.main_task,
        working_memory=deps.working_memory_description,
    )

#-----------

#Tools

@database_agent.tool_plain
def read_database_schema() -> List[str]:
    """Return the list of available columns in the thermoelectric database."""
    return df.columns.tolist()

@database_agent.tool_plain
def find_material_variants(material_hint: str) -> List[str]:
    """Find all material formulas in the database that contain the given hint (e.g. element symbol or name).
    Args:
        material_hint: (str) a partial string such as 'Bi', 'Sb', or 'bismuth'
    Output:
        (List[str]) all matching formulas
    """
    matches = df[df["Formula"].str.contains(material_hint, case=False, regex=False)]
    unique_formulas = matches["Formula"].unique().tolist()
    return unique_formulas


@database_agent.tool_plain
def get_material_properties(formula: str) -> Dict[str, Any]:
    """Return all temperature-dependent property data for an exact formula.

    Args:
        formula: (str) Exact chemical formula (e.g., 'Bi2Te3')

    Output:
        (Dict) contains formula, number of data points, and full property table
    """
    # Filter database for exact matches
    subset = df[df["Formula"].str.strip().str.lower() == formula.strip().lower()]

    if subset.empty:
        return {
            "formula": formula,
            "message": "No exact match found in database.",
            "data_points": 0,
            "results": []
        }

    # Convert rows to dicts for easier JSON serialization
    results = subset.to_dict(orient="records")

    return {
        "formula": formula,
        "data_points": len(results),
        "temperature_range": f"{subset['temperature(K)'].min():.0f}K - {subset['temperature(K)'].max():.0f}K",
        "results": results
    }



@database_agent.tool_plain
def smart_material_search(query: str, max_results: int = 10) -> Dict[str, Any]:
    """Smart search for materials with intelligent ranking and filtering.
    Args:
        query: (str) search query (element, property, or general description)
        max_results: (int) maximum number of results to return (default: 10)
    Output:
        (Dict) ranked results with material info and relevance scores
    """
    # Convert query to lowercase for case-insensitive search
    query_lower = query.lower()
    
    # Initialize results
    results = []
    
    # Check if this is an element search (single element symbol or name)
    element_search = False
    if len(query.strip()) <= 3 and query.strip().isalpha():
        element_search = True
    
    if element_search:
        # For element searches, prioritize materials where the element is a major component
        formula_matches = df[df["Formula"].str.contains(query, case=False, regex=False)]
        
        # Calculate composition relevance for each formula
        composition_scores = {}
        for formula in formula_matches["Formula"].unique():
            score = calculate_composition_relevance(formula, query)
            composition_scores[formula] = score
        
        # Filter out materials where the element is just a tiny dopant (< 5% composition)
        major_component_formulas = [f for f, score in composition_scores.items() if score > 0.05]
        
        if major_component_formulas:
            formula_matches = df[df["Formula"].isin(major_component_formulas)]
        else:
            # If no major component materials found, fall back to all matches
            formula_matches = df[df["Formula"].str.contains(query, case=False, regex=False)]
    else:
        # For non-element searches, use regular string matching
        formula_matches = df[df["Formula"].str.contains(query, case=False, regex=False)]
    
    # Search for specific properties mentioned
    property_keywords = {
        'high zt': ['ZT'],
        'high seebeck': ['seebeck_coefficient(μV/K)'],
        'high conductivity': ['electrical_conductivity(S/m)'],
        'low thermal': ['thermal_conductivity(W/mK)'],
        'high power factor': ['power_factor(W/mK2)'],
        'efficient': ['ZT', 'power_factor(W/mK2)'],
        'performance': ['ZT', 'power_factor(W/mK2)']
    }
    
    # Check if query mentions specific properties
    relevant_properties = []
    for keyword, props in property_keywords.items():
        if keyword in query_lower:
            relevant_properties.extend(props)
    
    # If no specific properties mentioned, default to ZT (most important)
    if not relevant_properties:
        relevant_properties = ['ZT']
    
    # Group by formula and calculate average properties
    formula_groups = formula_matches.groupby('Formula').agg({
        'ZT': 'mean',
        'power_factor(W/mK2)': 'mean',
        'seebeck_coefficient(μV/K)': 'mean',
        'electrical_conductivity(S/m)': 'mean',
        'thermal_conductivity(W/mK)': 'mean',
        'temperature(K)': ['min', 'max', 'count']
    }).round(4)
    
    # Flatten column names
    formula_groups.columns = ['_'.join(col).strip() for col in formula_groups.columns.values]
    
    # Calculate relevance score based on properties and composition
    for formula, row in formula_groups.iterrows():
        # Normalize properties to 0-1 scale for scoring
        zt_score = min(row['ZT_mean'] / 2.0, 1.0) if pd.notna(row['ZT_mean']) else 0  # ZT typically 0-2
        pf_score = min(row['power_factor(W/mK2)_mean'] / 0.01, 1.0) if pd.notna(row['power_factor(W/mK2)_mean']) else 0  # Power factor typically 0-0.01
        seebeck_score = min(abs(row['seebeck_coefficient(μV/K)_mean']) / 500, 1.0) if pd.notna(row['seebeck_coefficient(μV/K)_mean']) else 0  # Seebeck typically 0-500
        
        # Calculate composition relevance if this is an element search
        composition_score = 0
        if element_search:
            composition_score = calculate_composition_relevance(formula, query)
        
        # Calculate weighted relevance score
        if element_search:
            # For element searches, weight composition heavily (60%) and properties (40%)
            relevance_score = (0.6 * composition_score + 0.3 * zt_score + 0.1 * pf_score)
        else:
            # For property searches, weight properties heavily
            relevance_score = (0.5 * zt_score + 0.3 * pf_score + 0.2 * seebeck_score)
        
        results.append({
            'formula': formula,
            'relevance_score': round(relevance_score, 3),
            'composition_score': round(composition_score, 3) if element_search else None,
            'avg_ZT': row['ZT_mean'],
            'avg_power_factor': row['power_factor(W/mK2)_mean'],
            'avg_seebeck': row['seebeck_coefficient(μV/K)_mean'],
            'avg_electrical_conductivity': row['electrical_conductivity(S/m)_mean'],
            'avg_thermal_conductivity': row['thermal_conductivity(W/mK)_mean'],
            'temperature_range': f"{row['temperature(K)_min']:.0f}K - {row['temperature(K)_max']:.0f}K",
            'data_points': int(row['temperature(K)_count'])
        })
    
    # Sort by relevance score (descending)
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Limit results
    results = results[:max_results]
    
    return {
        'query': query,
        'total_matches': len(formula_groups),
        'returned_results': len(results),
        'element_search': element_search,
        'results': results
    }

def calculate_composition_relevance(formula: str, element: str) -> float:
    """Calculate how relevant a material is based on the composition of the searched element.
    
    Args:
        formula: Chemical formula (e.g., 'Bi2Te3', 'Cu0.01Bi1.99Te3')
        element: Element symbol to search for (e.g., 'Bi')
    
    Returns:
        float: Composition relevance score (0-1, higher = more relevant)
    """
    element = element.strip().capitalize()
    
    # Simple parsing for common formula patterns
    # This could be improved with a proper chemical formula parser
    
    # Check if element appears at the beginning (likely major component)
    if formula.startswith(element):
        return 1.0
    
    # Check for element with number (e.g., Bi2, Bi0.5)
    import re
    pattern = rf'{element}(\d+\.?\d*)'
    matches = re.findall(pattern, formula)
    
    if matches:
        # Convert to float and normalize
        numbers = [float(match) for match in matches]
        max_number = max(numbers)
        
        # Normalize based on typical stoichiometric ratios
        if max_number >= 2.0:
            return 1.0  # Major component (e.g., Bi2)
        elif max_number >= 1.5:
            return 0.9  # Major component (e.g., Bi1.8)
        elif max_number >= 1.0:
            return 0.8  # Significant component (e.g., Bi1)
        elif max_number >= 0.5:
            return 0.6  # Moderate component (e.g., Bi0.5)
        elif max_number >= 0.2:
            return 0.4  # Minor component (e.g., Bi0.2)
        elif max_number >= 0.05:
            return 0.2  # Trace component (e.g., Bi0.05)
        else:
            return 0.1  # Very trace component (e.g., Bi0.01)
    
    # If element appears but no clear stoichiometry, assume moderate relevance
    if element in formula:
        return 0.5
    
    return 0.0

@database_agent.tool_plain
def search_major_component_materials(element: str, min_composition: float = 0.2, max_results: int = 15) -> Dict[str, Any]:
    """Search for materials where the specified element is a major component.
    
    Args:
        element: (str) Element symbol to search for (e.g., 'Bi', 'Te', 'Cu')
        min_composition: (float) Minimum composition threshold (0.0-1.0, default: 0.2 = 20%)
        max_results: (int) Maximum number of results to return (default: 15)
    
    Output:
        (Dict) Materials where the element is a major component, ranked by performance
    """
    element = element.strip().capitalize()
    
    # Find all materials containing the element
    formula_matches = df[df["Formula"].str.contains(element, case=False, regex=False)]
    
    # Calculate composition relevance for each formula
    composition_scores = {}
    for formula in formula_matches["Formula"].unique():
        score = calculate_composition_relevance(formula, element)
        if score >= min_composition:
            composition_scores[formula] = score
    
    # Filter to only major component materials
    major_component_formulas = list(composition_scores.keys())
    
    if not major_component_formulas:
        return {
            'element': element,
            'min_composition': min_composition,
            'message': f'No materials found where {element} is at least {min_composition*100:.0f}% of the composition',
            'total_matches': 0,
            'returned_results': 0,
            'results': []
        }
    
    # Get data for major component materials
    filtered_df = df[df["Formula"].isin(major_component_formulas)]
    
    # Group by formula and calculate average properties
    formula_groups = filtered_df.groupby('Formula').agg({
        'ZT': 'mean',
        'power_factor(W/mK2)': 'mean',
        'seebeck_coefficient(μV/K)': 'mean',
        'electrical_conductivity(S/m)': 'mean',
        'thermal_conductivity(W/mK)': 'mean',
        'temperature(K)': ['min', 'max', 'count']
    }).round(4)
    
    # Flatten column names
    formula_groups.columns = ['_'.join(col).strip() for col in formula_groups.columns.values]
    
    # Calculate combined score (composition + performance)
    results = []
    for formula, row in formula_groups.iterrows():
        composition_score = composition_scores[formula]
        
        # Normalize properties to 0-1 scale for scoring
        zt_score = min(row['ZT_mean'] / 2.0, 1.0) if pd.notna(row['ZT_mean']) else 0
        pf_score = min(row['power_factor(W/mK2)_mean'] / 0.01, 1.0) if pd.notna(row['power_factor(W/mK2)_mean']) else 0
        
        # Combined score: 60% composition, 40% performance
        combined_score = 0.6 * composition_score + 0.4 * (0.7 * zt_score + 0.3 * pf_score)
        
        results.append({
            'formula': formula,
            'composition_score': round(composition_score, 3),
            'combined_score': round(combined_score, 3),
            'avg_ZT': row['ZT_mean'],
            'avg_power_factor': row['power_factor(W/mK2)_mean'],
            'avg_seebeck': row['seebeck_coefficient(μV/K)_mean'],
            'avg_electrical_conductivity': row['electrical_conductivity(S/m)_mean'],
            'avg_thermal_conductivity': row['thermal_conductivity(W/mK)_mean'],
            'temperature_range': f"{row['temperature(K)_min']:.0f}K - {row['temperature(K)_max']:.0f}K",
            'data_points': int(row['temperature(K)_count'])
        })
    
    # Sort by combined score (descending)
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Limit results
    results = results[:max_results]
    
    return {
        'element': element,
        'min_composition': min_composition,
        'total_matches': len(major_component_formulas),
        'returned_results': len(results),
        'results': results
    }

@database_agent.tool_plain
def get_top_performers(property_name: str = "ZT", max_results: int = 10, min_temperature: float = None, max_temperature: float = None) -> Dict[str, Any]:
    """Get top performing materials by specific property with optional temperature filtering.
    Args:
        property_name: (str) property to rank by: 'ZT', 'power_factor', 'seebeck', 'electrical_conductivity', 'thermal_conductivity'
        max_results: (int) maximum number of results to return (default: 10)
        min_temperature: (float) minimum temperature in K (optional)
        max_temperature: (float) maximum temperature in K (optional)
    Output:
        (Dict) top performing materials with their properties
    """
    # Map property names to column names
    property_map = {
        'ZT': 'ZT',
        'power_factor': 'power_factor(W/mK2)',
        'seebeck': 'seebeck_coefficient(μV/K)',
        'electrical_conductivity': 'electrical_conductivity(S/m)',
        'thermal_conductivity': 'thermal_conductivity(W/mK)'
    }
    
    if property_name not in property_map:
        return {"error": f"Invalid property. Choose from: {list(property_map.keys())}"}
    
    column_name = property_map[property_name]
    
    # Apply temperature filtering if specified
    filtered_df = df.copy()
    if min_temperature is not None:
        filtered_df = filtered_df[filtered_df['temperature(K)'] >= min_temperature]
    if max_temperature is not None:
        filtered_df = filtered_df[filtered_df['temperature(K)'] <= max_temperature]
    
    # Group by formula and calculate average properties
    formula_groups = filtered_df.groupby('Formula').agg({
        'ZT': 'mean',
        'power_factor(W/mK2)': 'mean',
        'seebeck_coefficient(μV/K)': 'mean',
        'electrical_conductivity(S/m)': 'mean',
        'thermal_conductivity(W/mK)': 'mean',
        'temperature(K)': ['min', 'max', 'count']
    }).round(4)
    
    # Flatten column names and create a mapping
    formula_groups.columns = ['_'.join(col).strip() for col in formula_groups.columns.values]
    
    # Create a mapping for the property column names
    property_column_map = {
        'ZT': 'ZT_mean',
        'power_factor': 'power_factor(W/mK2)_mean',
        'seebeck': 'seebeck_coefficient(μV/K)_mean',
        'electrical_conductivity': 'electrical_conductivity(S/m)_mean',
        'thermal_conductivity': 'thermal_conductivity(W/mK)_mean'
    }
    
    sort_column = property_column_map[property_name]
    
    # Sort by the specified property (descending for most properties, ascending for thermal conductivity)
    if property_name == 'thermal_conductivity':
        formula_groups = formula_groups.sort_values(sort_column, ascending=True)
    else:
        formula_groups = formula_groups.sort_values(sort_column, ascending=False)
    
    # Get top results
    top_results = []
    for formula, row in formula_groups.head(max_results).iterrows():
        top_results.append({
            'formula': formula,
            'avg_ZT': row['ZT_mean'],
            'avg_power_factor': row['power_factor(W/mK2)_mean'],
            'avg_seebeck': row['seebeck_coefficient(μV/K)_mean'],
            'avg_electrical_conductivity': row['electrical_conductivity(S/m)_mean'],
            'avg_thermal_conductivity': row['thermal_conductivity(W/mK)_mean'],
            'temperature_range': f"{row['temperature(K)_min']:.0f}K - {row['temperature(K)_max']:.0f}K",
            'data_points': int(row['temperature(K)_count'])
        })
    
    return {
        'property_ranked_by': property_name,
        'temperature_filter': f"{min_temperature}K - {max_temperature}K" if min_temperature and max_temperature else "All temperatures",
        'total_materials': len(formula_groups),
        'returned_results': len(top_results),
        'results': top_results
    }

@database_agent.tool_plain
def get_material_summary() -> Dict[str, Any]:
    """Get a summary of the database including statistics and top performers."""
    # Basic statistics
    total_materials = df['Formula'].nunique()
    total_measurements = len(df)
    temperature_range = f"{df['temperature(K)'].min():.0f}K - {df['temperature(K)'].max():.0f}K"
    
    # Top ZT materials
    top_zt = df.groupby('Formula')['ZT'].max().sort_values(ascending=False).head(5)
    top_zt_list = [{'formula': formula, 'max_ZT': round(zt, 3)} for formula, zt in top_zt.items()]
    
    # Top power factor materials
    top_pf = df.groupby('Formula')['power_factor(W/mK2)'].max().sort_values(ascending=False).head(5)
    top_pf_list = [{'formula': formula, 'max_power_factor': round(pf, 5)} for formula, pf in top_pf.items()]
    
    # Element distribution
    all_elements = set()
    for formula in df['Formula'].unique():
        # Simple element extraction (this could be improved with proper chemical formula parsing)
        elements = ''.join([c for c in formula if c.isupper() or c.islower()])
        all_elements.update(elements.split())
    
    return {
        'database_summary': {
            'total_unique_materials': total_materials,
            'total_measurements': total_measurements,
            'temperature_range': temperature_range,
            'unique_elements': len(all_elements),
            'elements': sorted(list(all_elements))
        },
        'top_performers': {
            'highest_ZT': top_zt_list,
            'highest_power_factor': top_pf_list
        }
    }

#----------

async def call_database_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call DatabaseAgent to execute a query on thermoelectric materials DB."""
    agent_name = "DatabaseAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")

    user_prompt = f"Current Task of your role: {message2agent}"

    result = await database_agent.run(user_prompt, deps=deps)

    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")

    return output
