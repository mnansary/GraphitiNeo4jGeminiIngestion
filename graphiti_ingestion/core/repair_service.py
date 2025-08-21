# graphiti_ingestion/core/repair_service.py

import json
import logging
from typing import Any, Callable, Dict

from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.dedupe_nodes import NodeResolutions
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.extract_nodes import ExtractedEntities, EntitySummary
from graphiti_core.prompts.invalidate_edges import InvalidatedEdges
from graphiti_core.prompts.models import Message
from graphiti_core.prompts.summarize_nodes import Summary
from pydantic import BaseModel, ValidationError
#from graphiti_ingestion.core.compatible_openai_client import CompatibleOpenAIClient

# Use Python's standard logging module
logger = logging.getLogger(__name__)

# --- Step 1: Define all individual repair functions ---

def _default_fixer(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """A default fixer that does nothing. Used for schemas with no known common errors."""
    return raw_dict

def _fix_entity_extraction(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Applies compatibility fixes for the ExtractedEntities schema."""
    fixed_dict = raw_dict.copy()
    if "entities" in fixed_dict and "extracted_entities" not in fixed_dict:
        logger.warning("[Repair] Renaming key 'entities' -> 'extracted_entities'")
        fixed_dict["extracted_entities"] = fixed_dict.pop("entities")

    if "extracted_entities" in fixed_dict and isinstance(fixed_dict["extracted_entities"], list):
        for entity in fixed_dict["extracted_entities"]:
            if isinstance(entity, dict):
                if "entity_name" in entity and "name" not in entity:
                    entity["name"] = entity.pop("entity_name")
                elif "entity_text" in entity and "name" not in entity:
                    entity["name"] = entity.pop("entity_text")
    return fixed_dict

def _fix_node_resolution(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Applies compatibility fixes for the NodeResolutions schema."""
    fixed_dict = raw_dict.copy()
    if "entity_resolutions" in fixed_dict and isinstance(fixed_dict["entity_resolutions"], list):
        for resolution in fixed_dict["entity_resolutions"]:
            if isinstance(resolution, dict) and "duplicates" not in resolution:
                logger.warning("[Repair] Adding missing 'duplicates' field to a node resolution")
                resolution["duplicates"] = []
    return fixed_dict

def _fix_edge_extraction(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Applies compatibility fixes for the ExtractedEdges schema."""
    fixed_dict = raw_dict.copy()
    if "facts" in fixed_dict and "edges" not in fixed_dict:
        logger.warning("[Repair] Renaming key 'facts' -> 'edges'")
        fixed_dict["edges"] = fixed_dict.pop("facts")

    if "edges" in fixed_dict and isinstance(fixed_dict["edges"], list):
        for edge in fixed_dict["edges"]:
            if isinstance(edge, dict):
                if "subject_id" in edge and "source_entity_id" not in edge:
                    edge["source_entity_id"] = edge.pop("subject_id")
                if "object_id" in edge and "target_entity_id" not in edge:
                    edge["target_entity_id"] = edge.pop("object_id")
                fact_from_model = edge.pop("fact", None)
                fact_text_from_model = edge.pop("fact_text", None)
                if fact_text_from_model and "fact" not in edge:
                    edge["fact"] = fact_text_from_model
                if fact_from_model and "relation_type" not in edge:
                    edge["relation_type"] = fact_from_model
    return fixed_dict

def _fix_edge_duplicate(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Applies compatibility fixes for the EdgeDuplicate schema."""
    fixed_dict = raw_dict.copy()
    if "duplicates" in fixed_dict and "duplicate_facts" not in fixed_dict:
        logger.warning("[Repair] Renaming key 'duplicates' -> 'duplicate_facts'")
        fixed_dict["duplicate_facts"] = fixed_dict.pop("duplicates")
    if "contradictions" in fixed_dict and "contradicted_facts" not in fixed_dict:
        logger.warning("[Repair] Renaming key 'contradictions' -> 'contradicted_facts'")
        fixed_dict["contradicted_facts"] = fixed_dict.pop("contradictions")
    return fixed_dict

# --- Step 2: Create the EXHAUSTIVE, data-driven dispatcher ---

REPAIR_DISPATCHER: Dict[type[BaseModel], Callable[[Dict], Dict]] = {
    # from extract_nodes.py
    ExtractedEntities: _fix_entity_extraction,
    EntitySummary: _default_fixer,
    
    # from dedupe_nodes.py
    NodeResolutions: _fix_node_resolution,
    
    # from extract_edges.py
    ExtractedEdges: _fix_edge_extraction,
    
    # from dedupe_edges.py
    EdgeDuplicate: _fix_edge_duplicate,
    
    # from invalidate_edges.py
    InvalidatedEdges: _default_fixer,
    
    # from summarize_nodes.py
    Summary: _default_fixer,
}

# --- Step 3: The Self-Healing and Orchestration Logic ---

async def _attempt_llm_self_healing(
    llm_client: "CompatibleOpenAIClient", # Add type hint for clarity
    original_messages: list[Message],
    failed_json: str,
    validation_error: str,
    response_model: type[BaseModel],
) -> Dict[str, Any]:
    # ... (the healing_prompt construction is the same) ...
    model_name = response_model.__name__
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)

    healing_prompt = f"""
    Your previous attempt to generate a JSON object failed. You must correct your mistake.
    Here was the original final instruction:
    ---
    {original_messages[-1].content}
    ---
    Here is the invalid JSON you produced:
    ---
    {failed_json}
    ---
    Here is the Pydantic validation error that occurred:
    ---
    {validation_error}
    ---
    TASK:
    Carefully analyze the validation error and the original instruction.
    Correct the JSON object to make it perfectly valid according to the schema.
    Your response MUST be ONLY the corrected JSON object, with no other text, explanations, or markdown formatting.
    The required schema is:
    ---
    {schema_json}
    ---
    """
    healing_messages = [
        Message(role="system", content="You are an expert at correcting malformed JSON data to match a given Pydantic schema."),
        Message(role="user", content=healing_prompt),
    ]

    logger.warning("Attempting LLM self-healing call...")
    
    # --- THIS IS THE CRITICAL CHANGE ---
    # We now call the new _direct_llm_call method to prevent a recursive loop.
    healed_dict = await llm_client._direct_llm_call(healing_messages)
    # --- END OF CHANGE ---

    return healed_dict

async def repair_and_validate(
    raw_dict: Dict[str, Any],
    response_model: type[BaseModel],
    llm_client: OpenAIClient,
    original_messages: list[Message],
) -> Dict[str, Any]:
    model_name = response_model.__name__
    logger.info(f"Processing payload for schema: '{model_name}'")

    try:
        response_model.model_validate(raw_dict)
        logger.info(f"Validation successful for '{model_name}' without any repairs.")
        return raw_dict
    except ValidationError:
        logger.warning(f"Initial validation for '{model_name}' failed. Applying programmatic repairs...")
        
        repair_function = REPAIR_DISPATCHER.get(response_model, _default_fixer)
        programmatically_repaired_dict = repair_function(raw_dict)

        try:
            validated_model = response_model.model_validate(programmatically_repaired_dict)
            logger.info(f"Successfully repaired and validated payload for '{model_name}' programmatically.")
            return validated_model.model_dump(mode="json")
        except ValidationError as programmatic_error:
            logger.error(f"Programmatic repair failed for '{model_name}'. Escalating to LLM self-healing.", exc_info=False)
            logger.debug(f"--- Programmatic Repair Error ---\n{programmatic_error}")

            try:
                healed_dict = await _attempt_llm_self_healing(
                    llm_client,
                    original_messages,
                    json.dumps(programmatically_repaired_dict, indent=2),
                    str(programmatic_error),
                    response_model,
                )
                
                final_validated_model = response_model.model_validate(healed_dict)
                logger.info(f"LLM self-healing was successful for '{model_name}'!")
                return final_validated_model.model_dump(mode="json")
                
            except Exception as final_error:
                logger.critical(f"FATAL: LLM self-healing also failed for '{model_name}'.", exc_info=True)
                logger.debug(f"--- Original dictionary from LLM:\n{raw_dict}")
                logger.debug(f"--- Programmatically Repaired (and failed) dictionary:\n{programmatically_repaired_dict}")
                logger.debug(f"--- Final Validation Error after Self-Healing:\n{final_error}")
                raise final_error