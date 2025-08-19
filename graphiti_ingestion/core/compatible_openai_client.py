import logging
from typing import Any

from graphiti_core.llm_client.config import ModelSize
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class CompatibleOpenAIClient(OpenAIClient):
    """
    An OpenAI client that is more compatible with instruction-tuned models like Gemma.

    This client overrides the response generation to handle common failure
    modes where the model returns a valid JSON but with incorrect key names
    or structure. This is a complete compatibility layer for the `add_episode` pipeline.
    """

    def _fix_entity_extraction_output(self, raw_dict: dict) -> dict:
        """Fixes JSON for the Entity Extraction step."""
        fixed_dict = raw_dict.copy()
        if "entities" in fixed_dict and "extracted_entities" not in fixed_dict:
            logger.info("Fixing top-level key: 'entities' -> 'extracted_entities'")
            fixed_dict["extracted_entities"] = fixed_dict.pop("entities")

        if "extracted_entities" in fixed_dict and isinstance(fixed_dict["extracted_entities"], list):
            logger.info("Fixing nested entity keys: 'entity_name' or 'entity_text' -> 'name'")
            for entity in fixed_dict["extracted_entities"]:
                if isinstance(entity, dict):
                    # Handle both variations seen from text vs. json prompts
                    if "entity_name" in entity:
                        entity["name"] = entity.pop("entity_name")
                    elif "entity_text" in entity:
                        entity["name"] = entity.pop("entity_text")
        return fixed_dict

    def _fix_node_resolution_output(self, raw_dict: dict) -> dict:
        """Fixes JSON for the Node Resolution step."""
        fixed_dict = raw_dict.copy()
        if "entity_resolutions" in fixed_dict and isinstance(fixed_dict["entity_resolutions"], list):
            logger.info("Fixing node resolutions: adding missing 'duplicates' field.")
            for resolution in fixed_dict["entity_resolutions"]:
                if isinstance(resolution, dict) and "duplicates" not in resolution:
                    resolution["duplicates"] = []
        return fixed_dict

    def _fix_edge_extraction_output(self, raw_dict: dict) -> dict:
        """Fixes JSON for the Edge Extraction step."""
        fixed_dict = raw_dict.copy()
        if "facts" in fixed_dict and "edges" not in fixed_dict:
            logger.info("Fixing top-level key: 'facts' -> 'edges'")
            fixed_dict["edges"] = fixed_dict.pop("facts")

        if "edges" in fixed_dict and isinstance(fixed_dict["edges"], list):
            logger.info("Fixing nested edge keys based on complete schema...")
            for edge in fixed_dict["edges"]:
                if isinstance(edge, dict):
                    if "subject_id" in edge:
                        edge["source_entity_id"] = edge.pop("subject_id")
                    if "object_id" in edge:
                        edge["target_entity_id"] = edge.pop("object_id")
                    
                    fact_from_model = edge.pop("fact", None)
                    fact_text_from_model = edge.pop("fact_text", None)
                    if fact_text_from_model:
                        edge["fact"] = fact_text_from_model
                    if fact_from_model:
                        edge["relation_type"] = fact_from_model
        return fixed_dict

    def _fix_edge_duplicate_output(self, raw_dict: dict) -> dict:
        """Fixes JSON for the Edge Duplicate/Contradiction check step."""
        fixed_dict = raw_dict.copy()
        if "duplicates" in fixed_dict:
            logger.info("Fixing edge duplicate key: 'duplicates' -> 'duplicate_facts'")
            fixed_dict["duplicate_facts"] = fixed_dict.pop("duplicates")
        if "contradictions" in fixed_dict:
            logger.info("Fixing edge contradiction key: 'contradictions' -> 'contradicted_facts'")
            fixed_dict["contradicted_facts"] = fixed_dict.pop("contradictions")
        return fixed_dict

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """
        Generates a response and applies compatibility fixes before final validation.
        """
        raw_response_dict = await super().generate_response(
            messages, response_model=None, max_tokens=max_tokens, model_size=model_size
        )

        if not response_model:
            return raw_response_dict

        try:
            response_model.model_validate(raw_response_dict)
            return raw_response_dict
        except ValidationError:
            model_name = response_model.__name__
            logger.warning(f"Initial validation for '{model_name}' failed. Applying comprehensive compatibility fixes.")
            
            if model_name == "ExtractedEntities":
                cleaned_dict = self._fix_entity_extraction_output(raw_response_dict)
            elif model_name == "NodeResolutions":
                cleaned_dict = self._fix_node_resolution_output(raw_response_dict)
            elif model_name == "ExtractedEdges":
                cleaned_dict = self._fix_edge_extraction_output(raw_response_dict)
            elif model_name == "EdgeDuplicate":
                cleaned_dict = self._fix_edge_duplicate_output(raw_response_dict)
            else:
                cleaned_dict = raw_response_dict

            try:
                response_model.model_validate(cleaned_dict)
                logger.info(f"Successfully fixed and validated the LLM response for '{model_name}'.")
                return cleaned_dict
            except ValidationError as final_e:
                logger.error(f"FATAL: Validation failed for '{model_name}' even after applying all fixes: {final_e}")
                logger.debug(f"Original dictionary from LLM: {raw_response_dict}")
                logger.debug(f"Cleaned dictionary that failed: {cleaned_dict}")
                raise final_e