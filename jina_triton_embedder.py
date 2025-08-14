# jina_triton_embedder.py

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional,Union

import aiohttp
import numpy as np
from pydantic import Field
from transformers import AutoTokenizer

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

# --- Configuration Class for your Embedder ---
class JinaV3TritonEmbedderConfig(EmbedderConfig):
    """
    Configuration for the JinaV3TritonEmbedder.
    
    This configures the connection to a Triton server running separate
    Jina V3 query and passage embedding models.
    """
    triton_url: str = Field(
        description="Base URL for the Triton Inference Server, e.g., 'http://localhost:8000'"
    )
    query_model_name: str = Field(
        default="jina_query",
        description="Name of the query embedding model in Triton."
    )
    passage_model_name: str = Field(
        default="jina_passage",
        description="Name of the passage/document embedding model in Triton."
    )
    tokenizer_name: str = Field(
        default="jinaai/jina-embeddings-v3",
        description="Hugging Face tokenizer name for Jina V3."
    )
    triton_output_name: str = Field(
        default="text_embeds",
        description="The name of the output tensor from the Triton model."
    )
    request_timeout: int = Field(
        default=60,
        description="Request timeout in seconds."
    )

# --- Graphiti-Compatible Embedder Client ---
class JinaV3TritonEmbedder(EmbedderClient):
    """
    An embedder client that connects to Jina V3 models hosted on Triton,
    compatible with the graphiti-core framework.
    """

    def __init__(
        self,
        config: JinaV3TritonEmbedderConfig,
        batch_size: int = 1,
        client_session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initializes the embedder.

        Args:
            config: The configuration object with Triton and model details.
            batch_size: The number of texts to process in a single batch.
            client_session: An optional aiohttp session. If not provided, one will be created.
        """
        # The super().__init__ call has been removed.
        # We directly initialize the instance attributes.
        self.config = config
        self.batch_size = batch_size  # Make sure to set the batch_size here
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        
        # Use a provided session or create a new one
        self._client_session = client_session
        self._owns_session = client_session is None
        
        logger.info(f"JinaV3TritonEmbedder configured for Triton at {self.config.triton_url}")

    async def _get_client_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession."""
        if self._client_session is None:
            self._client_session = aiohttp.ClientSession()
        return self._client_session

    def _build_triton_payload(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> Dict[str, Any]:
        """Constructs the JSON payload for Triton."""
        return {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": list(input_ids.shape),
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist()
                },
                {
                    "name": "attention_mask",
                    "shape": list(attention_mask.shape),
                    "datatype": "INT64",
                    "data": attention_mask.flatten().tolist()
                }
            ],
            "outputs": [{"name": self.config.triton_output_name}]
        }
    
    async def _embed_batch(self, texts: List[str], model_name: str) -> List[List[float]]:
        """
        Asynchronously tokenizes, sends a request to Triton, and post-processes a single batch.
        """
        if not texts:
            return []

        api_url = f"{self.config.triton_url.rstrip('/')}/v2/models/{model_name}/infer"
        
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=8192, return_tensors="np"
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        
        payload = self._build_triton_payload(input_ids, attention_mask)
        
        session = await self._get_client_session()
        
        try:
            async with session.post(
                api_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.config.request_timeout
            ) as response:
                response.raise_for_status()
                response_json = await response.json()

            output_data = next(
                (out for out in response_json['outputs'] if out['name'] == self.config.triton_output_name), 
                None
            )
            if output_data is None:
                raise ValueError(f"Triton response did not contain '{self.config.triton_output_name}' output.")

            shape = output_data['shape']
            flat_embeddings = np.array(output_data['data'], dtype=np.float32)
            last_hidden_state = flat_embeddings.reshape(shape)

            # Perform mean pooling using the attention mask
            input_mask_expanded = np.expand_dims(attention_mask, -1)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = np.maximum(input_mask_expanded.sum(1), 1e-9)
            pooled_embeddings = sum_embeddings / sum_mask

            # Perform L2 normalization
            normalized_embeddings = pooled_embeddings / np.linalg.norm(
                pooled_embeddings, ord=2, axis=1, keepdims=True
            )
            
            return normalized_embeddings.tolist()

        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP Error connecting to Triton: {e.status} {e.message}")
            if response:
                logger.error(f"Response Body: {await response.text()}")
            raise
        except Exception as e:
            logger.error(f"Failed to get embeddings from Triton at {api_url}. Error: {e}")
            raise

    async def create(self, input_data: Union[str, List[str]]) -> List[float]:
        """
        Creates an embedding for a single input query.
        Handles cases where the input is a string or a list containing one string.
        This method uses the QUERY model.
        """
        texts_to_embed = []
        if isinstance(input_data, str):
            texts_to_embed = [input_data]
        elif isinstance(input_data, list):
            texts_to_embed = input_data
        else:
            raise TypeError(f"create() expects a string or list of strings, but got {type(input_data)}")

        if not texts_to_embed:
            raise ValueError("input_data cannot be empty.")

        # Pass the flat list directly to the batch embedder.
        embeddings = await self._embed_batch(texts_to_embed, self.config.query_model_name)

        if not embeddings:
            raise ValueError("API returned no embedding for the input.")
        
        # The 'create' method's contract is to return a single embedding vector.
        return embeddings[0]
    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Creates embeddings for a batch of strings.
        This method uses the PASSAGE model.
        """
        if not input_data_list:
            return []

        all_embeddings = []
        for i in range(0, len(input_data_list), self.batch_size):
            batch_texts = input_data_list[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch_texts, self.config.passage_model_name)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    async def close(self):
        """Close the underlying aiohttp client session if we own it."""
        if self._client_session and self._owns_session:
            await self._client_session.close()
            self._client_session = None