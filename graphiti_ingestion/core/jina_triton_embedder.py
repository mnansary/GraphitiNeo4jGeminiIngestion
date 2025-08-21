# graphiti_ingestion/core/jina_triton_embedder.py

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp
import numpy as np
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from pydantic import Field
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class JinaV3TritonEmbedderConfig(EmbedderConfig):
    """Configuration for the JinaV3TritonEmbedder."""

    triton_url: str = Field(
        description="Base URL for the Triton Inference Server"
    )
    triton_request_timeout: int = Field(
        default=60, description="Request timeout in seconds for connecting to Triton."
    )
    query_model_name: str = Field(
        default="jina_query", description="Name of the query embedding model in Triton."
    )
    passage_model_name: str = Field(
        default="jina_passage",
        description="Name of the passage/document embedding model in Triton.",
    )
    tokenizer_name: str = Field(
        default="jinaai/jina-embeddings-v3",
        description="Hugging Face tokenizer name for Jina V3.",
    )
    triton_output_name: str = Field(
        default="text_embeds",
        description="The name of the output tensor from the Triton model.",
    )
    batch_size: int = Field(
        default=4,
        description="Number of texts to process in a single batch request to Triton.",
    )


class JinaV3TritonEmbedder(EmbedderClient):
    """
    An async embedder client for Jina V3 models on Triton, compatible with graphiti-core.
    """

    def __init__(
        self,
        config: JinaV3TritonEmbedderConfig,
        client_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__()
        self.config = config
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name, trust_remote_code=True
            )
        except Exception as e:
            logger.critical(f"Failed to load Hugging Face tokenizer '{self.config.tokenizer_name}'. Error: {e}")
            raise

        self._client_session = client_session
        self._owns_session = client_session is None
        logger.info(f"JinaV3TritonEmbedder configured for Triton at {self.config.triton_url}")

    @property
    async def client_session(self) -> aiohttp.ClientSession:
        if self._client_session is None:
            self._client_session = aiohttp.ClientSession()
        return self._client_session

    def _build_triton_payload(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> Dict[str, Any]:
        """Constructs the JSON payload for Triton."""
        return {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": list(input_ids.shape),
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist(),
                },
                {
                    "name": "attention_mask",
                    "shape": list(attention_mask.shape),
                    "datatype": "INT64",
                    "data": attention_mask.flatten().tolist(),
                },
            ],
            "outputs": [{"name": self.config.triton_output_name}],
        }

    async def _embed_batch(
        self, texts: List[str], model_name: str
    ) -> List[List[float]]:
        """Asynchronously tokenizes, sends a request to Triton, and post-processes a single batch."""
        if not texts:
            return []

        api_url = f"{str(self.config.triton_url).rstrip('/')}/v2/models/{model_name}/infer"
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=8192, return_tensors="np"
        )
        payload = self._build_triton_payload(
            tokens["input_ids"].astype(np.int64),
            tokens["attention_mask"].astype(np.int64),
        )
        session = await self.client_session
        timeout = aiohttp.ClientTimeout(total=self.config.triton_request_timeout)

        try:
            async with session.post(api_url, data=json.dumps(payload), timeout=timeout) as response:
                response.raise_for_status()
                response_json = await response.json()

            output_data = next(
                (out for out in response_json["outputs"] if out["name"] == self.config.triton_output_name),
                None,
            )
            if output_data is None:
                raise ValueError(f"Triton response did not contain '{self.config.triton_output_name}' output.")

            shape = output_data["shape"]
            last_hidden_state = np.array(output_data["data"], dtype=np.float32).reshape(shape)
            
            # Perform mean pooling and L2 normalization (correct logic from your test script)
            attention_mask = tokens["attention_mask"]
            input_mask_expanded = np.expand_dims(attention_mask, -1)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = np.maximum(input_mask_expanded.sum(1), 1e-9)
            pooled_embeddings = sum_embeddings / sum_mask
            normalized_embeddings = pooled_embeddings / np.linalg.norm(
                pooled_embeddings, ord=2, axis=1, keepdims=True
            )
            return normalized_embeddings.tolist()

        except asyncio.TimeoutError:
            logger.error(f"TimeoutError: Request to Triton at {api_url} timed out after {self.config.triton_request_timeout} seconds.")
            raise
        except aiohttp.ClientConnectorError as e:
            logger.error(f"ClientConnectorError: Could not connect to Triton at {api_url}. Is the URL correct and the server reachable from this machine? Error: {e}")
            raise
        except aiohttp.ClientResponseError as e:
            error_body = await e.response.text()
            logger.error(f"HTTP Error {e.status} from Triton: {e.message}. Response: {error_body}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while getting embeddings from Triton at {api_url}.", exc_info=True)
            raise

    async def create(self, input_data: Union[str, List[str]]) -> List[float]:
        """Creates an embedding for a single query string using the QUERY model."""
        text_to_embed = input_data[0] if isinstance(input_data, list) else input_data
        if not text_to_embed or not isinstance(text_to_embed, str):
            raise TypeError(f"create() expects a non-empty string, but got {type(input_data)}")

        embeddings = await self._embed_batch([text_to_embed], self.config.query_model_name)
        if not embeddings:
            raise ValueError("API returned no embedding for the input.")
        return embeddings[0]

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """Creates embeddings for a batch of strings using the PASSAGE model."""
        if not input_data_list:
            return []
        all_embeddings = []
        for i in range(0, len(input_data_list), self.config.batch_size):
            batch_texts = input_data_list[i : i + self.config.batch_size]
            batch_embeddings = await self._embed_batch(batch_texts, self.config.passage_model_name)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    async def close(self):
        """Closes the underlying aiohttp client session if this instance created it."""
        if self._client_session and self._owns_session:
            await self._client_session.close()
            self._client_session = None