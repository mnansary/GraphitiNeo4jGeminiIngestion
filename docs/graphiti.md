# Graphiti Prompts

Below is a breakdown of the prompts used, categorized by their function in the graph-building pipeline.

### 1. Node & Attribute Extraction

These prompts are responsible for identifying entities in the text and populating their attributes.

| Prompt Function (`file`) | Purpose | Input Components Influencing Token Count | Output Structure & Token Factors |
| --- | --- | --- | --- |
| **`extract_message`**, **`extract_json`**, **`extract_text`** (`extract_nodes.py`) | Extracts and classifies entities from conversational messages, JSON, or plain text. | <ul><li>`entity_types`: The number and verbosity of defined entity schemas.</li><li>`previous_episodes`: The number and length of past conversational turns.</li><li>`episode_content`: The length of the current message being processed.</li></ul> | A JSON object containing a list of `ExtractedEntity` objects. Output tokens scale directly with the number of entities found. |
| **`classify_nodes`** (`extract_nodes.py`) | Assigns a specific entity type to previously extracted nodes based on conversational context. | <ul><li>`previous_episodes`, `episode_content`: The amount of conversational history.</li><li>`extracted_entities`: The number of entities needing classification.</li><li>`entity_types`: The number of possible classifications.</li></ul> | A JSON object with a list of `EntityClassificationTriple`. Output tokens depend on the number of entities classified. |
| **`extract_attributes`** (`extract_nodes.py`) | Fills in the specific attributes of an entity based on its Pydantic model definition and conversational context. | <ul><li>`previous_episodes`, `episode_content`: The amount of conversational history.</li><li>`node`: The JSON schema of the specific entity type, which can be complex.</li></ul> | A JSON object matching the dynamic schema of the entity's attributes. Output tokens depend heavily on the complexity of the entity's Pydantic model. |
| **`extract_summary`**, **`summarize_context`** (`extract_nodes.py`, `summarize_nodes.py`) | Generates or updates a summary for an entity based on new information. | <ul><li>`previous_episodes`, `episode_content`: The amount of text to be summarized.</li><li>`node`: The entity's existing data, including its previous summary.</li></ul> | A JSON object with a `summary` field. Output tokens are guided by the "under 250 words" constraint. |

---

### 2. Edge (Fact) Extraction

These prompts identify the relationships or facts that connect the entities.

| Prompt Function (`file`) | Purpose | Input Components Influencing Token Count | Output Structure & Token Factors |
| --- | --- | --- | --- |
| **`edge`** (`extract_edges.py`) | Extracts factual relationships (triples) between entities mentioned in the text. | <ul><li>`edge_types`: Definitions of possible relationships.</li><li>`previous_episodes`, `episode_content`: The conversational context.</li><li>`nodes`: The list of identified entities in the current context.</li><li>`reference_time`: Current timestamp for resolving relative dates.</li></ul> | A JSON object containing a list of `Edge` objects. Output tokens scale with the number of facts extracted. |
| **`extract_attributes`** (`extract_edges.py`) | Populates the attributes of a specific relationship based on its predefined schema. | <ul><li>`episode_content`, `reference_time`: Context for extraction.</li><li>`fact`: The schema definition for the specific edge type.</li></ul> | A JSON object matching the dynamic schema of the edge's attributes. Token count depends on the complexity of the edge's Pydantic model. |
| **`v1`** (`extract_edge_dates.py`) | Extracts start (`valid_at`) and end (`invalid_at`) dates for a given fact. | <ul><li>`previous_episodes`, `current_episode`: The conversational context.</li><li>`edge_fact`: The text of the fact being analyzed.</li></ul> | A JSON object with two optional ISO 8601 date strings. The output token count is small and relatively fixed. |

---

### 3. Deduplication and Invalidation

These prompts are crucial for maintaining the integrity of the knowledge graph by merging duplicate information and resolving contradictions.

| Prompt Function (`file`) | Purpose | Input Components Influencing Token Count | Output Structure & Token Factors |
| --- | --- | --- | --- |
| **`node`** (`dedupe_nodes.py`) | Determines if a newly extracted entity is a duplicate of an existing one. | <ul><li>`previous_episodes`, `episode_content`: Conversational context.</li><li>`extracted_node`: The new entity being checked.</li><li>`existing_nodes`: The list of potential duplicates from the graph.</li></ul> | A JSON object (`NodeDuplicate`) containing the ID of the new entity and the index of its duplicate. The output token count is small. |
| **`resolve_edge`** (`dedupe_edges.py`) | Checks if a new fact is a duplicate of or contradicts existing facts in the graph. | <ul><li>`new_edge`: The new fact.</li><li>`existing_edges`, `edge_invalidation_candidates`: Lists of facts from the graph to compare against.</li></ul> | A JSON object (`EdgeDuplicate`) listing duplicate and contradicted fact IDs. The output size depends on the number of matches found. |
| **`v2`** (`invalidate_edges.py`) | Identifies which existing facts are explicitly contradicted by a new fact. | <ul><li>`existing_edges`: The list of facts to check for contradictions.</li><li>`new_edge`: The new, potentially contradicting fact.</li></ul> | A JSON object (`InvalidatedEdges`) containing a list of contradicted fact IDs. Output tokens scale with the number of contradictions found. |

---

### 4. Quality Control & Evaluation

These prompts appear to be designed for internal testing and evaluation of the graph-building process.

| Prompt Function (`file`) | Purpose | Input Components Influencing Token Count | Output Structure & Token Factors |
| --- | --- | --- | --- |
| **`reflexion`** (`extract_nodes.py`, `extract_edges.py`) | A "self-correction" prompt that identifies entities or facts that might have been missed in the initial extraction pass. | <ul><li>`previous_episodes`, `episode_content`: The original text.</li><li>`extracted_entities` or `extracted_facts`: The list of what was already found.</li></ul> | A JSON object listing the names of missed entities or facts. Output size depends on how many items were missed. |
| **`eval_add_episode_results`** (`eval.py`) | Compares a "baseline" graph extraction result against a "candidate" result to judge which is higher quality. | <ul><li>`previous_messages`, `message`: The source text.</li><li>`baseline`, `candidate`: Two sets of extracted graph data (nodes/edges) serialized as JSON. This can be very large.</li></ul> | A JSON object (`EvalAddEpisodeResults`) with a boolean and a short reasoning string. The output is small. |



Here is a breakdown of which models are best suited for each task category:

| Task Category | Primary Model Recommendation | Justification | Fallback Options |
| :--- | :--- | :--- | :--- |
| **Core Extraction**<br>(Nodes & Edges) | **`gemini-2.5-flash`** | This is your main, high-frequency workload. It requires a strong balance of performance (10 RPM), quality, and cost. `gemini-2.5-flash` has `thinking` and `structured_outputs` capabilities, ensuring high-quality JSON extraction, and a large 65,536 token output limit for complex episodes. | `gemini-2.5-flash-lite` (for cost savings), `gemini-2.5-pro` (for maximum quality) |
| **Complex Reasoning**<br>(Deduplication, Invalidation, Reflexion) | **`gemini-2.5-pro`** | These tasks are critical for graph integrity. An error here (e.g., incorrectly merging two distinct people) is more damaging than missing a single fact during initial extraction. `gemini-2.5-pro` offers the highest reasoning quality. Its lower RPM (5) is acceptable as these validation steps may be less frequent than raw extraction. | `gemini-2.5-flash` |
| **Summarization & Attribute Filling**<br>(Node Summaries, Attribute Extraction) | **`gemini-2.5-flash-lite`** | These tasks are important but generally less complex than coreference resolution or contradiction detection. `gemini-2.5-flash-lite` provides excellent performance (15 RPM), all necessary capabilities, and is the most cost-effective, making it perfect for these high-volume, moderate-complexity jobs. | `gemini-2.5-flash` |
| **Simple, High-Throughput Tasks**<br>(e.g., Date Extraction) | **`gemini-2.0-flash-lite`** | For highly constrained tasks with small, predictable outputs (like extracting just `valid_at` and `invalid_at` dates), this model's very high rate limit (30 RPM) is advantageous. However, its smaller 8,192 token output limit makes it unsuitable for extracting lists of nodes or edges. | `gemini-2.5-flash-lite` |
| **Internal Evaluation**<br>(Comparing baseline vs. candidate extractions) | **`gemini-2.5-pro`** | This task requires nuanced judgment to compare two large, complex JSON outputs and provide a reasoned explanation. For the most reliable evaluation and clear reasoning, the highest-quality model is the best choice. | `gemini-2.5-flash` |

