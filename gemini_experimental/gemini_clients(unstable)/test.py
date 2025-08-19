import asyncio
import logging
import os
import yaml
from datetime import datetime, timezone

# --- Graphiti Core ---
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType

# --- Custom Components ---
from gemini_api_manager.core import ComprehensiveManager
from gemini_api_manager.utils.logging_config import setup_logging
from managed_gemini_client import ManagedGeminiClient
from jina_triton_embedder import JinaV3TritonEmbedder, JinaV3TritonEmbedderConfig
from managed_gemini_reranker import ManagedGeminiReranker
os.environ['SEMAPHORE_LIMIT'] = '1'
# --- Passage to be Ingested ---
BENGALI_GOVT_PASSAGE = """
মেশিন রিডেবল পাসপোর্ট (এমআরপি) ইস্যু
সেবা প্রাপ্তির সংক্ষিপ্ত বিবরণ	
অন লাইনে/হাতে পূরণকৃত (ছবিসহ সত্যায়িত করতে হয় এমআরপি আবেদন ফরম, পাসপোর্ট ফিস জমাদানের ব্যাংক রসিদ, জাতীয় পরিচয়পত্র/ডিজিটাল জন্ম নিবন্ধন সনদের সত্যায়িত কপি এবং প্রযোজ্য ক্ষেত্রে বিদ্যমান পাসপোর্টের ফটোকপি, সরকারি আদেশ (GO)/ছাড়পত্র (NOC) অবসর গ্রহণের প্রমাণপত্র ও প্রাসঙ্গিক টেকনিক্যাল সনদসমূহের (যেমন: ডাক্তার, ইঞ্জিনিয়ার, ড্রাইভার ইত্যাদি) সত্যায়িত ফটোকপিমগ) এমআরপি আবেদন ফরম বিভাগীয় পাসপোর্ট ও ভিসা অফিস/আঞ্চলিক পাসপোর্ট অফিস/পররাষ্ট্র মন্ত্রণালয়/বিদেশস্থ্ বাংলাদেশ মিশনে আবেদনকারীকে স্বশরীরে উপস্থিত হয়ে দাখিল করতে হয়।

কম্পিউটারে আবেদনকারীর প্রাক পরিচিতি সংক্রান্ত তথ্য এন্ট্রি এবং বায়োমেট্রিক তথ্য (ছবি, আঙুলের ছাপ, ডিজিটাল স্বাক্ষর) গ্রহণ করে আবেদনকারীকে একটি বিতরণ রসিদ প্রদান করা হয়।

কালো তালিকা যাচাই, পেমেন্ট ভেরিফিকেশন, প্রযোজ্য ক্ষেত্রে অনুকূল পুলিশ প্রতিবেদন এবং কর্তৃপক্ষের অনুমোদনের পর পাসপোর্ট পার্সোনালাইজেশন করে নির্ধারিত অফিসসমূহে ডাকযোগে পাঠানো হয়।

নির্ধারিত তারিখে সংশ্লিষ্ট অফিস থেকে আবেদনকারী পাসপোর্ট সংগ্রহ করে থাকেন।

সেবা প্রাপ্তির সময়	
সাধারণ ফিসের ক্ষেত্রে- ১৫ কর্মদিবস পর • জরুরি ফিস এর ক্ষেত্রে- ৭ কর্মদিবস

প্রয়োজনীয় ফি	
১. সাধারণ ফি: ৩০০০/ টাকা + ১৫% ভ্যাট ২. জরুরি ফি: ৬০০০/ টাকা + ১৫% ভ্যাট ৩. সরকারি আদেশে (জিও এর ক্ষেত্রে) বিনামূল্যে

সেবা প্রাপ্তির স্থান	
১. বিভাগীয় পাসপোর্ট ও ভিসা অফিস ২. আঞ্চলিক পাসপোর্ট অফিস ৩. বিদেশস্থ বাংলাদেশ মিশন ৪. পররাষ্ট্র মন্ত্রণালয় (কূটনৈতিক পাসপোর্ট ইস্যুর ক্ষেত্রে)

দায়িত্বপ্রাপ্ত কর্মকর্তা/কর্মচারী	
১. নিয়ন্ত্রণাধীন বিভাগীয় পাসপোর্ট ও ভিসা অফিস/আঞ্চলিক পাসপোর্ট অফিসের দায়িত্বে নিয়োজিত পরিচালক/উপ-পরিচালক/সহকারী পরিচালক/উপ-সহকারী পরিচালক; ২. বিদেশস্থ বাংলাদেশি মিশনসমূহে দায়িত্বপ্রাপ্ত কর্মকর্তা ৩. পররাষ্ট্র মন্ত্রণালয়ের দায়িত্বপ্রাপ্ত কর্মকর্তা

প্রয়োজনীয় কাগজপত্র	
পূরণকৃত ছবিসহ সত্যায়িত এমআরপি আবেদন ফরম, পাসপোর্ট ফিস জমাদানের ব্যাংক রসিদ, জাতীয় পরিচয়পত্র/ডিজিটাল জন্ম নিবন্ধন সনদের সত্যায়িত কপি এবং প্রযোজ্য ক্ষেত্রে বিদ্যমান পাসপোর্টের ফটোকপি, সরকারি আদেশ (GO)/ছাড়পত্র (NOC) অবসর গ্রহণের প্রমাণপত্র ও প্রাসঙ্গিক টেকনিক্যাল সনদসমূহের (যেমন: ডাক্তার, ইঞ্জিনিয়ার, ড্রাইভার ইত্যাদি) সত্যায়িত ফটোকপি।

সেবা প্রাপ্তির শর্তাবলি	
বাংলাদেশের নাগরিক হওয়া, কালো তালিকামুক্ত হওয়া, নির্ধারিত ফিস, প্রযোজ্য ক্ষেত্রে নির্ধারিত সময়ের মধ্যে অনুকূল পুলিশ তদন্ত প্রতিবেদন এবং সরকারি, আধাসরকারি, স্বায়ত্তশাসিত ও রাষ্ট্রায়ত্ত সংস্থার স্থায়ী কর্মকর্তা ও কর্মচারীর ক্ষেত্রে যথাযথ কর্তৃপক্ষের অনুমোদন।

সংশ্লিষ্ট আইন ও বিধি	
১. বাংলাদেশ পাসপোর্ট অর্ডার, ১৯৭৩

২. বাংলাদেশ পাসপোর্ট রুলস, ১৯৭৪

৩. নির্বাহী আদেশ

সেবা প্রদানে ব্যর্থ হলে প্রতিকারকারী কর্মকর্তা	
মহাপরিচালক, বহিরাগমন ও পাসপোর্ট অধিদপ্তর/সিনিয়র সচিব, স্বরাষ্ট্র মন্ত্রণালয়
"""

async def main():
    """
    Main function to initialize all services from config and ingest a single passage.
    """
    # 1. Load Configuration from YAML
    try:
        with open("configs/configs.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("FATAL: configs/configs.yaml not found. Please ensure the file exists.")
        return

    # 2. Setup Logging
    log_config = config['logging']
    setup_logging(
        console_level=log_config['console_level'],
        file_level=log_config['file_level'],
        log_dir=log_config['log_dir'],
        retention=log_config['retention'],
        rotation=log_config['rotation']
    )
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Single Passage Ingestion Test ---")

    jina_embedder = None
    graphiti = None

    try:
        # 3. Initialize Gemini API Manager
        logger.info("Initializing ComprehensiveManager...")
        manager_config = config['gemini_api_manager']
        manager = ComprehensiveManager(
            api_key_csv_path=manager_config['api_key_csv_path'],
            model_config_path=manager_config['model_config_path'],
        )
        logger.info("ComprehensiveManager initialized successfully.")

        # 4. Initialize Custom Managed Gemini Client (LLM)
        logger.info("Initializing ManagedGeminiClient...")
        llm_client_config = config['managed_gemini_client']
        managed_llm_client = ManagedGeminiClient(
            manager=manager,
            config=LLMConfig(temperature=llm_client_config['temperature'])
        )
        logger.info("ManagedGeminiClient initialized successfully.")

        # 5. Initialize Custom Managed Gemini Reranker
        logger.info("Initializing ManagedGeminiRerankerClient...")
        reranker_config = config['managed_gemini_reranker']
        managed_reranker = ManagedGeminiReranker(
            manager=manager,
            config=LLMConfig(model=reranker_config.get('model'))
        )
        logger.info("ManagedGeminiRerankerClient initialized successfully.")


        # 6. Initialize Custom Jina Triton Embedder
        logger.info("Initializing JinaV3TritonEmbedder...")
        embedder_config = config['jina_triton_embedder']
        jina_embedder_config = JinaV3TritonEmbedderConfig(**embedder_config)
        jina_embedder = JinaV3TritonEmbedder(
            config=jina_embedder_config,
            batch_size=embedder_config['batch_size']
        )
        logger.info("JinaV3TritonEmbedder initialized successfully.")

        # 7. Initialize Graphiti with all Custom Components
        logger.info("Initializing Graphiti with custom clients...")
        neo4j_config = config['neo4j']
        graphiti = Graphiti(
            neo4j_config['uri'],
            neo4j_config['user'],
            neo4j_config['password'],
            llm_client=managed_llm_client,
            embedder=jina_embedder,
            cross_encoder=managed_reranker,
        )
        logger.info("Graphiti initialized successfully.")

        # 8. Build Indices and Ingest the Episode
        logger.info("Building graph indices and constraints (if they don't exist)...")
        await graphiti.build_indices_and_constraints()

        logger.info(f"Ingesting passage into the graph...")
        await graphiti.add_episode(
            name="MRP Passport Issuance Guideline",
            episode_body=BENGALI_GOVT_PASSAGE,
            source=EpisodeType.text,
            source_description="Government service guideline for Machine Readable Passports in Bangladesh",
            reference_time=datetime.now(timezone.utc),
        )

        logger.info("=" * 60)
        logger.info("✅ SUCCESS: The passage has been processed and ingested.")
        logger.info("=" * 60)
        logger.info("➡️ NEXT STEP: To see the result, follow these steps:")
        logger.info("   1. Open your browser and go to http://localhost:7474 (your Neo4j Browser).")
        logger.info("   2. Run the following Cypher query in the query bar:")
        logger.info("      MATCH (n) RETURN n LIMIT 50")
        logger.info("   3. You should see a graph with nodes representing entities like 'মেশিন রিডেবল পাসপোর্ট', 'জাতীয় পরিচয়পত্র', etc.")
        logger.info("=" * 60)

    except ConnectionRefusedError as e:
        logger.error(f"FATAL: Connection failed. Is a required service (Neo4j, Redis, or Triton) running? Error: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the process: {e}")
    finally:
        # 9. Cleanly close all connections
        logger.info("Cleaning up resources...")
        if jina_embedder:
            await jina_embedder.close()
            logger.info("Jina embedder connection closed.")
        if graphiti:
            await graphiti.close()
            logger.info("Graphiti (Neo4j) connection closed.")
        logger.info("--- Test script finished. ---")


if __name__ == '__main__':
    asyncio.run(main())