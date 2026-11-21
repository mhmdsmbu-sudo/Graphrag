import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j Connection Settings (Aura Cloud or Local Docker)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# LLM & Embedding Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384  # Standard dimension for intfloat/multilingual-e5-small

# Data Paths
RAW_DATA_PATH = "data/raw/tech_startups_dataset_STUDENT.xlsx"
CLEAN_DATA_PATH = "data/cleaned/cleaned_tech_startups.xlsx"
