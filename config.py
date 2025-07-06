import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
THRESHOLD = 0.5
NUM_KEYWORDS = 10
NUM_SEARCH_RESULTS = 10

if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
    raise ValueError("Missing Google API credentials in .env file")
