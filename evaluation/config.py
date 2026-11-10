import os
from pathlib import Path
from dotenv import load_dotenv
from deepeval.models import LiteLLMModel
from deepeval.models import GeminiModel

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# # Groq Judge Model
judge_model1 = LiteLLMModel(
    model="groq/openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

judge_model2= GeminiModel(
    model="gemini-3.5-flash-lite",
    
)