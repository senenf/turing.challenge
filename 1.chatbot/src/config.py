import os
import json

from dotenv import load_dotenv

load_dotenv("utils/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "No OPENAI_API_KEY found in utils/.env!"

# Parse models
model_names = json.loads(os.getenv("MODEL_NAMES_JSON"))
embedding_models = json.loads(os.getenv("EMBEDDING_MODELS_JSON"))
# Access models
MODEL_NAME = model_names.get("general_purpose", os.getenv("DEFAULT_MODEL_NAME"))
EMBEDDING_MODEL = embedding_models.get("small", os.getenv("DEFAULT_EMBEDDING_MODEL"))
