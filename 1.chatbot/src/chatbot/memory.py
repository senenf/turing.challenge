import os
import json

from dotenv import load_dotenv
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI


load_dotenv("utils/.env")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2000))

model_json = os.getenv("MODEL_NAMES_JSON")
if not model_json:
    raise EnvironmentError("Missing MODEL_NAMES_JSON in environment variables")
model_names = json.loads(model_json)
MODEL_NAME = model_names.get("general_purpose", os.getenv("DEFAULT_MODEL_NAME"))

def get_memory():
    print(f"Using model: {MODEL_NAME} with max tokens: {MAX_TOKENS}")
    return ConversationSummaryBufferMemory(
        llm=ChatOpenAI(model=MODEL_NAME, temperature=0.0),
        memory_key="chat_history",
        max_token_limit=MAX_TOKENS,
        return_messages=True
    )


