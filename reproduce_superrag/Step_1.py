import os
import json
import time
import numpy as np

from superrag import SuperRAG
from superrag.utils import EmbeddingFunc
from superrag.llm import openai_complete_if_cache, openai_embedding
from superrag.llm import ollama_model_complete, ollama_embedding

# os.environ["OPENAI_API_KEY"] = ""

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "solar-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="solar-embedding-1-large-query",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
    )



def insert_text(rag, file_path):
    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 1
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


cls = "agriculture"
WORKING_DIR = f"./datasets/{cls}"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# # # superRAG with solar-mini
# rag = SuperRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=llm_model_func,
#     embedding_func=EmbeddingFunc(
#         embedding_dim=4096, max_token_size=8192, func=embedding_func
#     ),
# )

# superRAG with qwen2 model
rag = SuperRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2m",
    # llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)


insert_text(rag, f"./datasets/unique_contexts/{cls}_unique_contexts.json")
