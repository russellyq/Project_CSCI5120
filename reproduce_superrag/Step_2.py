import os
import json
from transformers import GPT2Tokenizer
import numpy as np
from superrag.llm import ollama_model_complete, ollama_embedding
from superrag import SuperRAG, QueryParam
from superrag.utils import EmbeddingFunc

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

import ollama

def get_summary(context, tot_tokens=2000):
    tokens = tokenizer.tokenize(context)
    half_tokens = tot_tokens // 2

    start_tokens = tokens[1000 : 1000 + half_tokens]
    end_tokens = tokens[-(1000 + half_tokens) : 1000]

    summary_tokens = start_tokens + end_tokens
    summary = tokenizer.convert_tokens_to_string(summary_tokens)

    return summary


clses = ["mix"]
for cls in clses:
    with open(f"./datasets/unique_contexts/{cls}_unique_contexts.json", mode="r") as f:
        unique_contexts = json.load(f)

    summaries = [get_summary(context) for context in unique_contexts]

    total_description = "\n\n".join(summaries)

    prompt = f"""
    Given the following description of a dataset:

    {total_description}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
            - Question 3:
            - Question 4:
            - Question 5:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """
    messages = []
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(model='qwen2m', messages=messages)

    result = response['message']['content']
    
    file_path = f"datasets/questions/{cls}_questions.txt"
    with open(file_path, "w") as file:
        file.write(result)

    print(f"{cls}_questions written to {file_path}")
