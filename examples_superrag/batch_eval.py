import re
import json
import jsonlines

# from ollama import Client
import ollama


def batch_eval(query_file, result1_file, result2_file, output_file_path):
    with open(query_file, "r") as f:
        data = f.read()

    queries = re.findall(r"- Question \d+: (.+)", data)

    with open(result1_file, "r") as f:
        answers1 = json.load(f)
    answers1 = [i["result"] for i in answers1]

    with open(result2_file, "r") as f:
        answers2 = json.load(f)
    answers2 = [i["result"] for i in answers2]

    requests = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """


        requests.append({"role": "system", "content": sys_prompt})
        requests.append({"role": "user", "content": prompt})

        results = ollama.chat(model='qwen2m', messages=requests)

        with open(output_file_path, mode="a") as writer:
            writer.write(results['message']['content'])


if __name__ == "__main__":
    query_file = './datasets/questions/mix_questions.txt'
    result1_file = './datasets/results/mix_result.json'
    result2_file = './datasets/results/mix_result_2.json'
    output_file = './datasets/compare_mix.json'
    batch_eval(query_file, result1_file, result2_file, output_file)
    print('writing results into ', output_file)
