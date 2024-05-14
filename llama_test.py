import json
import requests
from llamaapi import LlamaAPI
from collections import Counter

def download_squad_data():
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    response = requests.get(url)
    data = response.json()
    return data

def extract_questions_and_contexts(data):
    items = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                answer_text = "?" if qa['is_impossible'] else qa['answers'][0]['text'] if qa['answers'] else "?"
                items.append({
                    "question": qa['question'],
                    "context": context,
                    "answer_text": answer_text,
                    "is_impossible": qa['is_impossible']
                })
                if len(items) >= 10:
                    return items
    return items

def initialize_llama(api_key):
    return LlamaAPI(api_key)

def ask_llama(llama, question, context):
    api_request_json = {
        "model": "llama3-70b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nProvide a concise, exact answer based on the context; direct answer only, no need complete sentence(Extract from the following context the minimal span word for word that best answers the question); no need quotation marks in the answer; follow the original upper&lowercase; if cannot find answer in the context, then return '?'."}
        ]
    }

    response = llama.run(api_request_json)

    if response.status_code != 200:
        print(f"Failed to fetch data: HTTP Status {response.status_code}")
        print("Response text:", response.text) 
        return "?"

    try:
        response_data = response.json()
        print("Full API Response:", json.dumps(response_data, indent=2))
        answer = response_data['choices'][0]['message']['content']
        return answer.strip()
    except (KeyError, IndexError, TypeError) as e:
        print("Failed to extract answer from response or bad response format:", e)
        return "?"

def exact_match_score(prediction, truth):
    return prediction.strip().lower() == truth.strip().lower()

def f1_score(prediction, truth):
    if truth == "?":
        return 1.0 if prediction == "?" else 0.0
    pred_tokens = prediction.lower().split()
    truth_tokens = truth.lower().split()
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def main(api_key):
    llama = initialize_llama(api_key)
    squad_data = download_squad_data()
    items = extract_questions_and_contexts(squad_data)
    request_count = 0
    total_em_score = 0
    total_f1_score = 0

    with open('llama.txt', 'w') as file:
        for item in items:
            answer = ask_llama(llama, item['question'], item['context'])
            exact_match = exact_match_score(answer, item['answer_text'])
            f1 = f1_score(answer, item['answer_text'])
            total_em_score += exact_match
            total_f1_score += f1

            output = f"Question: {item['question']}\n"
            output += f"Context: {item['context']}\n"
            output += f"Correct Answer: {item['answer_text']}\n"
            output += f"Model Answer: {answer}\n"
            output += f"Exact Match: {exact_match}\n"
            output += f"F1 Score: {f1:.2f}\n\n"
            file.write(output)
            
            request_count += 1
            if request_count % 100 == 0:
                print(f"Processed {request_count} questions.")

        avg_em_score = total_em_score / len(items)
        avg_f1_score = total_f1_score / len(items)
        summary = f"Average Exact Match Score: {avg_em_score:.2f}\nAverage F1 Score: {avg_f1_score:.2f}\n"
        file.write(summary)
        print(summary)

if __name__ == "__main__":
    api_key = ""
    main(api_key)
