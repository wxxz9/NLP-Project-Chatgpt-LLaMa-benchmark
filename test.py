import json
import requests
import openai
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
                if not qa['is_impossible'] and len(qa['answers']) > 0:
                    items.append({
                        "question": qa['question'],
                        "context": context,
                        "answer_text": qa['answers'][0]['text'],
                        "answer_start": qa['answers'][0]['answer_start']
                    })
                if len(items) >= 50:
                    return items
    return items


def ask_chatgpt(question, context, api_key):
    prompt = f"Context: {context}\nQuestion: {question}\nProvide a concise, exact answer based on the context; direct answer only, no need complete sentence(Extract from the following context the minimal span word for word that best answers the question); follow the original upper&lowercase."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=30,
        temperature=0.3,
        api_key=api_key
    )
    answer = response['choices'][0]['message']['content'].strip()
    return answer


def exact_match_score(prediction, truth):
    return prediction.strip().lower() == truth.strip().lower()

def f1_score(prediction, truth):
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
    squad_data = download_squad_data()
    items = extract_questions_and_contexts(squad_data)
    request_count = 0
    total_em_score = 0
    total_f1_score = 0

    with open('output_answers.txt', 'w') as file:
        for item in items:
            answer = ask_chatgpt(item['question'], item['context'], api_key)
            exact_match = exact_match_score(answer, item['answer_text'])
            f1 = f1_score(answer, item['answer_text'])
            total_em_score += exact_match
            total_f1_score += f1

            output = f"Question: {item['question']}\n"
            output += f"Correct Answer: {item['answer_text']}\n"
            output += f"Model Answer: {answer}\n"
            output += f"Exact Match: {exact_match}\n"
            output += f"F1 Score: {f1:.2f}\n\n"
            print(output)
            file.write(output)
            
            request_count += 1
            if request_count >= 200:
                print("Reached daily limit of 200 requests. Please try again tomorrow.")
                break

        avg_em_score = total_em_score / len(items)
        avg_f1_score = total_f1_score / len(items)
        print(f"Average Exact Match Score: {avg_em_score:.2f}")
        print(f"Average F1 Score: {avg_f1_score:.2f}")


if __name__ == "__main__":
    api_key = ""
    main(api_key)
