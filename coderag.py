from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import faiss
import requests
import json

tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')


def get_code_embedding(code):
    inputs = tokenizer(code, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def get_faiss_index_for_embeddings(code_snippets):

    # Compute embeddings
    embeddings = np.vstack([get_code_embedding(snippet).detach().numpy() for snippet in code_snippets])

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Print the shape of the embedding
    print("Embedding shape:", embeddings.shape)
    print("Embedding:", embeddings)

    return index


def retrieve_similar_code(query_code,index, k=1):
    query_embedding = get_code_embedding(query_code).detach().numpy()
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def call_llm(query_prompt,rag_index,model="llamma3.1"):
    if model == "llamma3.1":
        distances, indices = retrieve_similar_code(query_code, rag_index)
        for idx in indices[0]:
            closest_documents = code_snippets[idx]
        print(closest_documents)
        final_query_prompt = query_prompt + closest_documents

        url = "http://localhost:11434/api/generate"

        # Define the payload
        payload = {
            "model": "llama3",
            "prompt": final_query_prompt,
            "stream": False
        }

        # Convert the payload to a JSON string
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(payload), headers=headers)

        # Check the response
        if response.status_code == 200:
            print("Response received:")
            print(response.json())  # Or response.text if the response is not JSON
        else:
            print(f"Failed to get a response. Status code: {response.status_code}")
            print("Response:", response.text)



code_snippets = [
        "this is fibonacci code: def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
        "this is quicksort code: def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
    ]
rag_index = get_faiss_index_for_embeddings(code_snippets)
query_code = "Can you find quicksort code in my code base. Properly query the vector database"
call_llm(query_code,rag_index)