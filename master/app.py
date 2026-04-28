from fastapi import FastAPI
import requests
from transformers import GPT2Tokenizer

app = FastAPI()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

WORKER1_URL = "http://localhost:8001/process"
WORKER2_URL = "http://localhost:8002/process"


@app.get("/generate")
def generate(prompt: str):

    # Tokenization
    tokens = tokenizer.encode(prompt)

    print("Sending to Worker1...")

    # Worker1 call
    res1 = requests.post(
        WORKER1_URL,
        json={"input_ids": [tokens]}
    ).json()

    hidden_states = res1["hidden_states"]

    print("Received from Worker1")

    print("Sending to Worker2...")

    # Worker2 call
    res2 = requests.post(
        WORKER2_URL,
        json={"hidden_states": hidden_states}
    ).json()

    print("Received from Worker2")

    return {
        "output": res2
    }
