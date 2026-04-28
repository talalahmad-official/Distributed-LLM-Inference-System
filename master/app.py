from fastapi import FastAPI
from pydantic import BaseModel
import requests
from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F

app = FastAPI()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Local deployment
WORKER1_URL = "http://localhost:8001/process"
WORKER2_URL = "http://localhost:8002/process"

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 10


@app.get("/generate")
def generate(prompt: str, max_tokens: int = 10):
    tokens = tokenizer.encode(prompt)

    for _ in range(max_tokens):
        try:
            res1 = requests.post(
                WORKER1_URL,
                json={"input_ids": tokens},
                timeout=30
            ).json()

            if "error" in res1:
                return {"error": f"Worker1: {res1['error']}"}

            res2 = requests.post(
                WORKER2_URL,
                json={"hidden_states": res1["hidden_states"]},
                timeout=30
            ).json()

            if "error" in res2:
                return {"error": f"Worker2: {res2['error']}"}

        except requests.exceptions.RequestException as e:
            return {"error": f"Connection failed: {str(e)}"}

        logits = torch.tensor(res2["logits"])
        logits_last = logits[0, -1]

        # Repetition Penalty
        for token_id in set(tokens):
            if logits_last[token_id] > 0:
                logits_last[token_id] /= 2.5  # positive ko chota karo
            else:
                logits_last[token_id] *= 2.5  # negative ko aur negative karo

        # Temperature — diversity
        temperature = 0.9
        logits_last = logits_last / temperature

        # Top-k sampling
        values, indices = torch.topk(logits_last, 50)
        probs = F.softmax(values, dim=-1)
        next_token = indices[torch.multinomial(probs, 1)].item()
        tokens.append(next_token)

    return {"generated_text": tokenizer.decode(tokens)}
