from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F
import asyncio
from typing import Dict, Optional
from datetime import datetime
import uuid

app = FastAPI()

# CORS configuration for Frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

WORKER1_URL = "http://localhost:8001/process"
WORKER2_URL = "http://localhost:8002/process"

# Global Databases
queries_db: Dict[str, 'QueryState'] = {}
query_queue = asyncio.Queue()


class QueryState:
    def __init__(self, query_id, prompt, max_tokens):
        self.query_id = query_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.tokens = tokenizer.encode(prompt)
        self.status = "queued"
        self.current_iteration = 0
        self.result = None
        self.error = None
        self.created_at = datetime.now().isoformat()
        self.hidden_states = None
        self.current_stage = "queued"
        self.last_updated = datetime.now()
        self.worker1_pid = None
        self.worker2_pid = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 10


# --- Endpoints ---

@app.post("/generate")
async def generate(request: GenerateRequest):
    query_id = str(uuid.uuid4())[:8]
    queries_db[query_id] = QueryState(
        query_id=query_id,
        prompt=request.prompt,
        max_tokens=request.max_tokens
    )
    await query_queue.put(query_id)
    return {
        "query_id": query_id,
        "status": "queued",
        "message": "Query queued for processing"
    }


@app.get("/status/{query_id}")
async def get_status(query_id: str):
    if query_id not in queries_db:
        return {"error": "Query not found"}
    q = queries_db[query_id]
    return {
        "query_id": query_id,
        "status": q.status,
        "prompt": q.prompt,
        "result": q.result,
        "error": q.error,
        "progress": f"{q.current_iteration}/{q.max_tokens}"
    }


@app.get("/all-queries")
async def get_all_queries():
    return {
        "total": len(queries_db),
        "queries": {
            qid: {
                "status": q.status,
                "prompt": q.prompt,
                "result": q.result,
                "error": q.error,
                "progress": f"{q.current_iteration}/{q.max_tokens}",
                "current_stage": q.current_stage,
                "worker1_pid": q.worker1_pid,
                "worker2_pid": q.worker2_pid,
            }
            for qid, q in queries_db.items()
        }
    }


@app.post("/clear")
async def clear_data():
    """Frontend calls this to reset history and counters"""
    global queries_db
    queries_db.clear()
    # Note: Clear the queue as well to stop pending tasks
    while not query_queue.empty():
        try:
            query_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    print("[MASTER] 🗑️ All queries and history cleared.")
    return {"status": "success", "message": "Backend memory cleared"}


@app.get("/health")
async def health():
    return {"status": "ok", "queued_queries": query_queue.qsize()}


# --- Helper Functions for Workers ---

async def call_worker1(query_id: str, query: QueryState):
    try:
        response = await asyncio.to_thread(
            lambda: requests.post(
                WORKER1_URL,
                json={"input_ids": query.tokens},
                timeout=30
            ).json()
        )
        if "error" in response:
            raise Exception(f"Worker1 error: {response['error']}")

        w1_pid = response.get("worker_pid", "unknown")
        return response["hidden_states"], w1_pid
    except Exception as e:
        print(f"[MASTER] [W1] ERROR for {query_id}: {str(e)}")
        raise


async def call_worker2(query_id: str, query: QueryState):
    try:
        response = await asyncio.to_thread(
            lambda: requests.post(
                WORKER2_URL,
                json={"hidden_states": query.hidden_states},
                timeout=30
            ).json()
        )
        if "error" in response:
            raise Exception(f"Worker2 error: {response['error']}")

        w2_pid = response.get("worker_pid", "unknown")
        return response["logits"], w2_pid
    except Exception as e:
        print(f"[MASTER] [W2] ERROR for {query_id}: {str(e)}")
        raise


# --- Core Scheduler ---

async def process_queries_pipelined():
    pending_worker1: Dict[str, QueryState] = {}
    pending_worker2: Dict[str, QueryState] = {}

    while True:
        try:
            # 1. New Queries pick karna
            while not query_queue.empty():
                query_id = query_queue.get_nowait()
                if query_id in queries_db:
                    query = queries_db[query_id]
                    query.status = "queued"
                    pending_worker1[query_id] = query
                    print(f"[MASTER] [Scheduler] {query_id[:4]} QUEUED")
        except asyncio.QueueEmpty:
            pass

        # 2. Worker 1 Stage
        worker1_tasks = []
        for qid in list(pending_worker1.keys()):
            q = pending_worker1[qid]
            if q.current_iteration < q.max_tokens:
                q.status = "worker1"
                task = asyncio.create_task(call_worker1(qid, q))
                worker1_tasks.append((qid, task))

        if worker1_tasks:
            for qid, task in worker1_tasks:
                try:
                    h_states, w1_pid = await task
                    if qid in queries_db:
                        q = queries_db[qid]
                        q.hidden_states = h_states
                        q.worker1_pid = w1_pid
                        q.status = "returned"
                        del pending_worker1[qid]
                        pending_worker2[qid] = q
                except Exception as e:
                    if qid in queries_db:
                        queries_db[qid].status = "error"
                        queries_db[qid].error = str(e)
                    if qid in pending_worker1: del pending_worker1[qid]

        # 3. Worker 2 Stage
        worker2_tasks = []
        for qid in list(pending_worker2.keys()):
            q = pending_worker2[qid]
            q.status = "worker2"
            task = asyncio.create_task(call_worker2(qid, q))
            worker2_tasks.append((qid, task))

        if worker2_tasks:
            for qid, task in worker2_tasks:
                try:
                    logits, w2_pid = await task
                    if qid in queries_db:
                        q = queries_db[qid]
                        q.worker2_pid = w2_pid

                        # Post-processing (Sampling Logic)
                        logits_tensor = torch.tensor(logits)[0, -1]

                        # Penalty for repetition (Simplified)
                        for t_id in set(q.tokens):
                            logits_tensor[t_id] /= 1.5

                        probs = F.softmax(logits_tensor / 0.9, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()

                        q.tokens.append(next_token)
                        q.current_iteration += 1

                        del pending_worker2[qid]
                        if q.current_iteration >= q.max_tokens:
                            q.result = tokenizer.decode(q.tokens)
                            q.status = "completed"
                            print(f"[MASTER] {qid[:4]} ✅ COMPLETED")
                        else:
                            pending_worker1[qid] = q  # Loop back to W1
                except Exception as e:
                    if qid in queries_db:
                        queries_db[qid].status = "error"
                        queries_db[qid].error = str(e)
                    if qid in pending_worker2: del pending_worker2[qid]

        await asyncio.sleep(0.05)  # CPU relief


@app.on_event("startup")
async def startup_event():
    print("[MASTER] Starting pipelined query scheduler...")
    asyncio.create_task(process_queries_pipelined())
