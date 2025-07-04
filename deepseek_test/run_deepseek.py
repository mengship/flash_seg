#!/user/bin/env python3
# -*- coding: utf-8 -*-
from fastapi import FastAPI
import requests

app = FastAPI()
OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/chat")
def chat(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={"model": "nezahatkorkmaz/deepseek-v3:latest", "prompt": prompt, "stream": False}
    )
    return response.json()

# 运行：uvicorn main:app --reload --port 8000