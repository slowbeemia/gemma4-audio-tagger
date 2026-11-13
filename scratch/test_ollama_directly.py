import httpx

print("Sending chat request directly to Ollama...")
try:
    resp = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "gemma4:12b",
            "messages": [
                {"role": "user", "content": "Say hello!"}
            ],
            "stream": False
        },
        timeout=60.0
    )
    print("Ollama Response status:", resp.status_code)
    print("Ollama Response:", resp.json())
except Exception as e:
    print("Error:", e)
