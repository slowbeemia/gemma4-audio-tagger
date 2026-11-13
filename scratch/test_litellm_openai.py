import httpx

print("Sending chat request to LiteLLM OpenAI endpoint...")
try:
    resp = httpx.post(
        "http://localhost:9379/v1/chat/completions",
        json={
            "model": "gemma4:12b",
            "messages": [
                {"role": "user", "content": "Say hello!"}
            ],
            "stream": False
        },
        headers={"x-litellm-disable-callbacks": "true"},
        timeout=60.0
    )
    print("LiteLLM Response status:", resp.status_code)
    print("LiteLLM Response:", resp.json())
except Exception as e:
    print("Error:", e)
