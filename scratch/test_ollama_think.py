import httpx

print("Sending chat request directly to Ollama with think: False...")
try:
    resp = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "gemma4:12b",
            "messages": [
                {"role": "user", "content": "Hello! Please tell me if you can hear me. Answer in 3 words."}
            ],
            "options": {
                "think": False
            },
            "stream": False
        },
        timeout=30.0
    )
    print("Ollama Response status:", resp.status_code)
    data = resp.json()
    print("Message content:", data.get("message", {}).get("content"))
    # Print the full response to see if there is any thinking field or how it's represented
    print("Full response keys:", data.keys())
    if "message" in data:
        print("Message keys:", data["message"].keys())
        # check if reasoning/thinking is in the message
        for k, v in data["message"].items():
            if k != "content":
                print(f"  {k}: {repr(v)[:200]}")
except Exception as e:
    print("Error:", e)
