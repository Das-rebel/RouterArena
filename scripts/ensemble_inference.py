#!/usr/bin/env python3
"""Run multiple models on ALL queries for ensemble preparation."""
import json, os, hashlib, httpx, time, sys

# Models to run
MODELS = [
    {"name": "deepseek-chat", "api_key": os.environ["DEEPSEEK_API_KEY"],
     "base_url": "https://api.deepseek.com/chat/completions"},
    {"name": "meta/llama-3.3-70b-instruct", "api_key": os.environ["NVIDIA_API_KEY"],
     "base_url": "https://integrate.api.nvidia.com/v1/chat/completions"},
]

# Load predictions
with open("router_inference/predictions/a3m-router.json") as f:
    preds = json.load(f)

regular = [p for p in preds if not p.get('for_optimality', False)]
print(f"Running {len(MODELS)} models on {len(regular)} queries")
print(f"Total API calls: {len(MODELS) * len(regular)}")

# Output: ensemble predictions with multiple generated_results per query
output = []
for i, p in enumerate(regular):
    entry = {
        "global_index": p.get("global_index", p.get("global index", i)),
        "prompt": p.get("prompt", ""),
        "generated_results": [],
        "prediction": "ensemble"
    }
    
    for model in MODELS:
        try:
            resp = httpx.post(model["base_url"],
                headers={"Authorization": f"Bearer {model['api_key']}", "Content-Type": "application/json"},
                json={"model": model["name"], "messages": [{"role":"user","content": p["prompt"]}], 
                      "max_tokens": 1024, "temperature": 0},
                timeout=60).json()
            
            if "choices" in resp:
                entry["generated_results"].append({
                    "model": model["name"],
                    "generated_answer": resp["choices"][0]["message"]["content"],
                    "token_usage": resp.get("usage", {})
                })
        except Exception as e:
            print(f"Error {model['name']} on {i}: {e}")
        
        time.sleep(0.5)  # Rate limit spacing
    
    if (i+1) % 100 == 0:
        print(f"Progress: {i+1}/{len(regular)}")
        # Save checkpoint
        with open("router_inference/predictions/a3m-ensemble-checkpoint.json", "w") as f:
            json.dump(output, f)
    
    output.append(entry)

with open("router_inference/predictions/a3m-ensemble.json", "w") as f:
    json.dump(output, f)

print(f"\nSaved {len(output)} ensemble predictions")
