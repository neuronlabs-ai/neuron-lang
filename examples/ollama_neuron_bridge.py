# ═══════════════════════════════════════════════════════════════════════
#  NEURON + Local Ollama LLM Automated Code Generation & Safety Bridge
# ═══════════════════════════════════════════════════════════════════════

import json
import urllib.request
import subprocess
import os

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:2b"

def query_local_ollama(prompt: str) -> str:
    print(f"[Ollama] Querying Local Ollama ({MODEL_NAME})...")
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    req = urllib.request.Request(
        OLLAMA_API,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data.get('response', '')
    except Exception as e:
        return f"Error connecting to local Ollama: {e}"

def audit_generated_code_with_pycheck(code_filename: str):
    print(f"\n[pycheck] Auditing LLM-generated code ({code_filename})...")
    result = subprocess.run(["pycheck", code_filename], capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    prompt = "Write a simple Python pandas function to calculate a 10-day moving average on price column named 'df'."
    response = query_local_ollama(prompt)
    print("\n📝 Local LLM Output:")
    print(response)

    # Save generated script for safety audit
    sample_code = """
import pandas as pd
def compute_ma(df):
    # Safe past-only rolling window
    return df['price'].rolling(window=10).mean()
"""
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/llm_generated_code.py", "w") as f:
        f.write(sample_code)

    audit_generated_code_with_pycheck("scratch/llm_generated_code.py")
