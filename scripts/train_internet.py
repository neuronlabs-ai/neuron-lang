#!/usr/bin/env python3
"""
train_internet.py — Connects AGI to the internet by fetching live data from Wikipedia and RSS feeds,
preprocessing it into character-level training tokens, and saving it as NEURON binary dataset formats.

Usage:
    python scripts/train_internet.py --num-samples 1000 --seq-len 64
"""

import os
import struct
import json
import argparse
import urllib.request
import random

VOCAB_SIZE = 128
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2

WIKI_PAGES = [
    "Artificial_intelligence",
    "Physics",
    "Medicine",
    "Engineering",
    "Mathematics",
    "Causal_inference",
    "Reinforcement_learning",
    "Epistemology",
    "Cognitive_science",
    "Control_theory"
]

def fetch_wiki_summary(title: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NeuronAGI/1.0 (contact@neuronlabs.ai)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("extract", "")
    except Exception as e:
        print(f"  Warning: Failed to fetch Wikipedia page '{title}': {e}")
        return ""

def fetch_rss_feed() -> list[str]:
    # Live RSS feed of world news to teach the agent about current events
    url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
    headlines = []
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NeuronAGI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read().decode("utf-8")
            # Simple XML parsing of <title> tags
            parts = xml_data.split("<title>")
            for part in parts[1:]:
                headline = part.split("</title>")[0].strip()
                if headline and not headline.endswith("Google News"):
                    headlines.append(headline)
    except Exception as e:
        print(f"  Warning: Failed to fetch live Google News RSS feed: {e}")
    return headlines

def generate_fallback_templates() -> list[str]:
    print("  Generating fallback high-quality templates...")
    templates = [
        "Physics is the natural science that studies matter, its fundamental constituents, its motion and behavior through space and time.",
        "Medicine is the science and practice of caring for a patient, managing the diagnosis, prognosis, prevention, and treatment of injury.",
        "Engineering is the use of scientific principles to design and build machines, structures, tunnels, roads, vehicles, and buildings.",
        "Mathematics includes the study of such topics as quantity, structure, space, and change. It is essential in many fields.",
        "Artificial intelligence is intelligence demonstrated by machines, as opposed to the natural intelligence of humans.",
        "Causal inference is the process of determining the independent, actual effect of a particular component in a larger system.",
        "Reinforcement learning is an area of machine learning concerned with how intelligent agents take actions in an environment to maximize cumulative reward.",
        "Control theory deals with the control of continuously operating dynamical systems in engineered processes and machines."
    ]
    return templates

def tokenize_char(text: str, max_len: int) -> list[int]:
    tokens = [BOS_TOKEN]
    for ch in text[:max_len - 2]:
        code = ord(ch)
        if code < VOCAB_SIZE:
            tokens.append(code)
        else:
            tokens.append(ord("?"))
    tokens.append(EOS_TOKEN)
    while len(tokens) < max_len:
        tokens.append(PAD_TOKEN)
    return tokens[:max_len]

def save_binary_tensor(data: list[list[int]], filepath: str):
    num_samples = len(data)
    seq_len = len(data[0])
    total_rows = num_samples * seq_len
    
    with open(filepath, "wb") as f:
        for seq in data:
            for token_id in seq:
                one_hot = [0.0] * VOCAB_SIZE
                if 0 <= token_id < VOCAB_SIZE:
                    one_hot[token_id] = 1.0
                else:
                    one_hot[ord("?")] = 1.0
                for val in one_hot:
                    f.write(struct.pack("<d", val))
    print(f"  Saved {filepath} ({total_rows} x {VOCAB_SIZE} one-hot tensor)")

def main():
    parser = argparse.ArgumentParser(description="Internet scraping preprocessor for NEURON")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--seq-len", type=int, default=64, help="Tokens per sample")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("===============================================")
    print("  NeuronAGI Live Internet Scraper & Preprocessor")
    print("===============================================")

    # 1. Fetch live internet content
    print("1. Scraping Wikipedia abstracts...")
    text_data = []
    for page in WIKI_PAGES:
        print(f"  Fetching: {page}...")
        summary = fetch_wiki_summary(page)
        if summary:
            text_data.append(summary)

    print("2. Fetching live Google News headlines...")
    headlines = fetch_rss_feed()
    text_data.extend(headlines)

    # If no internet connection or empty, use fallback templates
    if not text_data:
        print("  Warning: No internet text retrieved. Using high-quality training templates...")
        text_data = generate_fallback_templates()

    print(f"\nRetrieved {len(text_data)} source text segments from the internet.")

    # 2. Tokenize and segment into training sequences
    print(f"3. Segmenting into training sequences (seq_len={args.seq_len})...")
    tokenized = []
    for text in text_data:
        # Split paragraph into sentences or shorter parts to create more sequences
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
        for sentence in sentences:
            tokens = tokenize_char(sentence, args.seq_len)
            tokenized.append(tokens)

    # Replicate/multiply sequences to match target num-samples if needed
    if len(tokenized) < args.num_samples:
        original = list(tokenized)
        while len(tokenized) < args.num_samples:
            tokenized.extend(original)

    random.seed(42)
    random.shuffle(tokenized)
    tokenized = tokenized[:args.num_samples]

    # 3. Create inputs and targets
    train_inputs = [seq[:-1] for seq in tokenized]
    train_targets = [seq[1:] for seq in tokenized]

    # 4. Save
    print("4. Saving binary tensors for model training...")
    save_binary_tensor(train_inputs, os.path.join(args.output_dir, "train_inputs.bin"))
    save_binary_tensor(train_targets, os.path.join(args.output_dir, "train_targets.bin"))

    # 5. Save metadata
    metadata = {
        "vocab_size": VOCAB_SIZE,
        "seq_len": len(train_inputs[0]),
        "num_samples": len(train_inputs),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nData pre-processing complete! Live internet training files saved successfully.")

if __name__ == "__main__":
    main()
