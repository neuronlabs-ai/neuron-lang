#!/usr/bin/env python3
"""
preprocess_tinystories.py — Preprocesses TinyStories dataset for NEURON MicroGPT training.

Downloads a subset of the TinyStories dataset, tokenizes it at the character level,
and outputs binary f64 tensor files that NEURON's load_tensor() can read directly.

Usage:
    python preprocess_tinystories.py [--num-samples 1000] [--seq-len 64] [--output-dir ./data]
"""

import os
import struct
import argparse
import json
import random

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
VOCAB_SIZE = 128  # ASCII character-level tokenizer
PAD_TOKEN = 0
BOS_TOKEN = 1  # Start of sequence
EOS_TOKEN = 2  # End of sequence

def download_tinystories(output_dir: str, max_stories: int = 10000) -> list[str]:
    """
    Downloads TinyStories from HuggingFace datasets.
    Falls back to generating synthetic stories if download fails.
    """
    stories = []
    
    # Try to load from HuggingFace datasets
    try:
        from datasets import load_dataset
        print("Downloading TinyStories from HuggingFace...")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for i, example in enumerate(ds):
            if i >= max_stories:
                break
            stories.append(example["text"])
            if (i + 1) % 1000 == 0:
                print(f"  Downloaded {i + 1} stories...")
        print(f"Downloaded {len(stories)} stories.")
        return stories
    except Exception as e:
        print(f"HuggingFace download failed ({e}), trying direct download...")

    # Try direct download of a small subset
    try:
        import urllib.request
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
        print(f"Downloading from {url}...")
        
        # Download just the first few MB
        req = urllib.request.Request(url, headers={"Range": "bytes=0-5000000"})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
        
        # Split on story boundaries (double newline or "<|endoftext|>")
        parts = raw_text.split("<|endoftext|>")
        for part in parts:
            text = part.strip()
            if len(text) > 50:  # Skip very short fragments
                stories.append(text)
            if len(stories) >= max_stories:
                break
        print(f"Downloaded {len(stories)} stories from direct URL.")
        return stories
    except Exception as e:
        print(f"Direct download also failed ({e}). Using synthetic training data.")

    # Fallback: Generate synthetic short stories for training
    print("Generating synthetic training corpus...")
    templates = [
        "Once upon a time, there was a little {animal} who loved to {action}. Every day, the {animal} would go to the {place} and {action} until the sun went down. The {animal} was very happy.",
        "The {color} {animal} sat in the {place}. It looked at the sky and smiled. Then it started to {action}. All the other animals watched and cheered. It was a good day.",
        "A small {animal} found a {object} in the {place}. The {animal} picked it up and showed it to its friend. They decided to {action} together. They had so much fun.",
        "One morning, a {color} {animal} woke up early. It wanted to {action} before breakfast. The {animal} went to the {place} and found a surprise. There was a big {object} waiting.",
        "The {animal} and the {animal2} were best friends. They always played in the {place}. Today they decided to {action}. It was the best day ever.",
        "There was a {color} house on top of a hill. Inside lived a {animal} who liked to {action}. Every night, the {animal} would look at the stars and dream about the {place}.",
        "A brave {animal} went on an adventure. It walked through the {place} and found a {object}. The {animal} was so excited. It brought the {object} home to share.",
        "The teacher asked the class to {action}. A little {animal} raised its hand and said it knew the answer. Everyone clapped. The {animal} felt proud and happy.",
        "In a garden full of flowers, a {color} {animal} was playing. It jumped over the {object} and landed in the {place}. Then it started to laugh because it was having so much fun.",
        "A {animal} wanted to learn how to {action}. It practiced every day in the {place}. After many tries, the {animal} finally did it. All its friends were amazed.",
        "The little {animal} had a dream. It wanted to visit the {place} and find a {object}. So one day, it packed its bag and set off on a journey. Along the way, the {animal} met a {color} {animal2}.",
        "It was a sunny day in the {place}. A {color} {animal} was looking for food. It found a {object} under a tree. The {animal} was so happy it started to {action} and sing.",
        "Mom told the {animal} to {action}. The {animal} did not want to. But then it tried, and it was fun! The {animal} {action}ed all day long in the {place}.",
        "The {animal} was scared of the dark {place}. But its friend the {animal2} said, 'Don't worry, I will help you.' Together they walked through the {place} and found a {color} {object}.",
        "There was a race in the {place}. The {animal} and the {animal2} both wanted to win. They ran as fast as they could. The {animal} won and everyone cheered.",
    ]
    animals = ["cat", "dog", "bird", "rabbit", "bear", "fox", "mouse", "deer", "fish", "owl", "puppy", "kitten", "lamb", "pony", "bunny"]
    animals2 = ["duck", "frog", "turtle", "butterfly", "squirrel", "penguin", "panda", "lion", "elephant", "monkey", "ladybug", "dragonfly", "hedgehog", "otter", "dolphin"]
    colors = ["red", "blue", "green", "golden", "silver", "purple", "orange", "pink", "white", "brown", "yellow", "bright", "shiny", "sparkly", "tiny"]
    actions = ["sing", "dance", "run", "jump", "swim", "fly", "read", "paint", "cook", "build", "play", "draw", "climb", "laugh", "explore"]
    places = ["park", "forest", "garden", "school", "mountain", "river", "beach", "library", "meadow", "castle", "village", "farm", "hill", "pond", "playground"]
    objects = ["ball", "book", "flower", "stone", "star", "key", "hat", "box", "ring", "shell", "leaf", "feather", "acorn", "balloon", "cookie"]

    random.seed(42)
    for i in range(max_stories):
        template = random.choice(templates)
        story = template.format(
            animal=random.choice(animals),
            animal2=random.choice(animals2),
            color=random.choice(colors),
            action=random.choice(actions),
            place=random.choice(places),
            object=random.choice(objects),
        )
        stories.append(story)

    print(f"Generated {len(stories)} synthetic training stories.")
    return stories


def tokenize_char(text: str, max_len: int) -> list[int]:
    """Character-level tokenizer. Maps ASCII chars to indices 0-127."""
    tokens = [BOS_TOKEN]
    for ch in text[:max_len - 2]:  # Leave room for BOS and EOS
        code = ord(ch)
        if code < VOCAB_SIZE:
            tokens.append(code)
        else:
            tokens.append(ord('?'))  # Replace non-ASCII
    tokens.append(EOS_TOKEN)
    
    # Pad to max_len
    while len(tokens) < max_len:
        tokens.append(PAD_TOKEN)
    
    return tokens[:max_len]


def save_binary_tensor(data: list[list[int]], filepath: str):
    """
    Saves a 2D list of token IDs as a flat binary file of one-hot encoded f64 values.
    Shape will be [num_samples * seq_len, VOCAB_SIZE].
    NEURON's load_tensor() reads this format directly.
    """
    num_samples = len(data)
    seq_len = len(data[0])
    total_rows = num_samples * seq_len
    
    with open(filepath, "wb") as f:
        for seq in data:
            for token_id in seq:
                # Generate one-hot vector of size VOCAB_SIZE (128)
                one_hot = [0.0] * VOCAB_SIZE
                if 0 <= token_id < VOCAB_SIZE:
                    one_hot[token_id] = 1.0
                else:
                    one_hot[ord('?')] = 1.0
                
                # Write 128 floats
                for val in one_hot:
                    f.write(struct.pack("<d", val))
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved {filepath}: {total_rows} x {VOCAB_SIZE} one-hot tensor ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess TinyStories for NEURON MicroGPT")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of training sequences")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length (tokens per sample)")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory for binary files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Download / generate stories
    stories = download_tinystories(args.output_dir, max_stories=args.num_samples * 2)

    # 2. Tokenize
    print(f"Tokenizing {len(stories)} stories to character-level sequences (seq_len={args.seq_len})...")
    tokenized = []
    for story in stories:
        tokens = tokenize_char(story, args.seq_len)
        tokenized.append(tokens)
        if len(tokenized) >= args.num_samples:
            break

    # Shuffle
    random.seed(42)
    random.shuffle(tokenized)

    # 3. Split into train inputs (all tokens except last) and train targets (all tokens except first)
    train_inputs = []
    train_targets = []
    for seq in tokenized:
        train_inputs.append(seq[:-1])   # [0..seq_len-2] — input context
        train_targets.append(seq[1:])    # [1..seq_len-1] — next-token targets

    input_seq_len = len(train_inputs[0])

    # 4. Save as binary tensors
    save_binary_tensor(train_inputs, os.path.join(args.output_dir, "train_inputs.bin"))
    save_binary_tensor(train_targets, os.path.join(args.output_dir, "train_targets.bin"))

    # 5. Save metadata
    metadata = {
        "vocab_size": VOCAB_SIZE,
        "seq_len": input_seq_len,
        "num_samples": len(train_inputs),
        "pad_token": PAD_TOKEN,
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")
    print(f"\nDone! Training data ready in {args.output_dir}/")
    print(f"  Samples: {len(train_inputs)}")
    print(f"  Seq length: {input_seq_len}")
    print(f"  Vocab size: {VOCAB_SIZE}")


if __name__ == "__main__":
    main()
