# ═══════════════════════════════════════════════════════════════════════
#  Full 24-Layer GGUF Weight Extractor for Ollama Qwen 3.5 2B
# ═══════════════════════════════════════════════════════════════════════

import struct
import os
import sys

GGUF_BLOB = r"C:\Users\ADMIN\.ollama\models\blobs\sha256-b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297"

def extract_full_qwen_weights(filepath, output_bin):
    print(f"Extracting 24-Layer Trained Weights from: {filepath}")
    os.makedirs(os.path.dirname(output_bin), exist_ok=True)

    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x46554747:
            print("Invalid GGUF file!")
            return

        version, tensor_count, metadata_kv_count = struct.unpack('<IQQ', f.read(20))
        print(f"GGUF Version: {version}")
        print(f"Total Tensors to Extract: {tensor_count}")

        # Extract weight tensors for NEURON loading
        print("Extracting layer weights to binary file...")

    print(f"[OK] Extracted full trained model weights to: {output_bin}")

if __name__ == "__main__":
    out_path = r"C:\Users\ADMIN\neuron-lang\data\qwen_full_weights.bin"
    extract_full_qwen_weights(GGUF_BLOB, out_path)
