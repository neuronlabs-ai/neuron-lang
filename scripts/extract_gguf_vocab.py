# ═══════════════════════════════════════════════════════════════════════
#  Real GGUF BPE Vocabulary Extractor for Local Qwen 3.5 2B
# ═══════════════════════════════════════════════════════════════════════

import struct
import json
import os

GGUF_BLOB = r"C:\Users\ADMIN\.ollama\models\blobs\sha256-b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297"
GGUF_MAGIC = 0x46554747

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')

def extract_real_vocabulary(filepath, output_json):
    print(f"Reading real GGUF BPE Vocabulary from: {filepath}")
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            print("Invalid GGUF file!")
            return

        version, tensor_count, metadata_kv_count = struct.unpack('<IQQ', f.read(20))
        print(f"Total Tensors: {tensor_count}")
        print(f"Metadata Key-Values: {metadata_kv_count}")

        vocab_tokens = []

        for _ in range(metadata_kv_count):
            key = read_string(f)
            val_type = struct.unpack('<I', f.read(4))[0]

            if key == "tokenizer.ggml.tokens":
                # Read string array
                arr_type = struct.unpack('<I', f.read(4))[0]
                count = struct.unpack('<Q', f.read(8))[0]
                print(f"Found tokenizer.ggml.tokens array with {count} BPE tokens!")

                for i in range(count):
                    t_str = read_string(f)
                    vocab_tokens.append(t_str)
                break
            else:
                # Skip value
                if val_type in [0, 1, 7]: f.seek(1, 1)
                elif val_type in [2, 3]: f.seek(2, 1)
                elif val_type in [4, 5, 6]: f.seek(4, 1)
                elif val_type in [10, 11, 12]: f.seek(8, 1)
                elif val_type == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.seek(slen, 1)
                elif val_type == 9:
                    arr_type = struct.unpack('<I', f.read(4))[0]
                    count = struct.unpack('<Q', f.read(8))[0]
                    elem_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
                    if arr_type in elem_sizes:
                        f.seek(count * elem_sizes[arr_type], 1)
                    elif arr_type == 8:
                        for _ in range(count):
                            slen = struct.unpack('<Q', f.read(8))[0]
                            f.seek(slen, 1)

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as out_f:
            json.dump(vocab_tokens, out_f, ensure_ascii=False, indent=2)

        print(f"Extracted {len(vocab_tokens)} real BPE tokens to {output_json}")
        print("Sample tokens [100:110]:", vocab_tokens[100:110])

if __name__ == "__main__":
    out_file = r"C:\Users\ADMIN\neuron-lang\data\qwen_vocab.json"
    extract_real_vocabulary(GGUF_BLOB, out_file)
