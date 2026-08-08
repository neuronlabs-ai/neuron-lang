# ═══════════════════════════════════════════════════════════════════════
#  Extract ALL 24 Transformer Blocks from Qwen 3.5 2B GGUF
#  Only extracts the core 6 weight matrices per block (skips SSM/vision)
# ═══════════════════════════════════════════════════════════════════════

import struct
import os
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

GGUF_BLOB = r"C:\Users\ADMIN\.ollama\models\blobs\sha256-b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297"
OUTPUT_DIR = r"C:\Users\ADMIN\neuron-lang\data\qwen_layers"

# Keep only the essential transformer weights per block
KEEP_SUFFIXES = [
    'attn_norm.weight',
    'attn_qkv.weight',
    'attn_gate.weight',
    'post_attention_norm.weight',
    'ffn_gate.weight',
    'ffn_up.weight',
    'ffn_down.weight',
]

GLOBAL_TENSORS = ['output_norm.weight', 'token_embd.weight', 'output.weight']


def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')


def dequantize_q8_0(raw_bytes, n_elements):
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)
    offset = 0
    for b in range(n_blocks):
        scale = np.frombuffer(raw_bytes[offset:offset+2], dtype=np.float16)[0]
        offset += 2
        quants = np.frombuffer(raw_bytes[offset:offset+32], dtype=np.int8)
        offset += 32
        result[b*block_size:(b+1)*block_size] = scale * quants.astype(np.float32)
    return result


def should_extract(name):
    """Check if this tensor is one we need for inference."""
    for suffix in KEEP_SUFFIXES:
        if name.endswith(suffix):
            return True
    for gname in GLOBAL_TENSORS:
        if name == gname:
            return True
    return False


def extract_all_blocks(filepath, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Opening GGUF: {filepath}")

    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version, tensor_count, kv_count = struct.unpack('<IQQ', f.read(20))
        print(f"Tensors: {tensor_count}, KVs: {kv_count}")

        # Skip metadata
        for _ in range(kv_count):
            read_string(f)
            val_type = struct.unpack('<I', f.read(4))[0]
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
                elem_sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                if arr_type in elem_sizes:
                    f.seek(count * elem_sizes[arr_type], 1)
                elif arr_type == 8:
                    for _ in range(count):
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.seek(slen, 1)

        # Read tensor metadata
        tensor_infos = []
        for i in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            type_id = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos.append((name, dims, type_id, offset))

        header_end = f.tell()
        data_start = ((header_end + 31) // 32) * 32

        # Extract needed tensors
        extracted = 0
        skipped = 0
        total_mb = 0.0

        for name, dims, type_id, offset in tensor_infos:
            if not should_extract(name):
                skipped += 1
                continue

            # Skip the massive token_embd (already extracted)
            if name == 'token_embd.weight':
                safe = name.replace('.', '_')
                out_path = os.path.join(output_dir, f"{safe}.bin")
                if os.path.exists(out_path):
                    print(f"  [SKIP] {name}: already extracted")
                    continue

            n_elements = 1
            for d in dims:
                n_elements *= d

            # Compute raw data size
            if type_id == 0:  # F32
                raw_size = n_elements * 4
            elif type_id == 1:  # F16
                raw_size = n_elements * 2
            elif type_id == 8:  # Q8_0
                raw_size = (n_elements // 32) * 34
            else:
                print(f"  [SKIP] {name}: unsupported type {type_id}")
                continue

            f.seek(data_start + offset)
            raw_data = f.read(raw_size)

            # Dequantize
            if type_id == 0:
                float_data = np.frombuffer(raw_data, dtype=np.float32)
            elif type_id == 1:
                float_data = np.frombuffer(raw_data, dtype=np.float16).astype(np.float32)
            elif type_id == 8:
                float_data = dequantize_q8_0(raw_data, n_elements)

            float_data = float_data.reshape(dims)
            safe_name = name.replace('.', '_')
            out_path = os.path.join(output_dir, f"{safe_name}.bin")

            # Save as float32 (half the size of f64)
            float_data.astype(np.float32).tofile(out_path)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            total_mb += size_mb
            extracted += 1

            block_num = ""
            if name.startswith("blk."):
                block_num = name.split('.')[1]
            print(f"  [OK] {name}: {dims} -> {size_mb:.1f} MB")

        print(f"\n[DONE] Extracted {extracted} tensors, skipped {skipped}")
        print(f"[DONE] Total disk: {total_mb:.1f} MB in {output_dir}")


if __name__ == "__main__":
    extract_all_blocks(GGUF_BLOB, OUTPUT_DIR)
