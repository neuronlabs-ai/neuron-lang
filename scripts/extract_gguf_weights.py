# ═══════════════════════════════════════════════════════════════════════
#  Full GGUF Tensor Dequantizer & Weight Extractor
#  Reads quantized tensors from Ollama GGUF blob → raw float .bin files
# ═══════════════════════════════════════════════════════════════════════

import struct
import os
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

GGUF_BLOB = r"C:\Users\ADMIN\.ollama\models\blobs\sha256-b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297"
OUTPUT_DIR = r"C:\Users\ADMIN\neuron-lang\data\qwen_layers"

# GGUF quantization type IDs
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9

# Bytes per block for each quantization type
BLOCK_SIZES = {
    GGML_TYPE_F32:  (1, 4),     # 1 element, 4 bytes
    GGML_TYPE_F16:  (1, 2),     # 1 element, 2 bytes
    GGML_TYPE_Q4_0: (32, 18),   # 32 elements, 18 bytes (2 scale + 16 data)
    GGML_TYPE_Q4_1: (32, 20),   # 32 elements, 20 bytes
    GGML_TYPE_Q5_0: (32, 22),   # 32 elements, 22 bytes
    GGML_TYPE_Q5_1: (32, 24),   # 32 elements, 24 bytes
    GGML_TYPE_Q8_0: (32, 34),   # 32 elements, 34 bytes (2 scale + 32 data)
    GGML_TYPE_Q8_1: (32, 36),   # 32 elements, 36 bytes
}


def dequantize_q8_0(raw_bytes, n_elements):
    """Dequantize Q8_0: each block = 32 int8 values + 1 f16 scale."""
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    offset = 0
    for b in range(n_blocks):
        # Read f16 scale (2 bytes)
        scale = np.frombuffer(raw_bytes[offset:offset+2], dtype=np.float16)[0]
        offset += 2
        # Read 32 int8 quantized values
        quants = np.frombuffer(raw_bytes[offset:offset+32], dtype=np.int8)
        offset += 32
        # Dequantize: float_val = scale * int8_val
        result[b*block_size:(b+1)*block_size] = scale * quants.astype(np.float32)

    return result


def dequantize_q4_0(raw_bytes, n_elements):
    """Dequantize Q4_0: each block = 32 values packed in 16 bytes + 2 byte f16 scale."""
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    offset = 0
    for b in range(n_blocks):
        # Read f16 scale
        scale = np.frombuffer(raw_bytes[offset:offset+2], dtype=np.float16)[0]
        offset += 2
        # Read 16 bytes = 32 nibbles (4-bit values)
        data = np.frombuffer(raw_bytes[offset:offset+16], dtype=np.uint8)
        offset += 16
        # Unpack nibbles
        low = (data & 0x0F).astype(np.float32) - 8.0
        high = ((data >> 4) & 0x0F).astype(np.float32) - 8.0
        quants = np.empty(32, dtype=np.float32)
        quants[0::2] = low
        quants[1::2] = high
        result[b*block_size:(b+1)*block_size] = scale * quants

    return result


def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')


def extract_all_tensors(filepath, output_dir, max_tensors=30):
    """Extract and dequantize tensor weights from GGUF file."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening GGUF: {filepath}")
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x46554747:
            print("Invalid GGUF!")
            return

        version, tensor_count, kv_count = struct.unpack('<IQQ', f.read(20))
        print(f"Version: {version}, Tensors: {tensor_count}, KVs: {kv_count}")

        # Skip metadata key-values
        for _ in range(kv_count):
            read_string(f)  # key
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

        # Compute data start (aligned to 32 bytes after header)
        header_end = f.tell()
        alignment = 32
        data_start = ((header_end + alignment - 1) // alignment) * alignment

        # Extract first N tensors (first transformer block)
        extracted = 0
        manifest = []
        for name, dims, type_id, offset in tensor_infos:
            if extracted >= max_tensors:
                break

            # Only extract block 0 tensors + embeddings + output
            if not (name.startswith('blk.0.') or name.startswith('token_embd') or name.startswith('output')):
                continue

            n_elements = 1
            for d in dims:
                n_elements *= d

            if type_id not in BLOCK_SIZES:
                print(f"  SKIP {name}: unsupported quant type {type_id}")
                continue

            block_elems, block_bytes = BLOCK_SIZES[type_id]
            n_blocks = n_elements // block_elems
            raw_size = n_blocks * block_bytes

            # Seek to tensor data
            f.seek(data_start + offset)
            raw_data = f.read(raw_size)

            # Dequantize
            if type_id == GGML_TYPE_F32:
                float_data = np.frombuffer(raw_data, dtype=np.float32)
            elif type_id == GGML_TYPE_F16:
                float_data = np.frombuffer(raw_data, dtype=np.float16).astype(np.float32)
            elif type_id == GGML_TYPE_Q8_0:
                float_data = dequantize_q8_0(raw_data, n_elements)
            elif type_id == GGML_TYPE_Q4_0:
                float_data = dequantize_q4_0(raw_data, n_elements)
            else:
                print(f"  SKIP {name}: dequant not implemented for type {type_id}")
                continue

            # Reshape and save
            float_data = float_data.reshape(dims)
            safe_name = name.replace('.', '_')
            out_path = os.path.join(output_dir, f"{safe_name}.bin")

            # Save as f64 for NEURON compatibility
            float_data.astype(np.float64).tofile(out_path)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            manifest.append((name, dims, type_id, safe_name, size_mb))
            print(f"  [OK] {name}: shape={dims} type={type_id} -> {size_mb:.2f} MB")
            extracted += 1

        # Save manifest
        manifest_path = os.path.join(output_dir, "manifest.txt")
        with open(manifest_path, 'w') as mf:
            for name, dims, type_id, safe_name, size_mb in manifest:
                dim_str = 'x'.join(str(d) for d in dims)
                mf.write(f"{safe_name}.bin | {name} | {dim_str} | type={type_id} | {size_mb:.2f}MB\n")

        print(f"\n[OK] Extracted {extracted} tensors to {output_dir}")
        print(f"[OK] Manifest saved to {manifest_path}")


if __name__ == "__main__":
    extract_all_tensors(GGUF_BLOB, OUTPUT_DIR, max_tensors=30)
