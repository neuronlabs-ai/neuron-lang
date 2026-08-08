# ═══════════════════════════════════════════════════════════════════════
#  GGUF Metadata & Header Parser for Ollama Model Tensors
# ═══════════════════════════════════════════════════════════════════════

import struct
import sys
import os

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')

def read_val(f, val_type):
    if val_type == GGUF_TYPE_UINT8: return struct.unpack('<B', f.read(1))[0]
    elif val_type == GGUF_TYPE_INT8: return struct.unpack('<b', f.read(1))[0]
    elif val_type == GGUF_TYPE_UINT16: return struct.unpack('<H', f.read(2))[0]
    elif val_type == GGUF_TYPE_INT16: return struct.unpack('<h', f.read(2))[0]
    elif val_type == GGUF_TYPE_UINT32: return struct.unpack('<I', f.read(4))[0]
    elif val_type == GGUF_TYPE_INT32: return struct.unpack('<i', f.read(4))[0]
    elif val_type == GGUF_TYPE_FLOAT32: return struct.unpack('<f', f.read(4))[0]
    elif val_type == GGUF_TYPE_BOOL: return struct.unpack('<?', f.read(1))[0]
    elif val_type == GGUF_TYPE_STRING: return read_string(f)
    elif val_type == GGUF_TYPE_UINT64: return struct.unpack('<Q', f.read(8))[0]
    elif val_type == GGUF_TYPE_INT64: return struct.unpack('<q', f.read(8))[0]
    elif val_type == GGUF_TYPE_FLOAT64: return struct.unpack('<d', f.read(8))[0]
    elif val_type == GGUF_TYPE_ARRAY:
        arr_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        # Fast skip for huge vocabulary/token arrays
        elem_sizes = {GGUF_TYPE_UINT8:1, GGUF_TYPE_INT8:1, GGUF_TYPE_UINT16:2, GGUF_TYPE_INT16:2,
                      GGUF_TYPE_UINT32:4, GGUF_TYPE_INT32:4, GGUF_TYPE_FLOAT32:4, GGUF_TYPE_BOOL:1,
                      GGUF_TYPE_UINT64:8, GGUF_TYPE_INT64:8, GGUF_TYPE_FLOAT64:8}
        if arr_type in elem_sizes:
            f.seek(count * elem_sizes[arr_type], 1)
            return f"[Array of {count} items]"
        elif arr_type == GGUF_TYPE_STRING:
            # Skip string array items fast
            for _ in range(count):
                slen = struct.unpack('<Q', f.read(8))[0]
                f.seek(slen, 1)
            return f"[String Array of {count} items]"
        return [read_val(f, arr_type) for _ in range(min(count, 5))]
    return None

def parse_gguf(filepath):
    print(f"Parsing GGUF Header: {filepath}")
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            print("Not a valid GGUF file!")
            return

        version, tensor_count, metadata_kv_count = struct.unpack('<IQQ', f.read(20))
        print(f"GGUF Version: {version}")
        print(f"Total Tensors: {tensor_count}")
        print(f"Metadata Key-Values: {metadata_kv_count}\n")

        metadata = {}
        for _ in range(metadata_kv_count):
            key = read_string(f)
            val_type = struct.unpack('<I', f.read(4))[0]
            val = read_val(f, val_type)
            metadata[key] = val

        print("--- KEY METADATA ---")
        for k in ['general.name', 'general.architecture', 'llama.context_length', 'llama.embedding_length', 'llama.block_count']:
            if k in metadata:
                print(f"  {k}: {metadata[k]}")

        print("\n--- SAMPLE TENSORS ---")
        for i in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            type_id = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            if i < 15 or i > tensor_count - 5:
                print(f"  Tensor [{i:3d}]: {name:<45} shape={dims} type={type_id}")
            elif i == 15:
                print("  ... [middle layers omitted] ...")

if __name__ == "__main__":
    blob_path = r"C:\Users\ADMIN\.ollama\models\blobs\sha256-b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297"
    if len(sys.argv) > 1:
        blob_path = sys.argv[1]
    parse_gguf(blob_path)
