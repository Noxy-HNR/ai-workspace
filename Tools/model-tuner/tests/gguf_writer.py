"""Helper to write synthetic GGUF files for testing."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int = 0  # Will be assigned by writer


@dataclass
class KVSpec:
    key: str
    value_type: int
    value: object


def _write_u8(f: BinaryIO, v: int) -> None:
    f.write(struct.pack('<B', v))


def _write_i8(f: BinaryIO, v: int) -> None:
    f.write(struct.pack('<b', v))


def _write_u16(f: BinaryIO, v: int) -> None:
    f.write(struct.pack('<H', v))


def _write_i16(f: BinaryIO, v: int) -> None:
    f.write(struct.pack('<h', v))


def _write_u32(f: BinaryIO, v: int) -> None:
    f.write(struct.pack('<I', v))


def _write_i32(f: BinaryIO, v: int) -> None:
    f.write(struct.pack('<i', v))


def _write_f32(f: BinaryIO, v: float) -> None:
    f.write(struct.pack('<f', v))


def _write_u64(f: BinaryIO, v: int) -> None:
    f.write(struct.pack('<Q', v))


def _write_i64(f: BinaryIO, v: int) -> None:
    f.write(struct.pack('<q', v))


def _write_f64(f: BinaryIO, v: float) -> None:
    f.write(struct.pack('<d', v))


def _write_string(f: BinaryIO, s: str) -> None:
    data = s.encode('utf-8')
    _write_u64(f, len(data))
    f.write(data)


def _write_value(f: BinaryIO, value_type: int, value: object) -> None:
    """Write a single value of the given type."""
    if value_type == 0:  # u8
        _write_u8(f, value)
    elif value_type == 1:  # i8
        _write_i8(f, value)
    elif value_type == 2:  # u16
        _write_u16(f, value)
    elif value_type == 3:  # i16
        _write_i16(f, value)
    elif value_type == 4:  # u32
        _write_u32(f, value)
    elif value_type == 5:  # i32
        _write_i32(f, value)
    elif value_type == 6:  # f32
        _write_f32(f, value)
    elif value_type == 7:  # bool
        _write_u8(f, 1 if value else 0)
    elif value_type == 8:  # string
        _write_string(f, value)
    elif value_type == 9:  # array
        _write_array(f, value)
    elif value_type == 10:  # u64
        _write_u64(f, value)
    elif value_type == 11:  # i64
        _write_i64(f, value)
    elif value_type == 12:  # f64
        _write_f64(f, value)
    else:
        raise ValueError(f"Unknown value type: {value_type}")


def _write_array(f: BinaryIO, value: list) -> None:
    """Write a GGUF array. Assumes homogeneous type."""
    if not value:
        # Empty array - default to u8 item type
        _write_u32(f, 0)
        _write_u64(f, 0)
        return

    # Determine item type from first element
    first = value[0]
    type_map = {
        int: 4,  # default to u32 for ints
        float: 6,  # f32
        bool: 7,
        str: 8,
        list: 9,
    }
    item_type = type_map.get(type(first), 4)

    _write_u32(f, item_type)
    _write_u64(f, len(value))

    for item in value:
        _write_value(f, item_type, item)


def write_gguf(
    path: str | Path,
    version: int = 3,
    alignment: int = 32,
    metadata: list[KVSpec] | None = None,
    tensors: list[TensorSpec] | None = None,
    tensor_data: bytes = b'',
) -> None:
    """Write a synthetic GGUF file.

    Args:
        path: Output file path.
        version: GGUF version (2 or 3).
        alignment: Tensor data alignment.
        metadata: List of KVSpec for metadata.
        tensors: List of TensorSpec for tensors.
        tensor_data: Raw tensor data to append (for size calculations).
    """
    path = Path(path)
    metadata = metadata or []
    tensors = tensors or []

    with open(path, 'wb') as f:
        # Header
        f.write(b'GGUF')
        _write_u32(f, version)
        _write_u64(f, len(tensors))
        _write_u64(f, len(metadata))

        # Metadata
        for kv in metadata:
            _write_string(f, kv.key)
            _write_u32(f, kv.value_type)
            _write_value(f, kv.value_type, kv.value)

        # Tensor info
        for t in tensors:
            _write_string(f, t.name)
            _write_u32(f, len(t.shape))
            for dim in t.shape:
                _write_u64(f, dim)
            _write_u32(f, t.ggml_type)
            _write_u64(f, t.offset)

        # Align to tensor data
        current = f.tell()
        data_offset = ((current + alignment - 1) // alignment) * alignment
        f.write(b'\x00' * (data_offset - current))

        # Tensor data
        f.write(tensor_data)


def make_test_gguf(path: str | Path) -> tuple[list[KVSpec], list[TensorSpec], int, int]:
    """Create a comprehensive test GGUF file and return specs for verification.

    Returns:
        (metadata, tensors, data_offset, alignment)
    """
    alignment = 64

    # Build metadata with all value types
    metadata = [
        KVSpec("general.architecture", 8, "test_arch"),
        KVSpec("general.name", 8, "Test Model"),
        KVSpec("general.version", 4, 2),
        KVSpec("general.alignment", 4, alignment),
        KVSpec("general.file_type", 4, 15),
        KVSpec("test.u8", 0, 255),
        KVSpec("test.i8", 1, -128),
        KVSpec("test.u16", 2, 65535),
        KVSpec("test.i16", 3, -32768),
        KVSpec("test.u32", 4, 4294967295),
        KVSpec("test.i32", 5, -2147483648),
        KVSpec("test.f32", 6, 3.14159),
        KVSpec("test.bool_true", 7, True),
        KVSpec("test.bool_false", 7, False),
        KVSpec("test.string", 8, "hello world"),
        KVSpec("test.u64", 10, 18446744073709551615),
        KVSpec("test.i64", 11, -9223372036854775808),
        KVSpec("test.f64", 12, 2.718281828459045),
        # Array of u32
        KVSpec("test.array_u32", 9, [1, 2, 3, 4, 5]),
        # Array of strings
        KVSpec("test.array_str", 9, ["a", "b", "c"]),
        # Nested array (array of arrays of u32)
        KVSpec("test.array_nested", 9, [[1, 2], [3, 4]]),
        # Architecture-specific
        KVSpec("test_arch.block_count", 4, 12),
        KVSpec("test_arch.attention.head_count", 4, 32),
        KVSpec("test_arch.attention.head_count_kv", 4, 8),
        KVSpec("test_arch.attention.sliding_window", 4, 4096),
    ]

    # Tensors with various types and shapes
    # Offsets will be calculated by the writer
    tensor_specs = [
        TensorSpec("token_embd.weight", (32000, 4096), 0),  # F32
        TensorSpec("blk.0.attn_q.weight", (4096, 4096), 2),  # Q4_0
        TensorSpec("blk.0.attn_k.weight", (4096, 1024), 2),  # Q4_0
        TensorSpec("blk.0.attn_v.weight", (4096, 1024), 2),  # Q4_0
        TensorSpec("blk.0.attn_out.weight", (4096, 4096), 2),  # Q4_0
        TensorSpec("blk.0.ffn_up.weight", (11008, 4096), 2),  # Q4_0
        TensorSpec("blk.0.ffn_down.weight", (4096, 11008), 2),  # Q4_0
        TensorSpec("blk.1.attn_q.weight", (4096, 4096), 3),  # Q4_1
        TensorSpec("blk.1.attn_k.weight", (4096, 1024), 3),  # Q4_1
        TensorSpec("blk.1.attn_v.weight", (4096, 1024), 3),  # Q4_1
        TensorSpec("blk.1.attn_out.weight", (4096, 4096), 3),  # Q4_1
        TensorSpec("blk.1.ffn_up.weight", (11008, 4096), 3),  # Q4_1
        TensorSpec("blk.1.ffn_down.weight", (4096, 11008), 3),  # Q4_1
        TensorSpec("output.weight", (32000, 4096), 1),  # F16
        TensorSpec("output_norm.weight", (4096,), 0),  # F32
    ]

    # Calculate offsets based on expected_nbytes
    from modeltuner.gguf import expected_nbytes

    offset = 0
    for t in tensor_specs:
        t.offset = offset
        nbytes = expected_nbytes(t.shape, t.ggml_type)
        if nbytes is None:
            nbytes = 0
        offset += nbytes

    # Add padding to make last tensor have some size
    tensor_data = b'\x00' * (offset + 128)

    write_gguf(path, version=3, alignment=alignment, metadata=metadata, tensors=tensor_specs, tensor_data=tensor_data)

    # Return the data_offset for verification
    # Need to compute it
    header_size = 4 + 4 + 8 + 8  # magic + version + tensor_count + kv_count
    for kv in metadata:
        header_size += 8 + len(kv.key.encode('utf-8'))  # string length + key
        header_size += 4  # value_type
        # value size varies - we'll compute data_offset from actual file
    # Just return alignment and let tests read the actual file
    return metadata, tensor_specs, alignment, 0


if __name__ == "__main__":
    # Quick test
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        make_test_gguf(f.name)
        print(f"Wrote test file to {f.name}")