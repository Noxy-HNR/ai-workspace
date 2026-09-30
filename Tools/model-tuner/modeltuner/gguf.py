"""GGUF file reader for Model Tuner.

Reads a GGUF file's header: metadata key/values (all value types, including arrays)
and the tensor table, without reading tensor data.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


class GGUFError(Exception):
    """Raised for any GGUF format error."""
    pass


@dataclass(frozen=True)
class ArraySummary:
    """Stands in for arrays longer than max_array_len."""
    item_type: int
    length: int


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    nbytes: int


# Known GGML quantization types: (block_size, type_size_in_bytes)
GGML_TYPE_INFO: dict[int, tuple[int, int]] = {
    0: (1, 4),    # F32
    1: (1, 2),    # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
    34: (256, 54),   # TQ1_0
    35: (256, 66),   # TQ2_0
}


def expected_nbytes(shape: tuple[int, ...], ggml_type: int) -> int | None:
    """Calculate expected tensor size in bytes for known types.

    Returns None for unknown types.
    """
    info = GGML_TYPE_INFO.get(ggml_type)
    if info is None:
        return None
    block_size, type_size = info
    elements = 1
    for dim in shape:
        elements *= dim
    if elements % block_size != 0:
        return None
    return (elements // block_size) * type_size


class _GGUFReader:
    """Internal reader for GGUF files."""

    def __init__(self, f: BinaryIO, max_array_len: int):
        self.f = f
        self.max_array_len = max_array_len

    def _read(self, n: int) -> bytes:
        """Read exactly n bytes, raising GGUFError on EOF."""
        data = self.f.read(n)
        if len(data) < n:
            raise GGUFError(f"Truncated file: expected {n} bytes, got {len(data)}")
        return data

    def _read_u8(self) -> int:
        return struct.unpack('<B', self._read(1))[0]

    def _read_i8(self) -> int:
        return struct.unpack('<b', self._read(1))[0]

    def _read_u16(self) -> int:
        return struct.unpack('<H', self._read(2))[0]

    def _read_i16(self) -> int:
        return struct.unpack('<h', self._read(2))[0]

    def _read_u32(self) -> int:
        return struct.unpack('<I', self._read(4))[0]

    def _read_i32(self) -> int:
        return struct.unpack('<i', self._read(4))[0]

    def _read_f32(self) -> float:
        return struct.unpack('<f', self._read(4))[0]

    def _read_u64(self) -> int:
        return struct.unpack('<Q', self._read(8))[0]

    def _read_i64(self) -> int:
        return struct.unpack('<q', self._read(8))[0]

    def _read_f64(self) -> float:
        return struct.unpack('<d', self._read(8))[0]

    def _read_string(self) -> str:
        """Read a GGUF string (u64 length + UTF-8 bytes)."""
        length = self._read_u64()
        if length > 64 * 1024 * 1024:  # 64 MB
            raise GGUFError(f"String length {length} exceeds 64 MB limit")
        data = self._read(length)
        return data.decode('utf-8', errors='replace')

    def _read_value(self, value_type: int) -> object:
        """Read a single value of the given type."""
        if value_type == 0:   # u8
            return self._read_u8()
        elif value_type == 1:  # i8
            return self._read_i8()
        elif value_type == 2:  # u16
            return self._read_u16()
        elif value_type == 3:  # i16
            return self._read_i16()
        elif value_type == 4:  # u32
            return self._read_u32()
        elif value_type == 5:  # i32
            return self._read_i32()
        elif value_type == 6:  # f32
            return self._read_f32()
        elif value_type == 7:  # bool
            b = self._read_u8()
            if b not in (0, 1):
                raise GGUFError(f"Invalid bool value: {b}")
            return bool(b)
        elif value_type == 8:  # string
            return self._read_string()
        elif value_type == 9:  # array
            return self._read_array()
        elif value_type == 10:  # u64
            return self._read_u64()
        elif value_type == 11:  # i64
            return self._read_i64()
        elif value_type == 12:  # f64
            return self._read_f64()
        else:
            raise GGUFError(f"Unknown value type: {value_type}")

    def _read_array(self) -> list | ArraySummary:
        """Read a GGUF array."""
        item_type = self._read_u32()
        count = self._read_u64()

        if count > 2**32:
            raise GGUFError(f"Array count {count} exceeds 2^32 limit")

        if count > self.max_array_len:
            # Skip the array data efficiently
            self._skip_array(item_type, count)
            return ArraySummary(item_type=item_type, length=count)

        result = []
        for _ in range(count):
            result.append(self._read_value(item_type))
        return result

    def _skip_array(self, item_type: int, count: int) -> None:
        """Efficiently skip over an array without decoding."""
        # For simple types, we can skip in bulk
        type_sizes = {
            0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
            7: 1, 10: 8, 11: 8, 12: 8,
        }
        if item_type in type_sizes:
            skip_bytes = count * type_sizes[item_type]
            # Read in chunks to avoid memory issues
            while skip_bytes > 0:
                chunk = min(skip_bytes, 1024 * 1024)  # 1 MB chunks
                self.f.read(chunk)
                skip_bytes -= chunk
            return

        # For strings and arrays, we must parse to skip
        for _ in range(count):
            if item_type == 8:  # string
                length = self._read_u64()
                if length > 64 * 1024 * 1024:
                    raise GGUFError(f"String length {length} exceeds 64 MB limit")
                self.f.seek(length, 1)
            elif item_type == 9:  # nested array
                nested_item_type = self._read_u32()
                nested_count = self._read_u64()
                self._skip_array(nested_item_type, nested_count)
            else:
                raise GGUFError(f"Unknown array item type: {item_type}")


@dataclass
class GGUFFile:
    path: Path
    version: int
    alignment: int
    data_offset: int
    file_size: int
    metadata: dict[str, object]
    tensors: list[TensorInfo]

    def get(self, key: str, default=None):
        return self.metadata.get(key, default)

    @property
    def architecture(self) -> str | None:
        return self.metadata.get('general.architecture')

    def arch_get(self, suffix: str, default=None):
        arch = self.architecture
        if arch is None:
            return default
        return self.metadata.get(f"{arch}.{suffix}", default)

    @staticmethod
    def layer_of(tensor_name: str) -> int | None:
        """Extract layer index from tensor name like 'blk.12.ffn_up.weight'."""
        import re
        match = re.search(r'blk\.(\d+)\.', tensor_name)
        if match:
            return int(match.group(1))
        return None

    def layer_bytes(self) -> dict[int, int]:
        """Total nbytes per block index."""
        result: dict[int, int] = {}
        for tensor in self.tensors:
            layer = self.layer_of(tensor.name)
            if layer is not None:
                result[layer] = result.get(layer, 0) + tensor.nbytes
        return result

    def non_layer_bytes(self) -> dict[str, int]:
        """Bytes per tensor not in a block (token_embd, output, etc.)."""
        result: dict[str, int] = {}
        for tensor in self.tensors:
            if self.layer_of(tensor.name) is None:
                result[tensor.name] = tensor.nbytes
        return result

    def total_tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)


def _read_u32_le(f: BinaryIO) -> int:
    data = f.read(4)
    if len(data) < 4:
        raise GGUFError("Truncated file: expected 4 bytes for u32")
    return struct.unpack('<I', data)[0]


def _read_u64_le(f: BinaryIO) -> int:
    data = f.read(8)
    if len(data) < 8:
        raise GGUFError("Truncated file: expected 8 bytes for u64")
    return struct.unpack('<Q', data)[0]


def _read_u32_be(f: BinaryIO) -> int:
    data = f.read(4)
    if len(data) < 4:
        raise GGUFError("Truncated file: expected 4 bytes for u32 (be)")
    return struct.unpack('>I', data)[0]


def read_gguf(path: str | Path, *, max_array_len: int = 4096) -> GGUFFile:
    """Read a GGUF file's header and tensor table.

    Args:
        path: Path to the GGUF file.
        max_array_len: Maximum array length to fully decode. Longer arrays
            are returned as ArraySummary.

    Returns:
        GGUFFile with metadata and tensor information.

    Raises:
        GGUFError: For any format error.
    """
    path = Path(path)
    file_size = path.stat().st_size

    with open(path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if len(magic) < 4:
            raise GGUFError("Truncated file: expected 4 bytes for magic")
        if magic != b'GGUF':
            # Check for byte-swapped (big-endian) version
            if magic == b'FUGG':
                raise GGUFError("Big-endian GGUF file detected (byte-swapped magic)")
            raise GGUFError(f"Bad magic: {magic!r}, expected b'GGUF'")

        # Read version (little-endian)
        version = _read_u32_le(f)
        if version == 1:
            raise GGUFError("GGUF version 1 is not supported")
        if version > 3:
            # Check for byte-swapped version by reading as big-endian
            # Need to seek back and re-read
            f.seek(4, 0)  # Back to version position
            be_version = _read_u32_be(f)
            if 1 <= be_version <= 3:
                raise GGUFError("Big-endian GGUF file detected")
            raise GGUFError(f"Unsupported GGUF version: {version}")

        tensor_count = _read_u64_le(f)
        kv_count = _read_u64_le(f)

        if tensor_count > 2**32:
            raise GGUFError(f"Tensor count {tensor_count} exceeds 2^32")
        if kv_count > 2**32:
            raise GGUFError(f"KV count {kv_count} exceeds 2^32")

        reader = _GGUFReader(f, max_array_len)

        # Read metadata
        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = reader._read_string()
            value_type = reader._read_u32()
            value = reader._read_value(value_type)
            metadata[key] = value

        # Read tensor info
        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = reader._read_string()
            n_dims = reader._read_u32()
            shape = []
            for _ in range(n_dims):
                shape.append(reader._read_u64())
            ggml_type = reader._read_u32()
            offset = reader._read_u64()
            tensors.append(TensorInfo(
                name=name,
                shape=tuple(shape),
                ggml_type=ggml_type,
                offset=offset,
                nbytes=0,  # Will compute after reading all tensors
            ))

        # Compute data_offset: end of tensor info table, rounded up to alignment
        alignment = metadata.get('general.alignment', 32)
        if not isinstance(alignment, int) or alignment <= 0:
            alignment = 32
        current_offset = f.tell()
        data_offset = ((current_offset + alignment - 1) // alignment) * alignment

        # Compute nbytes for each tensor from offsets
        # Sort tensors by offset to compute sizes
        tensors_sorted = sorted(tensors, key=lambda t: t.offset)
        for i, tensor in enumerate(tensors_sorted):
            if i + 1 < len(tensors_sorted):
                nbytes = tensors_sorted[i + 1].offset - tensor.offset
            else:
                nbytes = file_size - data_offset - tensor.offset
            # Update the tensor object (TensorInfo is not frozen)
            tensor.nbytes = nbytes

        return GGUFFile(
            path=path,
            version=version,
            alignment=alignment,
            data_offset=data_offset,
            file_size=file_size,
            metadata=metadata,
            tensors=tensors,
        )