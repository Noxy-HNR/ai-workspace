"""Tests for modeltuner.gguf module."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from modeltuner.gguf import (
    GGUFError,
    ArraySummary,
    TensorInfo,
    GGUFFile,
    read_gguf,
    expected_nbytes,
)
from tests.gguf_writer import (
    write_gguf,
    make_test_gguf,
    KVSpec,
    TensorSpec,
)


class TestExpectedNbytes:
    """Tests for expected_nbytes function."""

    def test_known_types(self):
        assert expected_nbytes((32,), 0) == 128  # F32: 32 * 4
        assert expected_nbytes((32,), 1) == 64   # F16: 32 * 2
        assert expected_nbytes((32,), 2) == 18   # Q4_0: 32/32 * 18
        assert expected_nbytes((64,), 2) == 36   # Q4_0: 64/32 * 18
        assert expected_nbytes((256,), 10) == 84  # Q2_K: 256/256 * 84
        assert expected_nbytes((512,), 10) == 168 # Q2_K: 512/256 * 84

    def test_block_size_mismatch_returns_none(self):
        # Shape not divisible by block size
        assert expected_nbytes((33,), 2) is None  # Q4_0 block=32
        assert expected_nbytes((255,), 10) is None  # Q2_K block=256

    def test_unknown_type_returns_none(self):
        assert expected_nbytes((32,), 999) is None
        assert expected_nbytes((32,), 143) is None  # PrismML type


class TestSyntheticGGUF:
    """Tests using synthetic GGUF files."""

    def test_roundtrip_basic(self):
        """Test reading a basic synthetic GGUF file."""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            metadata, tensor_specs, alignment, _ = make_test_gguf(path)
            gguf = read_gguf(path)

            assert gguf.version == 3
            assert gguf.alignment == alignment
            assert gguf.file_size == Path(path).stat().st_size
            assert gguf.data_offset > 0

            # Check metadata
            assert gguf.get("general.architecture") == "test_arch"
            assert gguf.get("general.name") == "Test Model"
            assert gguf.get("general.version") == 2
            assert gguf.get("general.alignment") == alignment
            assert gguf.get("test.u8") == 255
            assert gguf.get("test.i8") == -128
            assert gguf.get("test.u16") == 65535
            assert gguf.get("test.i16") == -32768
            assert gguf.get("test.u32") == 4294967295
            assert gguf.get("test.i32") == -2147483648
            assert abs(gguf.get("test.f32") - 3.14159) < 1e-5
            assert gguf.get("test.bool_true") is True
            assert gguf.get("test.bool_false") is False
            assert gguf.get("test.string") == "hello world"
            assert gguf.get("test.u64") == 18446744073709551615
            assert gguf.get("test.i64") == -9223372036854775808
            assert abs(gguf.get("test.f64") - 2.718281828459045) < 1e-10

            # Check arrays
            assert gguf.get("test.array_u32") == [1, 2, 3, 4, 5]
            assert gguf.get("test.array_str") == ["a", "b", "c"]
            assert gguf.get("test.array_nested") == [[1, 2], [3, 4]]

            # Check architecture
            assert gguf.architecture == "test_arch"
            assert gguf.arch_get("block_count") == 12
            assert gguf.arch_get("attention.head_count") == 32
            assert gguf.arch_get("attention.sliding_window") == 4096

            # Check tensors
            assert len(gguf.tensors) == len(tensor_specs)
            for i, (tensor, spec) in enumerate(zip(gguf.tensors, tensor_specs)):
                assert tensor.name == spec.name
                assert tensor.shape == spec.shape
                assert tensor.ggml_type == spec.ggml_type
                assert tensor.offset == spec.offset
                assert tensor.nbytes > 0

            # Check layer_of
            assert GGUFFile.layer_of("blk.0.attn_q.weight") == 0
            assert GGUFFile.layer_of("blk.12.ffn_up.weight") == 12
            assert GGUFFile.layer_of("token_embd.weight") is None
            assert GGUFFile.layer_of("output.weight") is None
            assert GGUFFile.layer_of("blk.0") is None

            # Check layer_bytes
            layer_bytes = gguf.layer_bytes()
            assert 0 in layer_bytes
            assert 1 in layer_bytes
            assert len(layer_bytes) == 2

            # Check non_layer_bytes
            non_layer = gguf.non_layer_bytes()
            assert "token_embd.weight" in non_layer
            assert "output.weight" in non_layer
            assert "output_norm.weight" in non_layer
            assert all("blk." not in k for k in non_layer.keys())

            # Check total_tensor_bytes
            total = gguf.total_tensor_bytes()
            assert total == gguf.file_size - gguf.data_offset

        finally:
            os.unlink(path)

    def test_max_array_len_short_arrays(self):
        """Arrays shorter than max_array_len should be returned as lists."""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            metadata = [
                KVSpec("short_array", 9, [1, 2, 3]),
                KVSpec("short_str_array", 9, ["a", "b"]),
            ]
            write_gguf(path, metadata=metadata)
            gguf = read_gguf(path, max_array_len=10)
            assert gguf.get("short_array") == [1, 2, 3]
            assert gguf.get("short_str_array") == ["a", "b"]
        finally:
            os.unlink(path)

    def test_max_array_len_long_arrays(self):
        """Arrays longer than max_array_len should be returned as ArraySummary."""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            long_array = list(range(100))
            metadata = [KVSpec("long_array", 9, long_array)]
            write_gguf(path, metadata=metadata)
            gguf = read_gguf(path, max_array_len=10)
            result = gguf.get("long_array")
            assert isinstance(result, ArraySummary)
            assert result.item_type == 4  # u32
            assert result.length == 100
        finally:
            os.unlink(path)

    def test_max_array_len_nested_arrays(self):
        """Nested arrays should respect max_array_len for outer array."""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            # Outer array has 100 inner arrays
            nested = [[i] for i in range(100)]
            metadata = [KVSpec("nested", 9, nested)]
            write_gguf(path, metadata=metadata)
            gguf = read_gguf(path, max_array_len=10)
            result = gguf.get("nested")
            assert isinstance(result, ArraySummary)
            assert result.item_type == 9  # array
            assert result.length == 100
        finally:
            os.unlink(path)


class TestErrorCases:
    """Tests for error handling."""

    def test_bad_magic(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'XXXX')  # Wrong magic
            f.write(b'\x03\x00\x00\x00')  # version
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')  # tensor_count
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')  # kv_count

        try:
            with pytest.raises(GGUFError, match="Bad magic"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_big_endian_magic(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'FUGG')  # Byte-swapped magic
            f.write(b'\x03\x00\x00\x00')  # version
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')

        try:
            with pytest.raises(GGUFError, match="Big-endian"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_version_1(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'GGUF')
            f.write(b'\x01\x00\x00\x00')  # version 1
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')

        try:
            with pytest.raises(GGUFError, match="version 1"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_version_too_high(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'GGUF')
            f.write(b'\x04\x00\x00\x00')  # version 4
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')

        try:
            with pytest.raises(GGUFError, match="Unsupported GGUF version"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_big_endian_version(self):
        # Big-endian version detection: version field contains 3 in big-endian
        # (0x00000003), which reads as 50331648 in little-endian.
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'GGUF')
            # Version in big-endian: 3 = 0x00000003
            f.write(b'\x00\x00\x00\x03')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')

        try:
            with pytest.raises(GGUFError, match="Big-endian"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_truncated_header(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'GGUF')
            f.write(b'\x03\x00\x00\x00')  # version
            # Truncated - missing tensor_count and kv_count

        try:
            with pytest.raises(GGUFError, match="Truncated"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_truncated_string(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'GGUF')
            f.write(b'\x03\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')  # tensor_count = 0
            f.write(b'\x01\x00\x00\x00\x00\x00\x00\x00')  # kv_count = 1
            # Key: length 100 but no data
            f.write(b'\x64\x00\x00\x00\x00\x00\x00\x00')
            # No string data follows

        try:
            with pytest.raises(GGUFError, match="Truncated"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_string_too_long(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'GGUF')
            f.write(b'\x03\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x01\x00\x00\x00\x00\x00\x00\x00')
            # String length > 64 MB (64MB = 67108864)
            f.write(b'\x01\x00\x00\x04\x00\x00\x00\x00')  # 64MB + 1 = 67108865

        try:
            with pytest.raises(GGUFError, match="exceeds 64 MB"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_array_count_too_large(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'GGUF')
            f.write(b'\x03\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x01\x00\x00\x00\x00\x00\x00\x00')
            # Key "test"
            f.write(b'\x04\x00\x00\x00\x00\x00\x00\x00test')
            # Array type
            f.write(b'\x09\x00\x00\x00')
            # Item type u32, count > 2^32 (2^32 + 1 = 4294967297)
            f.write(b'\x04\x00\x00\x00')
            f.write(b'\x01\x00\x00\x00\x01\x00\x00\x00')  # 2^32 + 1

        try:
            with pytest.raises(GGUFError, match="exceeds 2\\^32"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_unknown_value_type(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'GGUF')
            f.write(b'\x03\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x01\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x04\x00\x00\x00\x00\x00\x00\x00test')
            f.write(b'\x63\x00\x00\x00')  # Unknown type 99

        try:
            with pytest.raises(GGUFError, match="Unknown value type"):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_invalid_bool_value(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name
            f.write(b'GGUF')
            f.write(b'\x03\x00\x00\x00')
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x01\x00\x00\x00\x00\x00\x00\x00')
            f.write(b'\x04\x00\x00\x00\x00\x00\x00\x00test')
            f.write(b'\x07\x00\x00\x00')  # bool type
            f.write(b'\x02')  # Invalid bool value (not 0 or 1)

        try:
            with pytest.raises(GGUFError, match="Invalid bool value"):
                read_gguf(path)
        finally:
            os.unlink(path)


class TestRealFiles:
    """Tests for real GGUF files (skipped if missing)."""

    @pytest.mark.skipif(
        not os.path.exists(r"C:\AI\models\Qwen3.5-9B-UD-Q6_K_XL-mtp.gguf"),
        reason="Model file not found"
    )
    def test_qwen35_model(self):
        path = r"C:\AI\models\Qwen3.5-9B-UD-Q6_K_XL-mtp.gguf"
        start = time.time()
        gguf = read_gguf(path)
        elapsed = time.time() - start
        assert elapsed < 3.0, f"Read took {elapsed:.2f}s, expected < 3s"

        assert gguf.architecture == "qwen35"
        assert gguf.arch_get("block_count") == 33
        assert len(gguf.tensors) == 442
        assert gguf.arch_get("full_attention_interval") == 4
        assert gguf.arch_get("nextn_predict_layers") == 1

        # Check layer 32 has nextn tensors
        layer_32_tensors = [t for t in gguf.tensors if GGUFFile.layer_of(t.name) == 32]
        nextn_tensors = [t for t in layer_32_tensors if "nextn" in t.name]
        assert len(nextn_tensors) > 0, "Layer 32 should have nextn tensors"

        # Check total_tensor_bytes
        assert gguf.total_tensor_bytes() == gguf.file_size - gguf.data_offset

        # Check nbytes vs expected_nbytes for known types
        for tensor in gguf.tensors:
            expected = expected_nbytes(tensor.shape, tensor.ggml_type)
            if expected is not None:
                diff = tensor.nbytes - expected
                assert 0 <= diff < gguf.alignment, \
                    f"Tensor {tensor.name}: nbytes={tensor.nbytes}, expected={expected}, diff={diff}, alignment={gguf.alignment}"

    @pytest.mark.skipif(
        not os.path.exists(r"C:\AI\models\medgemma-27b-it-Q4_K_M.gguf"),
        reason="Model file not found"
    )
    def test_medgemma_model(self):
        path = r"C:\AI\models\medgemma-27b-it-Q4_K_M.gguf"
        start = time.time()
        gguf = read_gguf(path)
        elapsed = time.time() - start
        assert elapsed < 3.0, f"Read took {elapsed:.2f}s, expected < 3s"

        assert gguf.architecture == "gemma3"
        assert gguf.arch_get("attention.sliding_window") == 1024

        # Count layers
        layers = set()
        for tensor in gguf.tensors:
            layer = GGUFFile.layer_of(tensor.name)
            if layer is not None:
                layers.add(layer)
        assert len(layers) == 62

    @pytest.mark.skipif(
        not os.path.exists(r"C:\AI\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
        reason="Model file not found"
    )
    def test_ternary_bonsai_model(self):
        path = r"C:\AI\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
        start = time.time()
        gguf = read_gguf(path)
        elapsed = time.time() - start
        assert elapsed < 3.0, f"Read took {elapsed:.2f}s, expected < 3s"

        # Should read without error even with type 143
        # Check that type 143 tensors exist and expected_nbytes returns None
        type_143_tensors = [t for t in gguf.tensors if t.ggml_type == 143]
        assert len(type_143_tensors) > 0, "Should have type 143 tensors"
        for t in type_143_tensors:
            assert expected_nbytes(t.shape, t.ggml_type) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])