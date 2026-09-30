"""Live check of the Jev failure advisor on representative llama-server failures (needs Switchboard's key).

Not collected by pytest (no test_ prefix). Run: python -m tests.jev_eval
"""

from modeltuner import advisor

F = {"vram_total_mib": 12227, "vram_other_apps_mib": 0, "ctx": 131072, "gpu_layers": 999, "cache_type": "q8_0",
     "spec": "draft-mtp,ngram-mod", "ubatch": 512, "model_arch": "qwen35"}

CASES = [
    ("out_of_vram", dict(F, projected_device_mib=11900, available_ram_gb=4.2),
     "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 3120.00 MiB on device 0: cudaMalloc failed: out of memory\n"
     "E llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache"),
    ("system_ram_exhausted", dict(F, projected_device_mib=11600, projected_host_mib=9000, available_ram_gb=0.3),
     "W llama_model_load: mmap failed; falling back to read\n"
     "E ggml_aligned_malloc: insufficient memory (attempted to allocate 8912.00 MB)\n"
     "E llama_model_load: error loading model: unable to allocate CPU buffer"),
    ("unsupported_setting", dict(F, projected_device_mib=9000, available_ram_gb=5.0, spec="draft-mtp"),
     "E common_speculative_init: model 'OxCoder-9B.i1-Q6_K.gguf' has no MTP (nextn) layers; "
     "--spec-type draft-mtp is not supported\nE srv load_model: failed to initialize speculative decoding"),
    ("driver_or_runtime", dict(F, projected_device_mib=8500, available_ram_gb=6.0),
     "E CUDA error: an illegal memory access was encountered\n"
     "  current device: 0, in function ggml_backend_cuda_synchronize at ggml-cuda.cu:2533\nE ggml_cuda_error: CUDA error"),
    ("model_file_problem", dict(F, projected_device_mib=0, available_ram_gb=6.0),
     "E gguf_init_from_file_impl: failed to read tensor info\n"
     "E llama_model_load: error loading model: tensor data is not within file bounds (file truncated?)"),
    ("unknown", dict(F, projected_device_mib=8500, available_ram_gb=6.0),
     "I srv load_model: loading model\nI srv init: initializing slots, n_slots = 1"),
]


def main() -> None:
    ok = 0
    for expected, facts, log in CASES:
        d = advisor.diagnose_failure(log, facts)
        got = d.cause if d else None
        ok += got == expected
        print(f"{'OK  ' if got == expected else 'MISS'} expected {expected:22} got {got!s:22} "
              f"p={d.probability if d else 0:.2f} conf={d.confidence if d else 0:.2f} "
              f"actionable={d.actionable if d else None}")
    print(f"{ok}/{len(CASES)} correct")


if __name__ == "__main__":
    main()
