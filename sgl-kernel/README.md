# sglang-kernel (prior sgl-kernel)

[Kernel Library](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) for LLM inference engines

<div align="center">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/sgl-project/sglang/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/sglang-kernel)](https://pypi.org/project/sglang-kernel)

</div>

`sglang-kernel` provides optimized compute primitives for LLM inference engines, enabling efficient inference for large language models and vision-language models through custom kernel operations. The source tree remains under the `sgl-kernel/` directory and the Python import path remains `sgl_kernel`.

## Installation
Requires torch == 2.11.0

```bash
# Latest version
pip3 install sglang-kernel --upgrade
```

## Building from Source
Requires
- CMake ≥3.31,
- Python ≥3.10
- scikit-build-core
- ninja(optional)

### Use Makefile to build from the sgl-kernel source tree

```bash
make build
```

### Limit build resource usage (CPU / parallelism)

By default, `make build` uses all available CPU cores. You can override build parallelism and NVCC compile threads:

```bash
# Limit parallel jobs (controls both make and cmake parallelism)
make build MAX_JOBS=2

# Additionally limit NVCC internal threads (reduces CPU and peak memory)
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

### Building for A100 without the optional FA3 extension

Append `-DSGL_KERNEL_A100_BUILD=ON` to your existing `CMAKE_ARGS` to skip
`flash_ops` (including Hopper FA3 instantiations). This preset keeps SM80
support and both `common_ops` variants: A100 loads the precise-math library
installed under `sgl_kernel/sm100`, despite that directory's name. It does not
remove all non-SM80 code from the common libraries or change their Python API.
The preset is explicit to preserve cross-compilation and GPU-less wheel builds.

FP8 Tensor Core GEMM/BMM and expert-specialization kernels are replaced with
throwing implementations in this preset. Their registered symbols remain
available so `common_ops` can load, but calling them reports that the A100 build
does not support the operation. Per-token and per-group FP8 quantization kernels
remain compiled; the preset does not disable FP8 KV storage. This does not add
an FP8 GEMM fallback: the inference backend must select an A100-compatible path.

```bash
make build MAX_JOBS=4 CMAKE_ARGS="-DSGL_KERNEL_A100_BUILD=ON -DSGL_KERNEL_COMPILE_THREADS=1"
```

Keep any existing CUDA, NUMA and local-dependency arguments in `CMAKE_ARGS`.
Use a compatible LLM backend, for example `--attention-backend flashinfer`,
and `--mm-attention-backend fa2` for supported vision encoders. The separate
`flash_attn` package used by FA2 and BeeBee audio encoders is still required.
Importing `sgl_kernel.flash_attn` works without `flash_ops`, and
`is_fa3_supported()` returns false; calling an FA3 function gives an explicit
error instead of silently changing attention implementations.

Without the A100 preset, `SGL_KERNEL_ENABLE_FA3` defaults to ON for CUDA 12.4+
in a fresh build directory. An explicit `-DSGL_KERNEL_ENABLE_FA3=OFF` is now
respected, including an OFF value retained in an existing CMake cache. To enable
FA3 again, pass both `-DSGL_KERNEL_A100_BUILD=OFF` and
`-DSGL_KERNEL_ENABLE_FA3=ON`.

## Contribution

### Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/common_extension.cc)
4. Update [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/CMakeLists.txt) to include new CUDA source
5. Expose Python interface in [python](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel)
6. Add test and benchmark

### Development Tips

1. When creating torch extensions, add the function definition with `m.def`, and device binding with `m.impl`:

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // We need def with schema here for torch.compile
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
    "int cublas_handle) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
   ```

### Adapting C++ Native Types for Torch Compatibility

Third-party C++ libraries often use int and float, but PyTorch bindings require int64_t and double due to Python's type mapping.

Use make_pytorch_shim from sgl_kernel_torch_shim.h to handle conversions automatically:

```cpp

// Add type conversion for int -> int64_t
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "value too large");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "value too small");
    return arg;
  }
};
```
```cpp
// Wrap your function
m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### Testing & Benchmarking

1. Add pytest tests in [tests/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/tests), if you need to skip some test, please use `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
```

2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/benchmark)

   **We recommend using `triton.testing.do_bench_cudagraph` for kernel benchmarking**:

   Compared to `triton.testing.do_bench`, `do_bench_cudagraph` provides:
   - Reduced CPU overhead impact for more accurate kernel performance measurements
   - Incorporation of PDL (Programmatic Dependent Launch) effects into individual kernel results
   - More realistic performance data on PDL-supported architectures (SM >= 90)

3. Run test suite

## Kernel Size Analysis

Analyze CUDA kernel sizes in compiled wheel files to identify oversized kernels and template-instantiation bloat:

This tool requires `cubloaty` (install with `pip install cubloaty`) to work.

```bash
# Install cubloaty
pip install cubloaty

# Analyze a wheel file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl

# Custom output file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl --output my_analysis.txt
```

The tool generates:
- A text report with:
  - Kernel groups (by name prefix)
  - Individual kernel sizes (sorted by size)

Use this to identify large kernels and potential template instantiation bloat.

## FAQ
- Q: Segmentation fault with CUDA 12.6
- A: Update ptxas to 12.8, reference: [segment fault error](https://github.com/Dao-AILab/flash-attention/issues/1453)
