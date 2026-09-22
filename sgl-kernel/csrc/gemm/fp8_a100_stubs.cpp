// Keep the common_ops ABI complete without compiling kernels that require
// newer GPUs. Quantization and KV storage are separate and remain available.
#include "sgl_kernel_ops.h"

torch::Tensor fp8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "fp8_scaled_mm is unavailable in SGL_KERNEL_A100_BUILD. "
      "Use an A100-compatible backend, or rebuild without this preset for a supported GPU.");
}

torch::Tensor fp8_blockwise_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "fp8_blockwise_scaled_mm is unavailable in SGL_KERNEL_A100_BUILD. "
      "Use an A100-compatible backend, or rebuild without this preset for a supported GPU.");
}

void bmm_fp8(
    at::Tensor A,
    at::Tensor B,
    at::Tensor D,
    at::Tensor A_scale,
    at::Tensor B_scale,
    at::Tensor workspace_buffer,
    int64_t cublas_handle) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "bmm_fp8 is unavailable in SGL_KERNEL_A100_BUILD. "
      "Use an A100-compatible backend, or rebuild without this preset for a supported GPU.");
}

void fp8_blockwise_scaled_grouped_mm(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "fp8_blockwise_scaled_grouped_mm is unavailable in SGL_KERNEL_A100_BUILD. "
      "Use an A100-compatible backend, or rebuild without this preset for a supported GPU.");
}

void es_fp8_blockwise_scaled_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_d,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "es_fp8_blockwise_scaled_grouped_mm is unavailable in SGL_KERNEL_A100_BUILD. "
      "Use an A100-compatible backend, or rebuild without this preset for a supported GPU.");
}

void es_sm100_mxfp8_blockscaled_grouped_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb,
    torch::Tensor& d,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "es_sm100_mxfp8_blockscaled_grouped_mm is unavailable in SGL_KERNEL_A100_BUILD. "
      "Use an A100-compatible backend, or rebuild without this preset for a supported GPU.");
}

void es_sm100_mxfp8_blockscaled_grouped_quant(
    const torch::Tensor& input,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets,
    torch::Tensor& quant_output,
    torch::Tensor& scale_factor) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "es_sm100_mxfp8_blockscaled_grouped_quant is unavailable in SGL_KERNEL_A100_BUILD. "
      "Use an A100-compatible backend, or rebuild without this preset for a supported GPU.");
}

