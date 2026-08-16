"""对比测试: torch_fused 路径 vs numexpr 路径 vs 原始 HuggingFace 路径。

用法:
    conda run -n llava python python/sglang/srt/multimodal/test_precision_alignment.py
"""
import os
import time
import numpy as np

os.environ["SGLANG_IMG_PREPROCESS_BACKEND"] = "torch"


def test_all_paths():
    from sglang.srt.multimodal.image_processor_opt import Qwen25VLImageProcessorOptimized

    # 使用 beebeeomni 的典型参数
    processor = Qwen25VLImageProcessorOptimized(
        do_resize=False,
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1.0 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
    )

    # 模拟实际输入: 100 张 364×644×3 uint8 numpy 图片
    np.random.seed(42)
    H, W = 364, 644
    N = 100
    images_np = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(N)]

    # 构造 PIL images 用于原始 HF 路径
    import PIL.Image
    images_pil = [PIL.Image.fromarray(img) for img in images_np]

    NUM_RUNS = 5

    print("=" * 70)
    print(f"对比测试: 100 张 {H}×{W} 图片, 取 {NUM_RUNS} 次平均")
    print("=" * 70)

    # ── 路径 1: 原始 HuggingFace Qwen2VLImageProcessor (无任何优化) ──
    print("\n--- 路径 1: 原始 HuggingFace Qwen2VLImageProcessor ---")
    try:
        from transformers import Qwen2VLImageProcessor
        hf_processor = Qwen2VLImageProcessor(
            do_resize=False,
            do_center_crop=False,
            do_rescale=True,
            rescale_factor=1.0 / 255,
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            min_pixels=100 * 28 * 28,
            max_pixels=28 * 28 * 1280,
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
        )
        # warmup
        _ = hf_processor.preprocess(images_pil[:4], return_tensors="pt")

        times_hf = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            result_hf = hf_processor.preprocess(images_pil, return_tensors="pt")
            t1 = time.perf_counter()
            times_hf.append((t1 - t0) * 1000)

        avg_hf = sum(times_hf) / len(times_hf)
        print(f"  平均耗时: {avg_hf:.1f}ms (runs: {[f'{t:.1f}' for t in times_hf]})")
        pv_hf = result_hf["pixel_values"]
        print(f"  输出: dtype={pv_hf.dtype}, shape={pv_hf.shape}")
    except Exception as e:
        print(f"  跳过 (导入失败): {e}")
        avg_hf = None

    # ── 路径 2: numexpr 优化路径 ──
    print("\n--- 路径 2: numexpr 优化路径 (patch_reshape_method='torch') ---")
    processor._preprocess_backend = "numexpr"
    # warmup
    _ = processor.preprocess(images_np[:4], return_tensors="pt", patch_reshape_method="torch")

    times_ne = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        result_ne = processor.preprocess(images_np, return_tensors="pt", patch_reshape_method="torch")
        t1 = time.perf_counter()
        times_ne.append((t1 - t0) * 1000)

    avg_ne = sum(times_ne) / len(times_ne)
    print(f"  平均耗时: {avg_ne:.1f}ms (runs: {[f'{t:.1f}' for t in times_ne]})")
    pv_ne = result_ne["pixel_values"]
    print(f"  输出: dtype={pv_ne.dtype}, shape={pv_ne.shape}")
    print(f"  数据大小: {pv_ne.numel() * pv_ne.element_size() / 1024 / 1024:.1f} MB")

    # ── 路径 3: torch_fused 优化路径 ──
    print("\n--- 路径 3: torch_fused 优化路径 ---")
    processor._preprocess_backend = "torch"
    # warmup
    _ = processor.preprocess(images_np[:4], return_tensors="pt", patch_reshape_method="torch")

    times_torch = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        result_torch = processor.preprocess(images_np, return_tensors="pt", patch_reshape_method="torch")
        t1 = time.perf_counter()
        times_torch.append((t1 - t0) * 1000)

    avg_torch = sum(times_torch) / len(times_torch)
    print(f"  平均耗时: {avg_torch:.1f}ms (runs: {[f'{t:.1f}' for t in times_torch]})")
    pv_torch = result_torch["pixel_values"]
    print(f"  输出: dtype={pv_torch.dtype}, shape={pv_torch.shape}")
    print(f"  数据大小: {pv_torch.numel() * pv_torch.element_size() / 1024 / 1024:.1f} MB")

    # ── 精度对比 ──
    print("\n" + "=" * 70)
    print("精度对比")
    print("=" * 70)

    # torch vs numexpr (都是fp16)
    if pv_torch.dtype == pv_ne.dtype:
        diff = (pv_torch.float() - pv_ne.float()).abs()
    else:
        diff = (pv_torch.float() - pv_ne.float()).abs()
    print(f"\n  torch_fused vs numexpr:")
    print(f"    max diff: {diff.max().item():.2e}, mean diff: {diff.mean().item():.2e}")

    # ── 总结 ──
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    if avg_hf:
        print(f"  原始 HF:      {avg_hf:.1f}ms")
    print(f"  numexpr:      {avg_ne:.1f}ms")
    print(f"  torch_fused:  {avg_torch:.1f}ms")
    if avg_hf:
        print(f"  加速比 (HF→torch_fused): {avg_hf / avg_torch:.1f}x")
    print(f"  加速比 (numexpr→torch_fused): {avg_ne / avg_torch:.2f}x")


if __name__ == "__main__":
    test_all_paths()
