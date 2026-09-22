"""Diagnose CUDA/cuDNN/NCCL loading without a model checkpoint.

Run with the same Python and environment as the failing server:
    python scripts/diagnose_cudnn.py
    python scripts/diagnose_cudnn.py --nccl-only

The probe preserves BF16 Conv3d and reports runtime errors and loaded libraries.
It does not install packages, change library paths, or alter model code.
"""

import argparse
import ctypes
from datetime import timedelta
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


def loaded_cuda_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        name = Path(path).name
        if name.startswith(
            ("libcudnn", "libcudart", "libcuda.so", "libcublas", "libnccl")
        ):
            paths.add(path)
    return sorted(paths)


def report_libraries():
    print("\nLoaded CUDA/cuDNN/NCCL libraries:", flush=True)
    paths = loaded_cuda_libraries()
    for path in paths:
        print(path, flush=True)
        if "/stubs/" in path:
            print("  WARNING: a link-only stub is loaded at runtime", flush=True)
        if Path(path).name.startswith("libnccl.so"):
            try:
                library = ctypes.CDLL(path)
                library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
                library.ncclGetVersion.restype = ctypes.c_int
                version = ctypes.c_int()
                status = library.ncclGetVersion(ctypes.byref(version))
                print(
                    f"  Loaded library ncclGetVersion: status={status}, "
                    f"version={version.value}",
                    flush=True,
                )
            except Exception as error:
                print(f"  ncclGetVersion failed: {error}", flush=True)
        if shutil.which("readelf") and Path(path).name.startswith(
            ("libcudnn", "libnccl")
        ):
            result = subprocess.run(
                ["readelf", "-d", path], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if any(tag in line for tag in ("(NEEDED)", "(RPATH)", "(RUNPATH)")):
                    print("  " + line.strip(), flush=True)

    # Probe only runtimes already loaded by Torch/cuDNN, after the Conv3d test.
    # Doing this first could change initialization order and hide the failure.
    for path in paths:
        if not Path(path).name.startswith("libcudart.so"):
            continue
        print(f"\nCUDA runtime probe: {path}", flush=True)
        try:
            runtime = ctypes.CDLL(path)
            runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
            runtime.cudaGetErrorString.restype = ctypes.c_char_p
            for function_name in ("cudaRuntimeGetVersion", "cudaDriverGetVersion"):
                function = getattr(runtime, function_name)
                function.argtypes = [ctypes.POINTER(ctypes.c_int)]
                function.restype = ctypes.c_int
                version = ctypes.c_int()
                status = function(ctypes.byref(version))
                print(f"  {function_name}: status={status}, value={version.value}")
            runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
            runtime.cudaGetDeviceCount.restype = ctypes.c_int
            count = ctypes.c_int()
            status = runtime.cudaGetDeviceCount(ctypes.byref(count))
            message = runtime.cudaGetErrorString(status)
            print(
                f"  cudaGetDeviceCount: status={status}, count={count.value}, "
                f"message={message.decode() if message else None}",
                flush=True,
            )
        except Exception as error:
            print(f"  Probe failed: {error}", flush=True)


def probe_nccl(torch, device):
    import torch.distributed as dist

    print(f"Torch-reported NCCL version: {torch.cuda.nccl.version()}", flush=True)
    with tempfile.TemporaryDirectory(prefix="sglang-nccl-probe-") as directory:
        try:
            dist.init_process_group(
                backend="nccl",
                init_method=(Path(directory) / "rendezvous").as_uri(),
                rank=0,
                world_size=1,
                device_id=torch.device(device),
                timeout=timedelta(seconds=30),
            )
            value = torch.ones(1, device=device)
            dist.all_reduce(value)
            torch.cuda.synchronize()
            assert value.item() == 1
            print("NCCL single-rank initialization + all_reduce PASS", flush=True)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nccl-only", action="store_true")
    args = parser.parse_args()
    if args.nccl_only:
        os.environ.setdefault("NCCL_DEBUG", "INFO")
    print(f"Python: {sys.executable}", flush=True)
    for key in (
        "CONDA_PREFIX",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "CUDNN_LIB_CONFIG",
    ):
        print(f"{key}={os.environ.get(key, '<unset>')}", flush=True)
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name == "torch" or name.startswith(
            ("nvidia-cudnn", "nvidia-cuda-runtime", "nvidia-cublas", "nvidia-nccl")
        ):
            print(f"Package: {name}=={distribution.version}", flush=True)
            if name == "torch":
                for requirement in distribution.requires or []:
                    if requirement.startswith(
                        (
                            "nvidia-cudnn",
                            "nvidia-cuda-runtime",
                            "nvidia-cublas",
                            "nvidia-nccl",
                        )
                    ):
                        print(f"  Torch requires: {requirement}", flush=True)
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        print("nvidia-smi:\n" + result.stdout + result.stderr, flush=True)

    success = False
    try:
        import torch
        import torch.nn.functional as F

        print(
            f"Torch: {torch.__version__}, build CUDA: {torch.version.cuda}", flush=True
        )
        torch.cuda.set_device(args.device)
        device = f"cuda:{args.device}"
        if args.nccl_only:
            probe_nccl(torch, device)
            return 0
        with torch.inference_mode():
            # A small independent probe of BeeBee's patch Conv3d geometry.
            x = torch.randn(4, 3, 2, 16, 16, device=device, dtype=torch.bfloat16)
            weight = torch.randn(
                1152, 3, 2, 16, 16, device=device, dtype=torch.bfloat16
            )
            bias = torch.zeros(1152, device=device, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            print(f"GPU: {torch.cuda.get_device_name(args.device)}", flush=True)
            print(f"Free/total GPU bytes: {torch.cuda.mem_get_info()}", flush=True)
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}", flush=True)
            print(
                f"cuDNN runtime version: {torch.backends.cudnn.version()}", flush=True
            )
            y = F.conv3d(x, weight, bias, stride=(2, 16, 16))
            torch.cuda.synchronize()
            print(
                f"BF16 Conv3d PASS: shape={tuple(y.shape)}, dtype={y.dtype}", flush=True
            )
            success = True
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        report_libraries()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
