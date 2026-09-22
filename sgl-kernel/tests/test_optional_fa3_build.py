"""CPU-only checks: python tests/test_optional_fa3_build.py -v."""

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


class OptionalFA3ImportTests(unittest.TestCase):
    def load_wrapper(self, extension=None, error=None):
        # Isolate the optional wrapper from the unrelated native common_ops
        # bootstrap so these checks work before building a wheel or on CPU CI.
        package = types.ModuleType("sgl_kernel")
        package.__path__ = [str(ROOT / "python/sgl_kernel")]
        modules = {"sgl_kernel": package, "sgl_kernel.flash_ops": extension}
        spec = importlib.util.spec_from_file_location(
            "_test_optional_flash_attn", ROOT / "python/sgl_kernel/flash_attn.py"
        )
        wrapper = importlib.util.module_from_spec(spec)
        real_import = importlib.import_module

        def import_extension(name):
            if name == "sgl_kernel.flash_ops" and error is not None:
                raise error
            return real_import(name)

        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"SGLANG_KERNEL_API_LOGLEVEL": "0"}),
            patch("importlib.import_module", side_effect=import_extension),
        ):
            spec.loader.exec_module(wrapper)
        return wrapper

    def test_missing_extension_imports_and_reports_unavailable(self):
        wrapper = self.load_wrapper()
        with patch.object(torch.cuda, "get_device_capability") as capability:
            self.assertFalse(wrapper.is_fa3_supported())
            capability.assert_not_called()

    def test_missing_extension_calls_fail_before_tensor_processing(self):
        wrapper = self.load_wrapper()
        calls = [
            (wrapper.flash_attn_with_kvcache, (None, None, None)),
            (wrapper.flash_attn_varlen_func, (None,) * 5),
            (wrapper.get_scheduler_metadata, (1, 1, 1, 1, 1, 128, None)),
        ]
        for func, args in calls:
            with self.subTest(function=func.__name__):
                with self.assertRaisesRegex(ImportError, "FA3 flash_ops was not built"):
                    func(*args)

    def test_present_extension_preserves_a100_support(self):
        extension = types.ModuleType("sgl_kernel.flash_ops")
        wrapper = self.load_wrapper(extension)
        self.assertIs(wrapper.flash_ops, extension)
        with (
            patch.object(torch.version, "cuda", "12.6"),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        ):
            self.assertTrue(wrapper.is_fa3_supported())

    def test_broken_extension_errors_are_not_hidden(self):
        errors = [
            ImportError("undefined symbol"),
            OSError("shared library load failed"),
            ModuleNotFoundError("missing dependency", name="another_dependency"),
        ]
        for error in errors:
            with self.subTest(error=error):
                with self.assertRaises(type(error)):
                    self.load_wrapper(error=error)


@unittest.skipUnless(shutil.which("cmake"), "cmake is required")
class OptionalFA3CMakeTests(unittest.TestCase):
    def test_options_and_reconfiguration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.26)\n"
                "project(optional_kernel_test NONE)\n"
                'option(ENABLE_BELOW_SM90 "SM80 support" OFF)\n'
                f'include("{ROOT / "cmake/optional_kernels.cmake"}")\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/result.txt" '
                '"${SGL_KERNEL_ENABLE_FA3};${ENABLE_BELOW_SM90}")\n'
            )

            def configure(directory, *flags):
                build = root / directory
                subprocess.run(
                    ["cmake", "-S", str(root), "-B", str(build), *flags],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return (build / "result.txt").read_text()

            self.assertEqual(configure("old", "-DCUDAToolkit_VERSION=12.3"), "OFF;OFF")
            self.assertEqual(configure("new", "-DCUDAToolkit_VERSION=12.6"), "ON;OFF")
            self.assertEqual(configure("new", "-DSGL_KERNEL_ENABLE_FA3=OFF"), "OFF;OFF")
            self.assertEqual(
                configure(
                    "a100",
                    "-DCUDAToolkit_VERSION=12.6",
                    "-DSGL_KERNEL_ENABLE_FA3=ON",
                    "-DSGL_KERNEL_A100_BUILD=ON",
                ),
                "OFF;ON",
            )
            self.assertEqual(configure("a100", "-DSGL_KERNEL_A100_BUILD=OFF"), "ON;OFF")


if __name__ == "__main__":
    unittest.main()
