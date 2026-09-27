"""Check the current interpreter without downloading data or model weights."""

import importlib
from importlib.metadata import version
import platform
import shutil
import sys


PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "PIL": "Pillow",
    "cv2": "opencv-python", "skimage": "scikit-image",
    "sklearn": "scikit-learn", "scipy": "scipy", "joblib": "joblib",
    "yaml": "PyYAML", "matplotlib": "matplotlib", "seaborn": "seaborn",
    "fastapi": "fastapi", "uvicorn": "uvicorn", "pytest": "pytest", "torch": "torch",
    "torchvision": "torchvision", "anomalib.models": "anomalib",
}


def diagnose():
    report = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(),
        "disk_free_bytes": shutil.disk_usage(".").free,
        "imports": {}, "cpu": {"status": "unavailable"},
        "cuda": {"status": "unavailable"}, "ok": True,
    }
    modules = {}
    for name, distribution in PACKAGES.items():
        try:
            modules[name] = importlib.import_module(name)
            report["imports"][name] = {"version": version(distribution)}
        except Exception as error:
            report["imports"][name] = {"error": f"{type(error).__name__}: {error}"}
            report["ok"] = False
    if "torch" in modules:
        torch = modules["torch"]
        try:
            matrix = torch.ones((16, 16))
            torch.testing.assert_close(matrix @ matrix, torch.full((16, 16), 16.0))
            report["cpu"] = {"status": "passed"}
            report["cuda"]["runtime"] = torch.version.cuda
            if torch.cuda.is_available():
                matrix = matrix.to("cuda")
                torch.testing.assert_close(matrix @ matrix, torch.full((16, 16), 16.0, device="cuda"))
                torch.cuda.synchronize()
                gpu = torch.cuda.get_device_properties(0)
                report["cuda"].update(status="passed", name=gpu.name, memory_bytes=gpu.total_memory)
            else:
                report["cuda"]["reason"] = "CUDA unavailable; CPU tasks can continue."
        except Exception as error:
            report["ok"] = False
            report["tensor_error"] = f"{type(error).__name__}: {error}"
    return report
