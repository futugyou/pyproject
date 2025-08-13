import os
import shutil

CACHE_DIRS = [
    ".venv",
    ".ruff_cache",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
]

ROOT = "."

for dirpath, dirnames, filenames in os.walk(ROOT):
    for cache_dir in CACHE_DIRS:
        if cache_dir in dirnames:
            full_path = os.path.join(dirpath, cache_dir)
            print(f"Removing: {full_path}")
            shutil.rmtree(full_path, ignore_errors=True)
