from pathlib import Path

# Assume `mypath.py` lives in src/cogpy/utils/mypath.py
BASE_DIR = Path(__file__).resolve().parents[3]   # -> project root
assert (BASE_DIR / "src").exists(), f"Expected src/ directory at {BASE_DIR / 'src'}"
COGPY_DIR = BASE_DIR / "cogpy"
TEST_DIR = BASE_DIR / "tests"
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results"

def ensure_dirs(*dirsis ):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
