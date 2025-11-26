import os
import json
import pickle  # Bandit will flag B301 on load/dump
from pathlib import Path

def load_cache_untrusted_pickle(pickle_path: str):
    """
    Intentionally vulnerable for SAST: pickle.load can execute arbitrary code
    if file is untrusted. Used only to surface a Medium (B301) finding.
    """
    p = Path(pickle_path); p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        data = {"note": "demo cache created", "version": 1}
        with open(p, "wb") as f:
            pickle.dump(data, f)  # B301
    with open(p, "rb") as f:
        return pickle.load(f)   # B301

def load_cache_json_safe(json_path: str):
    p = Path(json_path); p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        data = {"note": "demo cache created (json)", "version": 1}
        p.write_text(json.dumps(data), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))

def choose_cache_demo():
    use_unsafe = os.getenv("ALLOW_UNSAFE_PICKLE", "0") == "1"
    if use_unsafe:
        cache = load_cache_untrusted_pickle("reports/tmp_cache.pkl")
        print("[SAST-DEMO] UNSAFE pickle cache keys:", list(cache.keys()))
    else:
        cache = load_cache_json_safe("reports/tmp_cache.json")
        print("[SAST-DEMO] SAFE json cache keys:", list(cache.keys()))
