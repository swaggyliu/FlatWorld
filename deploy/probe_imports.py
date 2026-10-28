import importlib
import sys

print("python", sys.executable, sys.version)
for m in ("torch", "warp", "numpy", "fastapi", "uvicorn", "tensordict", "rsl_rl"):
    try:
        mod = importlib.import_module(m)
        print("OK", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("NO", m, type(e).__name__, e)
