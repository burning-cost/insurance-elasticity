# Databricks notebook source
# Runs the full test suite for insurance-elasticity.
# Tests that require econml are skipped if econml cannot be installed.

# COMMAND ----------

import subprocess, sys, os
import platform

print("Python:", sys.version)
print("Platform:", platform.machine())

def safe_pip(*args, timeout=120):
    """Install a package; return True on success, False on failure."""
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + list(args),
        capture_output=True, text=True, timeout=timeout
    )
    out_short = (r.stdout + r.stderr)[-800:]
    if r.returncode == 0:
        print(f"OK: {' '.join(args[:2])}")
        return True
    else:
        print(f"FAILED ({r.returncode}): {' '.join(args[:2])}\n{out_short}")
        return False

# Check if econml is already present
r = subprocess.run([sys.executable, "-c", "import econml; print(econml.__version__)"],
                   capture_output=True, text=True)
econml_ok = r.returncode == 0
if econml_ok:
    print("econml already available:", r.stdout.strip())
else:
    print("econml not found, attempting install (will skip on failure)...")
    # Try --only-binary first to avoid source builds on aarch64
    econml_ok = safe_pip("econml>=0.15,<1.0", "--only-binary=:all:", timeout=60)
    if not econml_ok:
        print("No prebuilt econml wheel available; econml-dependent tests will be skipped.")

# Core deps (these have binary wheels)
safe_pip("statsmodels>=0.14", timeout=120)
safe_pip("catboost>=1.2", "--only-binary=:all:", timeout=120)
safe_pip("pandas>=2.0", "polars>=1.0", timeout=60)
safe_pip("pytest", "pytest-timeout", timeout=60)

# Show installed
r = subprocess.run([sys.executable, "-m", "pip", "show",
                    "econml", "catboost", "polars", "scikit-learn"],
                   capture_output=True, text=True)
print(r.stdout)

# COMMAND ----------

os.makedirs("/tmp/ie_repo", exist_ok=True)
if not os.path.exists("/tmp/ie_repo/tests"):
    clone = subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/burning-cost/insurance-elasticity.git",
         "/tmp/ie_repo"],
        capture_output=True, text=True, timeout=60
    )
    if clone.returncode != 0:
        raise RuntimeError(f"clone failed: {clone.stderr[-500:]}")
    print("Clone OK")
else:
    pull = subprocess.run(
        ["git", "pull", "--ff-only"],
        capture_output=True, text=True, cwd="/tmp/ie_repo", timeout=30
    )
    print("Pull:", (pull.stdout + pull.stderr).strip()[:200])

log = subprocess.run(["git", "log", "--oneline", "-3"],
                     capture_output=True, text=True, cwd="/tmp/ie_repo")
print("HEAD:", log.stdout.strip())

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/tmp/ie_repo"],
    capture_output=True, text=True, timeout=60
)
if r.returncode != 0:
    raise RuntimeError(f"Library install failed: {r.stderr[-500:]}")
print("Library installed OK")

# COMMAND ----------

test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short",
     "--no-header", "-p", "no:warnings"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
    timeout=600,
)
full_out = test_result.stdout + "\n" + test_result.stderr
print(full_out[-12000:])

# COMMAND ----------

status = "PASSED" if test_result.returncode == 0 else "FAILED"
summary_lines = [l for l in full_out.split("\n")
                 if any(k in l for k in ("passed", "failed", "FAILED", "ERROR", "skipped"))]
summary = "\n".join(summary_lines[-20:])
dbutils.notebook.exit(
    f"{status} rc={test_result.returncode}\n\nSUMMARY:\n{summary}\n\n"
    f"OUTPUT (last 3000):\n{full_out[-3000:]}"
)
