# Databricks notebook source
# Runs the full test suite for insurance-elasticity.

# COMMAND ----------

import subprocess, sys, os

def pip_install(*args):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + list(args),
        capture_output=True, text=True
    )
    out = result.stdout[-3000:]
    err = result.stderr[-2000:]
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install failed: {args}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        )
    return out

print("Python:", sys.version)
pip_install("econml==0.15.1")
print("econml OK")
pip_install("catboost>=1.2")
print("catboost OK")
pip_install("insurance-elasticity==0.1.0", "pytest")
print("insurance-elasticity OK")

# Check versions
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
relevant = [l for l in result.stdout.split("\n") if any(k in l.lower() for k in ("pandas", "econml", "polars", "catboost", "numpy", "scipy", "scikit"))]
print("Installed versions:")
print("\n".join(relevant))

# COMMAND ----------

os.makedirs("/tmp/ie_repo", exist_ok=True)
if not os.path.exists("/tmp/ie_repo/tests"):
    clone = subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/burning-cost/insurance-elasticity.git",
         "/tmp/ie_repo"],
        capture_output=True, text=True
    )
    if clone.returncode != 0:
        raise RuntimeError(f"clone failed: {clone.stderr}")

# First run just the fit tests with full error details
test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_fit.py", "-v", "--tb=long", "--no-header"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
full_out = test_result.stdout + "\n" + test_result.stderr
print(full_out[-12000:])

# COMMAND ----------

# Now run all tests
all_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=line", "--no-header", "-p", "no:warnings"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
all_out = all_result.stdout + "\n" + all_result.stderr
status = "PASSED" if all_result.returncode == 0 else "FAILED"
print(all_out[-10000:])

# COMMAND ----------

summary_lines = [l for l in all_out.split("\n") if any(k in l for k in ("passed", "failed", "FAILED", "ERROR"))]
summary = "\n".join(summary_lines[-20:])
dbutils.notebook.exit(f"{status} rc={all_result.returncode}\n\nFIT TEST OUTPUT (last 4000):\n{full_out[-4000:]}\n\nALL TESTS SUMMARY:\n{summary}")
