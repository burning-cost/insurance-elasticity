# Databricks notebook source
# Runs the full test suite for insurance-elasticity.

# COMMAND ----------

import subprocess, sys, os

def pip_install(*args):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + list(args),
        capture_output=True, text=True
    )
    out = result.stdout[-2000:]
    err = result.stderr[-2000:]
    if result.returncode != 0:
        raise RuntimeError(f"pip failed: {args}\n{out}\n{err}")
    return out

print("Python:", sys.version)

# statsmodels on Databricks serverless has a bug in deprecate_kwarg (old version).
# econml 0.15.1 calls statsmodels internally. Upgrade statsmodels first.
pip_install("--upgrade", "statsmodels>=0.14")
print("statsmodels OK")
pip_install("econml==0.15.1")
print("econml OK")
pip_install("catboost>=1.2")
print("catboost OK")
pip_install("pandas>=2.0", "polars>=0.20")
print("pandas+polars OK")
pip_install("insurance-elasticity==0.1.1", "pytest")
print("insurance-elasticity OK")

# Show versions
result = subprocess.run([sys.executable, "-m", "pip", "show", "statsmodels", "econml", "pandas", "catboost"], capture_output=True, text=True)
print(result.stdout)

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

# COMMAND ----------

test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header", "-p", "no:warnings"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
full_out = test_result.stdout + "\n" + test_result.stderr
print(full_out[-12000:])

# COMMAND ----------

status = "PASSED" if test_result.returncode == 0 else "FAILED"
summary_lines = [l for l in full_out.split("\n") if any(k in l for k in ("passed", "failed", "FAILED", "ERROR"))]
summary = "\n".join(summary_lines[-20:])
dbutils.notebook.exit(f"{status} rc={test_result.returncode}\n\nSUMMARY:\n{summary}\n\nOUTPUT (last 3000):\n{full_out[-3000:]}")
