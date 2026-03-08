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

# Install in a specific order to avoid dependency conflicts
pip_install("econml==0.15.1")
print("econml OK")
pip_install("catboost>=1.2")
print("catboost OK")
# Upgrade pandas and polars to ensure interop works
pip_install("pandas>=2.0", "polars>=0.20")
print("pandas+polars OK")
pip_install("insurance-elasticity==0.1.0", "pytest")
print("insurance-elasticity OK")

# Check versions
result = subprocess.run([sys.executable, "-m", "pip", "show", "pandas", "polars", "econml", "catboost"], capture_output=True, text=True)
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

# Quick smoke test: can we import and run to_pandas?
smoke = subprocess.run(
    [sys.executable, "-c",
     "import polars as pl, pandas as pd; "
     "from insurance_elasticity.data import make_renewal_data; "
     "df = make_renewal_data(n=100); "
     "df_pd = df.to_pandas(); "
     "print('Smoke test OK, pandas ver:', pd.__version__, 'polars ver:', pl.__version__)"],
    capture_output=True, text=True
)
print("STDOUT:", smoke.stdout)
print("STDERR:", smoke.stderr[-2000:])

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
summary = "\n".join(summary_lines[-25:])
dbutils.notebook.exit(f"{status} rc={test_result.returncode}\n\nSMOKE: {smoke.stdout.strip()}\n\nSUMMARY:\n{summary}\n\nOUTPUT (last 2500):\n{full_out[-2500:]}")
